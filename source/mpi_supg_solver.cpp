#include "mpi_supg_solver.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SUPGFluidSolver<dim>::BlockIncompSchurPreconditioner::SchurComplementTpp::
      SchurComplementTpp(TimerOutput &timer2,
                         const std::vector<IndexSet> &owned_partitioning,
                         const PETScWrappers::MPI::BlockSparseMatrix &system,
                         const PETScWrappers::PreconditionBase &Pvvinv)
      : timer2(timer2), system_matrix(&system), Pvv_inverse(&Pvvinv)
    {
      dumb_vector.reinit(owned_partitioning,
                         system_matrix->get_mpi_communicator());
    }

    template <int dim>
    void SUPGFluidSolver<dim>::BlockIncompSchurPreconditioner::
      SchurComplementTpp::vmult(PETScWrappers::MPI::Vector &dst,
                                const PETScWrappers::MPI::Vector &src) const
    {
      // this is the exact representation of Tpp = App - Apv * Pvv * Avp.
      PETScWrappers::MPI::Vector tmp1(dumb_vector.block(0)),
        tmp2(dumb_vector.block(0)), tmp3(src);
      system_matrix->block(0, 1).vmult(tmp1, src);
      Pvv_inverse->vmult(tmp2, tmp1);
      system_matrix->block(1, 0).vmult(tmp3, tmp2);
      system_matrix->block(1, 1).vmult(dst, src);
      dst -= tmp3;
    }

    template <int dim>
    SUPGFluidSolver<dim>::BlockIncompSchurPreconditioner::
      BlockIncompSchurPreconditioner(
        TimerOutput &timer2,
        const std::vector<IndexSet> &owned_partitioning,
        const PETScWrappers::MPI::BlockSparseMatrix &system,
        PETScWrappers::MPI::SparseMatrix &absA,
        PETScWrappers::MPI::SparseMatrix &schur,
        PETScWrappers::MPI::SparseMatrix &B2pp)
      : timer2(timer2),
        system_matrix(&system),
        Abs_A_matrix(&absA),
        schur_matrix(&schur),
        B2pp_matrix(&B2pp),
        Tpp_itr(0)
    {
      // Initialize the Pvv inverse (the ILU(0) factorization of Avv)
      Pvv_inverse.initialize(system_matrix->block(0, 0));
      // Initialize Tpp
      Tpp.reset(new SchurComplementTpp(
        timer2, owned_partitioning, *system_matrix, Pvv_inverse));

      // Compute B2pp matrix App - Apv*rowsum(|Avv|)^(-1)*Avp
      // as the preconditioner to solve Tpp^-1
      PETScWrappers::MPI::BlockVector IdentityVector, RowSumAvv, ReverseRowSum;
      IdentityVector.reinit(owned_partitioning,
                            system_matrix->get_mpi_communicator());
      RowSumAvv.reinit(owned_partitioning,
                       system_matrix->get_mpi_communicator());
      ReverseRowSum.reinit(owned_partitioning,
                           system_matrix->get_mpi_communicator());
      // Want to set ReverseRowSum to 1 to calculate the Rowsum first
      IdentityVector.block(0) = 1;
      // iterate the Avv matrix to set everything to positive.
      Abs_A_matrix->add(1, system_matrix->block(0, 0));
      Abs_A_matrix->compress(VectorOperation::add);

      // local information of the matrix is in unit of row, so we want to know
      // the range of global row indices that the local rank has.
      unsigned int row_start = Abs_A_matrix->local_range().first;
      unsigned int row_end = Abs_A_matrix->local_range().second;
      unsigned int row_range = row_end - row_start;
      // A temporal vector to cache the columns and values to be set.
      std::vector<std::vector<unsigned int>> cache_columns;
      std::vector<std::vector<double>> cache_values;
      cache_columns.resize(row_range);
      cache_values.resize(row_range);
      for (auto r = Abs_A_matrix->local_range().first;
           r < Abs_A_matrix->local_range().second;
           ++r)
        {
          // Allocation of memory for the input values
          cache_columns[r - row_start].resize(Abs_A_matrix->row_length(r));
          cache_values[r - row_start].resize(Abs_A_matrix->row_length(r));
          unsigned int col_count = 0;
          auto itr = Abs_A_matrix->begin(r);
          while (col_count < Abs_A_matrix->row_length(r))
            {
              cache_columns[r - row_start].push_back(itr->column());
              cache_values[r - row_start].push_back(std::abs(itr->value()));
              ++col_count;
              if (col_count == Abs_A_matrix->row_length(r))
                break;
              ++itr;
            }
        }
      for (auto r = Abs_A_matrix->local_range().first;
           r < Abs_A_matrix->local_range().second;
           ++r)
        {
          Abs_A_matrix->set(
            r, cache_columns[r - row_start], cache_values[r - row_start], true);
        }
      Abs_A_matrix->compress(VectorOperation::insert);

      // Compute the diag vector rowsum(|Avv|)^(-1)
      Abs_A_matrix->vmult(RowSumAvv.block(0), IdentityVector.block(0));
      // Reverse the vector and store in ReverseRowSum
      std::vector<double> cache_vector(
        ReverseRowSum.block(0).locally_owned_size());
      std::vector<unsigned int> cache_rows(
        ReverseRowSum.block(0).locally_owned_size());
      for (auto r = ReverseRowSum.block(0).local_range().first;
           r < ReverseRowSum.block(0).local_range().second;
           ++r)
        {
          cache_vector.push_back(1 / (RowSumAvv.block(0)(r)));
          cache_rows.push_back(r);
        }
      ReverseRowSum.block(0).set(cache_rows, cache_vector);
      ReverseRowSum.compress(VectorOperation::insert);

      // Compute Schur matrix Apv*rowsum(|Avv|)^(-1)*Avp
      system_matrix->block(1, 0).mmult(
        *schur_matrix, system_matrix->block(0, 1), ReverseRowSum.block(0));
      // Add in numbers to B2pp
      B2pp_matrix->add(-1, *schur_matrix);
      B2pp_matrix->add(1, system_matrix->block(1, 1));
      B2pp_matrix->compress(VectorOperation::add);
      B2pp_inverse.initialize(*B2pp_matrix);
    }

    /**
     * The vmult operation strictly follows the definition of
     * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
     */
    template <int dim>
    void SUPGFluidSolver<dim>::BlockIncompSchurPreconditioner::vmult(
      PETScWrappers::MPI::BlockVector &dst,
      const PETScWrappers::MPI::BlockVector &src) const
    {
      // Compute the intermediate vector:
      //      |I           0|*|src(0)| = |src(0)|
      //      |-ApvPvv^-1  I| |src(1)|   |ptmp  |
      /////////////////////////////////////////
      PETScWrappers::MPI::Vector ptmp1(src.block(0)), ptmp(src.block(1));
      Pvv_inverse.vmult(ptmp1, src.block(0));
      this->Apv().vmult(ptmp, ptmp1);
      ptmp *= -1.0;
      ptmp += src.block(1);

      // Compute the final vector:
      //      |Pvv^-1     -Pvv^-1*Avp*Tpp^-1|*|src(0)|
      //      |0           Tpp^-1           | |ptmp  |
      //                        =   |Pvv^-1*src(0) - Pvv^-1*Avp*Tpp^-1*ptmp|
      //                            |Tpp^-1 * ptmp                         |
      //////////////////////////////////////////
      // Compute Tpp^-1 * ptmp first, which is equal to the problem Tpp*x = ptmp
      // Set up initial guess first
      {
        PETScWrappers::MPI::Vector c(ptmp), Sc(ptmp);
        Tpp->vmult(Sc, c);
        double alpha = (ptmp * c) / (Sc * c);
        c *= alpha;
        dst.block(1) = c;
      }
      // Compute the multiplication
      timer2.enter_subsection("Solving Tpp");
      SolverControl solver_control(
        ptmp.size(), 1e-3 * ptmp.l2_norm(), true, true);
      GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
      SolverGMRES<PETScWrappers::MPI::Vector> gmres(
        solver_control,
        vector_memory,
        SolverGMRES<PETScWrappers::MPI::Vector>::AdditionalData(200));
      gmres.solve(*Tpp, dst.block(1), ptmp, B2pp_inverse);
      // B2pp_inverse.vmult(dst.block(1), ptmp);
      // Count iterations for this solver solving Tpp inverse
      Tpp_itr += solver_control.last_step();

      timer2.leave_subsection("Solving Tpp");

      // Compute Pvv^-1*src(0) - Pvv^-1*Avp*dst(1)
      PETScWrappers::MPI::Vector utmp1(src.block(0)), utmp2(src.block(0));
      this->Avp().vmult(utmp1, dst.block(1));
      Pvv_inverse.vmult(utmp2, utmp1);
      Pvv_inverse.vmult(dst.block(0), src.block(0));
      dst.block(0) -= utmp2;
    }

    template <int dim>
    SUPGFluidSolver<dim>::SUPGFluidSolver(
      parallel::distributed::Triangulation<dim> &tria,
      const Parameters::AllParameters &parameters)
      : FluidSolver<dim>(tria, parameters)
    {
    }

    template <int dim>
    void SUPGFluidSolver<dim>::initialize_system()
    {
      preconditioner.reset();
      system_matrix.clear();
      Abs_A_matrix.clear();
      schur_matrix.clear();
      B2pp_matrix.clear();

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
      Abs_A_matrix.reinit(owned_partitioning[0],
                          owned_partitioning[0],
                          dsp.block(0, 0),
                          mpi_communicator);

      // Compute the sparsity pattern for mass schur in advance.
      // The only nonzero block is (1, 1), which is the same as \f$BB^T\f$.
      DynamicSparsityPattern schur_dsp(dofs_per_block[1], dofs_per_block[1]);
      schur_dsp.compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                      sparsity_pattern.block(0, 1));

      // Compute the pattern for B2pp perconditioner
      for (auto itr = sparsity_pattern.block(1, 1).begin();
           itr != sparsity_pattern.block(1, 1).end();
           ++itr)
        {
          schur_dsp.add(itr->row(), itr->column());
        }

      B2pp_matrix.reinit(owned_partitioning[1],
                         owned_partitioning[1],
                         schur_dsp,
                         mpi_communicator);
      schur_matrix.reinit(owned_partitioning[1],
                          owned_partitioning[1],
                          schur_dsp,
                          mpi_communicator);

      // present_solution is ghosted because it is used in the
      // output and mesh refinement functions.
      present_solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      solution_increment.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      
      fluid_previous_solution.reinit(owned_partitioning,
                                       relevant_partitioning,
                                       mpi_communicator);
      // newton_update is non-ghosted because the linear solver needs
      // a completely distributed vector.
      newton_update.reinit(owned_partitioning, mpi_communicator);
      // evaluation_point is ghosted because it is used in the assembly.
      evaluation_point.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      // system_rhs is non-ghosted because it is only used in the linear
      // solver and residual evaluation.
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      fsi_acceleration.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);


      int stress_vec_size = dim + dim * (dim - 1) * 0.5;
      fsi_stress = std::vector<Vector<double>>(
      stress_vec_size, Vector<double>(scalar_dof_handler.n_dofs()));

      // Cell property
      setup_cell_property();

      stress = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        dim,
        std::vector<PETScWrappers::MPI::Vector>(
          dim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));

      if (initial_condition_field)
        {
          apply_initial_condition();
        }
      if (turbulence_model)
        {
          turbulence_model->initialize_system();
        }
    }

    template <int dim>
    std::pair<unsigned int, double>
    SUPGFluidSolver<dim>::solve(const bool use_nonzero_constraints)
    {
      // This section includes the work done in the preconditioner
      // and GMRES solver.
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      preconditioner.reset(
        new BlockIncompSchurPreconditioner(timer2,
                                           owned_partitioning,
                                           system_matrix,
                                           Abs_A_matrix,
                                           schur_matrix,
                                           B2pp_matrix));

      SolverControl solver_control(
        system_matrix.m(), 1e-6 * system_rhs.l2_norm(), true);

      // Because PETScWrappers::SolverGMRES requires preconditioner derived
      // from PETScWrappers::PreconditionBase, we use dealii SolverFGMRES.
      GrowingVectorMemory<PETScWrappers::MPI::BlockVector> vector_memory;
      SolverFGMRES<PETScWrappers::MPI::BlockVector> gmres(solver_control,
                                                          vector_memory);

      // The solution vector must be non-ghosted
      gmres.solve(system_matrix, newton_update, system_rhs, *preconditioner);

      const AffineConstraints<double> &constraints_used =
        use_nonzero_constraints ? nonzero_constraints : zero_constraints;
      constraints_used.distribute(newton_update);

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void SUPGFluidSolver<dim>::run_one_step(bool apply_nonzero_constraints,
                                            bool assemble_system)
    {
      (void)assemble_system;
      std::cout.precision(6);
      std::cout.width(12);

      if (time.get_timestep() == 0)
        {
          output_results(0);
        }

      time.increment();
      pcout << std::string(96, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;

      // Resetting
      double current_residual = 1.0;
      double initial_residual = 1.0;
      double relative_residual = 1.0;
      unsigned int outer_iteration = 0;
      evaluation_point = present_solution;
      while (relative_residual > parameters.fluid_tolerance &&
             current_residual > 1e-14)
        {
          AssertThrow(outer_iteration < parameters.fluid_max_iterations,
                      ExcMessage("Too many Newton iterations!"));

          newton_update = 0;

          // Since evaluation_point changes at every iteration,
          // we have to reassemble both the lhs and rhs of the system
          // before solving it.
          // If the Dirichlet BCs are time-dependent, nonzero_constraints
          // should be applied at the first iteration of every time step;
          // if they are time-independent, nonzero_constraints should be
          // applied only at the first iteration of the first time step.
          assemble(apply_nonzero_constraints && outer_iteration == 0);
          auto state = solve(apply_nonzero_constraints && outer_iteration == 0);
          current_residual = system_rhs.l2_norm();

           //output for MPI testing
           /*
                if ( Utilities::MPI::this_mpi_process(mpi_communicator)== 0)
                {
                  std::ofstream file_rhs("rhs.txt",std::ios_base::app);

                  file_rhs << time.current() << "\t" << current_residual << std::endl;
                  
                  file_rhs.close();

                }
          */

          // Update evaluation_point. Since newton_update has been set to
          // the correct bc values, there is no need to distribute the
          // evaluation_point again. Note we have to use a non-ghosted
          // vector as a buffer in order to do addition.
          PETScWrappers::MPI::BlockVector tmp;
          tmp.reinit(owned_partitioning, mpi_communicator);
          tmp = evaluation_point;
          tmp += newton_update;
          evaluation_point = tmp;

          if (outer_iteration == 0)
            {
              initial_residual = current_residual;
            }
          relative_residual = current_residual / initial_residual;

          pcout << std::scientific << std::left << " ITR = " << std::setw(2)
                << outer_iteration << " ABS_RES = " << current_residual
                << " REL_RES = " << relative_residual
                << " GMRES_ITR = " << std::setw(3) << state.first
                << " GMRES_RES = " << state.second
                << " INNER_GMRES_ITR = " << std::setw(3)
                << preconditioner->get_Tpp_itr_count() << std::endl;
          outer_iteration++;
        }
      // Update solution increment, which is used in FSI application.
      PETScWrappers::MPI::BlockVector tmp1, tmp2;
      tmp1.reinit(owned_partitioning, mpi_communicator);
      tmp2.reinit(owned_partitioning, mpi_communicator);
      tmp1 = evaluation_point;
      tmp2 = present_solution;
      tmp2 -= tmp1;
      solution_increment = tmp2;
      // Newton iteration converges, update time and solution
      present_solution = evaluation_point;
      // Update stress for output
      update_stress();

      calculate_fluid_KE();
      calculate_fluid_PE();
       
      //fluid_previous_solution = present_solution;
      
      // Output
      if (time.time_to_output())
        {
          output_results(time.get_timestep());
        }
      // Save checkpoint
      if (parameters.simulation_type == "Fluid" && time.time_to_save())
        {
          save_checkpoint(time.get_timestep());
        }
      if (parameters.simulation_type == "Fluid" && time.time_to_refine())
        {
          refine_mesh(parameters.global_refinements[0],
                      parameters.global_refinements[0] + 3);
        }
    }

    template <int dim>
    void SUPGFluidSolver<dim>::run()
    {
      pcout << "Running with PETSc on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      // Try load from previous computation.
      bool success_load = load_checkpoint();
      if (!success_load)
        {
          if (!hard_coded_boundary_values.empty())
            {
              for (auto &bc : hard_coded_boundary_values)
                {
                  bc.second.advance_time(time.get_delta_t());
                }
            }
          triangulation.refine_global(parameters.global_refinements[0]);
          setup_dofs();
          make_constraints();
          initialize_system();
        }

      // Time loop.
      // use_nonzero_constraints is set to true only at the first time step,
      // which means nonzero_constraints will be applied at the first iteration
      // in the first time step only, and never be used again.
      // This corresponds to time-independent Dirichlet BCs.
      if (!success_load)
        {
          if (turbulence_model)
            {
              turbulence_model->run_one_step(true);
            }
          run_one_step(true);
        }
      while (time.end() - time.current() > 1e-12)
        {
          if (turbulence_model)
            {
              turbulence_model->run_one_step(false);
            }
          if (!hard_coded_boundary_values.empty())
            {
              // Only for time dependent BCs!
              // Advance the time by delta_t and make constraints
              for (auto &bc : hard_coded_boundary_values)
                {
                  bc.second.advance_time(time.get_delta_t());
                }
              make_constraints();
              run_one_step(true);
            }
          else
            {
              run_one_step(false);
            }
        }
    }
    template class SUPGFluidSolver<2>;
    template class SUPGFluidSolver<3>;
  } // namespace MPI
} // namespace Fluid

#include "mpi_insim.h"

namespace Fluid
{
  namespace MPI
  {
    /**
     * In serial code, we initialize the direct solver in the constructor
     * to avoid repeatedly allocating memory. However it seems we can't
     * do the same thing to the PETSc direct solver.
     */
    template <int dim>
    InsIM<dim>::BlockSchurPreconditioner::BlockSchurPreconditioner(
      TimerOutput &timer2,
      double gamma,
      double viscosity,
      double rho,
      double dt,
      const std::vector<IndexSet> &owned_partitioning,
      const PETScWrappers::MPI::BlockSparseMatrix &system,
      const PETScWrappers::MPI::BlockSparseMatrix &mass,
      PETScWrappers::MPI::BlockSparseMatrix &schur)
      : timer2(timer2),
        gamma(gamma),
        viscosity(viscosity),
        rho(rho),
        dt(dt),
        system_matrix(&system),
        mass_matrix(&mass),
        mass_schur(&schur),
        A_inverse(dummy_sc, system_matrix->get_mpi_communicator())
    {
      TimerOutput::Scope timer_section(timer2, "CG for Sm");
      // The sparsity pattern of mass_schur is already set,
      // we calculate its value in the following.
      PETScWrappers::MPI::BlockVector tmp1, tmp2;
      tmp1.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
      tmp2.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
      tmp1 = 1;
      tmp2 = 0;
      // Jacobi preconditioner of matrix A is by definition inverse diag(A),
      // this is exactly what we want to compute.
      // Note that the mass matrix and mass schur do not include the density.
      PETScWrappers::PreconditionJacobi jacobi(mass_matrix->block(0, 0));
      jacobi.vmult(tmp2.block(0), tmp1.block(0));
      // The sparsity pattern has already been set correctly, so explicitly
      // tell mmult not to rebuild the sparsity pattern.
      system_matrix->block(1, 0).mmult(
        mass_schur->block(1, 1), system_matrix->block(0, 1), tmp2.block(0));
    }

    /**
     * The vmult operation strictly follows the definition of
     * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
     */
    template <int dim>
    void InsIM<dim>::BlockSchurPreconditioner::vmult(
      PETScWrappers::MPI::BlockVector &dst,
      const PETScWrappers::MPI::BlockVector &src) const
    {
      // First, buffer the velocity block of src vector (\f$v_0\f$).
      PETScWrappers::MPI::Vector utmp(src.block(0));
      PETScWrappers::MPI::Vector tmp(src.block(1));
      tmp = 0;
      // This function is part of "solve linear system", but it
      // is further profiled to get a better idea of how time
      // is spent on different solvers.
      // The next two blocks computes \f$u_1 = \tilde{S}^{-1} v_1\f$.
      {
        TimerOutput::Scope timer_section(timer2, "CG for Mp");

        // CG solver used for \f$M_p^{-1}\f$ and \f$S_m^{-1}\f$.
        SolverControl solver_control(
          src.block(1).size(), std::max(1e-10, 1e-6 * src.block(1).l2_norm()));
        PETScWrappers::SolverCG cg_mp(solver_control,
                                      mass_schur->get_mpi_communicator());

        // \f$-(\mu + \gamma\rho)M_p^{-1}v_1\f$
        PETScWrappers::PreconditionNone Mp_preconditioner;
        Mp_preconditioner.initialize(mass_matrix->block(1, 1));
        cg_mp.solve(
          mass_matrix->block(1, 1), tmp, src.block(1), Mp_preconditioner);
        tmp *= -(viscosity + gamma * rho);
      }

      {
        TimerOutput::Scope timer_section(timer2, "CG for Sm");
        SolverControl solver_control(
          src.block(1).size(), std::max(1e-10, 1e-3 * src.block(1).l2_norm()));
        // FIXME: There is a mysterious bug here. After refine_mesh is called,
        // the initialization of Sm_preconditioner will complain about zero
        // entries on the diagonal which causes division by 0 since
        // PreconditionBlockJacobi uses ILU(0) underneath. This is similar to
        // the serial code where SparseILU is used. However, 1. if we do not use
        // a preconditioner here, the code runs fine, suggesting that mass_schur
        // is correct; 2. if we do not call refine_mesh, the code also runs
        // fine. So the question is, why would refine_mesh generate diagonal
        // zeros?
        //
        // \f$-\frac{1}{dt}S_m^{-1}v_1\f$
        PETScWrappers::PreconditionNone Sm_preconditioner;
        Sm_preconditioner.initialize(mass_schur->block(1, 1));
        PETScWrappers::SolverCG cg_sm(solver_control,
                                      mass_schur->get_mpi_communicator());
        cg_sm.solve(mass_schur->block(1, 1),
                    dst.block(1),
                    src.block(1),
                    Sm_preconditioner);
        dst.block(1) *= -rho / dt;
        // Adding up these two, we get \f$\tilde{S}^{-1}v_1\f$.
        dst.block(1) += tmp;
      }

      // This block computes \f$v_0 - B^T\tilde{S}^{-1}v_1\f$ based on
      // \f$u_1\f$.
      {
        system_matrix->block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp += src.block(0);
      }

      // Finally, compute the product of \f$\tilde{A}^{-1}\f$ and utmp with
      // the direct solver.
      {
        TimerOutput::Scope timer_section(timer2, "MUMPS for A_inv");
        A_inverse.solve(system_matrix->block(0, 0), dst.block(0), utmp);
      }
    }

    template <int dim>
    InsIM<dim>::InsIM(parallel::distributed::Triangulation<dim> &tria,
                      const Parameters::AllParameters &parameters)
      : FluidSolver<dim>(tria, parameters)
    {
      Assert(
        parameters.fluid_velocity_degree - parameters.fluid_pressure_degree ==
          1,
        ExcMessage(
          "Velocity finite element should be one order higher than pressure!"));
    }

    template <int dim>
    void InsIM<dim>::initialize_system()
    {
      FluidSolver<dim>::initialize_system();
      preconditioner.reset();
      newton_update.reinit(owned_partitioning, mpi_communicator);
      evaluation_point.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
    }

    template <int dim>
    void InsIM<dim>::assemble(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      const double viscosity = parameters.viscosity;
      const double gamma = parameters.grad_div;
      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        gravity[i] = parameters.gravity[i];

      system_matrix = 0;
      mass_matrix = 0;
      system_rhs = 0;

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);
      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_normal_vectors |
                                         update_quadrature_points |
                                         update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int u_dofs = fe.base_element(0).dofs_per_cell;
      const unsigned int p_dofs = fe.base_element(1).dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_face_q_points = face_quad_formula.size();

      AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
                  ExcMessage("Wrong partitioning of dofs!"));

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
      std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
      std::vector<double> current_pressure_values(n_q_points);
      std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
      std::vector<Tensor<1, dim>> fsi_acc_values(n_q_points);

      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);

              fe_values.reinit(cell);

              local_matrix = 0;
              local_mass_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                evaluation_point, current_velocity_values);

              fe_values[velocities].get_function_gradients(
                evaluation_point, current_velocity_gradients);

              fe_values[pressure].get_function_values(evaluation_point,
                                                      current_pressure_values);

              fe_values[velocities].get_function_values(
                present_solution, present_velocity_values);

              fe_values[velocities].get_function_values(fsi_acceleration,
                                                        fsi_acc_values);

              // Assemble the system matrix and mass matrix simultaneouly.
              // The mass matrix only uses the (0, 0) and (1, 1) blocks.
              //
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const int ind = p[0]->indicator;
                  const double rho = parameters.fluid_rho;
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          // Let the linearized diffusion, continuity and
                          // Grad-Div
                          // term be written as
                          // the bilinear operator: \f$A = a((\delta{u},
                          // \delta{p}), (\delta{v}, \delta{q}))\f$,
                          // the linearized convection term be: \f$C =
                          // c(u;\delta{u}, \delta{v})\f$,
                          // and the linearized inertial term be:
                          // \f$M = m(\delta{u}, \delta{v})$, then LHS is: $(A +
                          // C) + M/{\Delta{t}}\f$
                          local_matrix(i, j) +=
                            (viscosity *
                               scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                             current_velocity_gradients[q] * phi_u[j] *
                               phi_u[i] * rho +
                             grad_phi_u[j] * current_velocity_values[q] *
                               phi_u[i] * rho -
                             div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                             gamma * div_phi_u[j] * div_phi_u[i] * rho +
                             phi_u[i] * phi_u[j] / time.get_delta_t() * rho) *
                            fe_values.JxW(q);
                          local_mass_matrix(i, j) +=
                            (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j]) *
                            fe_values.JxW(q);
                        }

                      // RHS is \f$-(A_{current} + C_{current}) -
                      // M_{present-current}/\Delta{t}\f$.
                      double current_velocity_divergence =
                        trace(current_velocity_gradients[q]);
                      local_rhs(i) +=
                        ((-viscosity *
                            scalar_product(current_velocity_gradients[q],
                                           grad_phi_u[i]) -
                          current_velocity_gradients[q] *
                            current_velocity_values[q] * phi_u[i] * rho +
                          current_pressure_values[q] * div_phi_u[i] +
                          current_velocity_divergence * phi_p[i] -
                          gamma * current_velocity_divergence * div_phi_u[i] *
                            rho) -
                         (current_velocity_values[q] -
                          present_velocity_values[q]) *
                           phi_u[i] / time.get_delta_t() * rho +
                         gravity * phi_u[i] * rho) *
                        fe_values.JxW(q);
                      if (ind == 1)
                        {
                          local_rhs(i) +=
                            (scalar_product(grad_phi_u[i], p[0]->fsi_stress) +
                             (fsi_acc_values[q] * rho * phi_u[i])) *
                            fe_values.JxW(q);
                        }
                    }
                }

              // Impose pressure boundary here if specified, loop over faces on
              // the
              // cell
              // and apply pressure boundary conditions:
              // \f$\int_{\Gamma_n} -p\bold{n}d\Gamma\f$
              if (parameters.n_fluid_neumann_bcs != 0)
                {
                  for (unsigned int face_n = 0;
                       face_n < GeometryInfo<dim>::faces_per_cell;
                       ++face_n)
                    {
                      if (cell->at_boundary(face_n) &&
                          parameters.fluid_neumann_bcs.find(
                            cell->face(face_n)->boundary_id()) !=
                            parameters.fluid_neumann_bcs.end())
                        {
                          fe_face_values.reinit(cell, face_n);
                          unsigned int p_bc_id =
                            cell->face(face_n)->boundary_id();
                          double boundary_values_p =
                            parameters.fluid_neumann_bcs[p_bc_id];
                          for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                  local_rhs(i) += -(
                                    fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                                }
                            }
                        }
                    }
                }

              cell->get_dof_indices(local_dof_indices);

              const AffineConstraints<double> &constraints_used =
                use_nonzero_constraints ? nonzero_constraints
                                        : zero_constraints;
              constraints_used.distribute_local_to_global(local_matrix,
                                                          local_rhs,
                                                          local_dof_indices,
                                                          system_matrix,
                                                          system_rhs,
                                                          true);
              constraints_used.distribute_local_to_global(
                local_mass_matrix, local_dof_indices, mass_matrix);
            }
        }

      system_matrix.compress(VectorOperation::add);
      mass_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned int, double>
    InsIM<dim>::solve(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      preconditioner.reset(new BlockSchurPreconditioner(timer2,
                                                        parameters.grad_div,
                                                        parameters.viscosity,
                                                        parameters.fluid_rho,
                                                        time.get_delta_t(),
                                                        owned_partitioning,
                                                        system_matrix,
                                                        mass_matrix,
                                                        mass_schur));

      SolverControl solver_control(
        system_matrix.m(), std::max(1e-12, 1e-4 * system_rhs.l2_norm()), true);
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
    void InsIM<dim>::run_one_step(bool apply_nonzero_constraints,
                                  bool assemble_system)
    {
      static_cast<void>(assemble_system);
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
             current_residual > 1e-11)
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
                << " GMRES_RES = " << state.second << std::endl;

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
      // Output
      if (parameters.simulation_type == "Fluid" && time.time_to_save())
        {
          save_checkpoint(time.get_timestep());
        }
      if (time.time_to_output())
        {
          output_results(time.get_timestep());
        }
      if (parameters.simulation_type == "Fluid" && time.time_to_refine())
        {
          refine_mesh(parameters.global_refinements[0],
                      parameters.global_refinements[0] + 3);
        }
    }

    template <int dim>
    void InsIM<dim>::run()
    {
      pcout << "Running with PETSc on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      // Try load from previous computation
      bool success_load = load_checkpoint();
      if (!success_load)
        {
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
      run_one_step(true);
      while (time.end() - time.current() > 1e-12)
        {
          run_one_step(false);
        }
    }

    template class InsIM<2>;
    template class InsIM<3>;
  } // namespace MPI
} // namespace Fluid

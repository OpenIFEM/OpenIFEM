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
      zero_constraints.distribute(present_solution);
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
      FEValues<dim> scalar_fe_values(scalar_fe,
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

      std::vector<double> fsi_stress_value(n_q_points);

      std::vector<std::vector<double>> fsi_cell_stress =
        std::vector<std::vector<double>>(fsi_stress.size(),
                                         std::vector<double>(n_q_points));

      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      auto cell = dof_handler.begin_active();
      auto scalar_cell = scalar_dof_handler.begin_active();

      for (; cell != dof_handler.end(), scalar_cell != scalar_dof_handler.end();
           ++cell, ++scalar_cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);

              fe_values.reinit(cell);
              scalar_fe_values.reinit(scalar_cell);

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

              for (unsigned int i = 0; i < fsi_stress.size(); i++)
                {
                  scalar_fe_values.get_function_values(fsi_stress[i],
                                                       fsi_stress_value);

                  fsi_cell_stress[i] = fsi_stress_value;
                }

              // Assemble the system matrix and mass matrix simultaneouly.
              // The mass matrix only uses the (0, 0) and (1, 1) blocks.
              //
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const int ind = p[0]->indicator;
                  const double ind_exact = p[0]->exact_indicator;

                  const double rho = parameters.fluid_rho;

                  const double rho_bar = parameters.solid_rho * ind_exact +
                                         parameters.fluid_rho * (1 - ind_exact);

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                    }
                  SymmetricTensor<2, dim> fsi_stress_tensor;

                  if (ind != 0)
                    {
                      int stress_index = 0;
                      for (unsigned int k = 0; k < dim; k++)
                        {
                          for (unsigned int m = 0; m < k + 1; m++)
                            {
                              fsi_stress_tensor[k][m] =
                                fsi_cell_stress[stress_index][q];
                              stress_index++;
                            }
                        }
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
                            
                            /*
                            if (ind != 0)
                            {
                              double theta = parameters.penalty_scale_factor;
                              const double alpha = theta
                              * (parameters.solid_rho / time.get_delta_t());
                              local_matrix(i, j) += alpha * (phi_u[i] * phi_u[j]) * fe_values.JxW(q);
                            }*/

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
                      // if (ind == 1)
                      if (ind != 0)
                        {
                          local_rhs(i) +=
                            //(scalar_product(grad_phi_u[i], fsi_stress_tensor)
                            //+
                            (scalar_product(grad_phi_u[i], p[0]->fsi_stress) +
                             (fsi_acc_values[q] * rho_bar * phi_u[i])) *
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

      compute_fluid_norms();
      compute_energy_estimates();
      // Update stress for output
      update_stress();
      // compute_fluid_energy();
      previous_solution = present_solution;
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
          previous_solution = present_solution;
          compute_fluid_norms();
          compute_energy_estimates();
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

    template <int dim>
    void InsIM<dim>::compute_fluid_norms()
    {
      // Set up FEValues with necessary update flags
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_JxW_values);

      const FEValuesExtractors::Vector velocities(0);
      const unsigned int n_q_points = volume_quad_formula.size();

      // Local sums for velocity and divergence norms
      double local_sum_vel = 0.0;
      double local_sum_div = 0.0;

      // Vectors to store velocity and divergence values at quadrature points
      std::vector<Tensor<1, dim>> velocity_values(n_q_points);
      std::vector<double> divergence_values(n_q_points);

      // Loop over all locally owned cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {

              auto p = cell_property.get_data(cell);

              if (p[0]->indicator == 1)
                {
                  continue;
                }

              fe_values.reinit(cell);

              // Get velocity and divergence values from the solution
              fe_values[velocities].get_function_values(present_solution,
                                                        velocity_values);
              fe_values[velocities].get_function_divergences(present_solution,
                                                             divergence_values);

              // Compute contributions to the integrals
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // Velocity L2 norm: integrate |u|^2
                  double vel_norm_sq = velocity_values[q].norm_square();
                  local_sum_vel += vel_norm_sq * fe_values.JxW(q);

                  // Divergence L2 norm: integrate (div u)^2
                  double div_u = divergence_values[q];
                  local_sum_div += div_u * div_u * fe_values.JxW(q);
                }
            }
        }

      // Global reduction to sum contributions across all processes
      double global_sum_vel = 0.0;
      double global_sum_div = 0.0;
      MPI_Allreduce(&local_sum_vel,
                    &global_sum_vel,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);
      MPI_Allreduce(&local_sum_div,
                    &global_sum_div,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      // Compute the L2 norms
      double L2_norm_vel = std::sqrt(global_sum_vel);
      double L2_norm_div = std::sqrt(global_sum_div);

      // Write results to files from process 0
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream file_vel, file_div;

          // If first time step, write headers
          if (time.current() == 0)
            {
              file_vel.open("velocity_L2_norm.txt");
              file_vel << "Time\tL2_norm_velocity\n";
              file_div.open("divergence_L2_norm.txt");
              file_div << "Time\tL2_norm_divergence\n";
            }
          else
            {
              file_vel.open("velocity_L2_norm.txt", std::ios_base::app);
              file_div.open("divergence_L2_norm.txt", std::ios_base::app);
            }

          // Write current time and norms
          file_vel << time.current() << "\t" << L2_norm_vel << "\n";
          file_div << time.current() << "\t" << L2_norm_div << "\n";

          file_vel.close();
          file_div.close();
        }
    }

    template <int dim>
    void InsIM<dim>::compute_energy_estimates()
    {
      TimerOutput::Scope timer_section(timer, "Compute energy estimates");

      // Set up FEValues
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_JxW_values);

      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values);

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_face_q_points = face_quad_formula.size();

      // Local accumulators
      double local_ke = 0.0;      // Kinetic energy
      double local_visc = 0.0;    // Viscous dissipation
      double local_p_div_u = 0.0; // Pressure-divergence term
      // double local_boundary_work = 0.0;
      double local_boundary_work_inlet = 0.0;
      double local_boundary_work_outlet = 0.0;

      // Quadrature point data
      std::vector<Tensor<1, dim>> velocity_values(n_q_points);
      std::vector<SymmetricTensor<2, dim>> sym_grad_u(n_q_points);
      std::vector<double> pressure_values(n_q_points);
      std::vector<double> div_u_values(n_q_points);

      // Loop over cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (!cell->is_locally_owned())
            continue;

          auto p = cell_property.get_data(cell);
          if (p[0]->indicator == 1) // Skip solid regions if applicable
            continue;

          fe_values.reinit(cell);

          // Extract field values
          fe_values[velocities].get_function_values(present_solution,
                                                    velocity_values);
          fe_values[velocities].get_function_symmetric_gradients(
            present_solution, sym_grad_u);
          fe_values[pressure].get_function_values(present_solution,
                                                  pressure_values);
          fe_values[velocities].get_function_divergences(present_solution,
                                                         div_u_values);

          // Quadrature loop
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Kinetic energy: 1/2 * rho * |u|^2
              double u_sq = velocity_values[q].norm_square();
              local_ke += 0.5 * parameters.fluid_rho * u_sq * fe_values.JxW(q);

              // Viscous dissipation: 2 * mu * eps:eps
              double eps_eps =
                sym_grad_u[q] * sym_grad_u[q]; // Double contraction
              local_visc +=
                2.0 * parameters.viscosity * eps_eps * fe_values.JxW(q);

              // Pressure-divergence term (diagnostic)
              local_p_div_u +=
                pressure_values[q] * div_u_values[q] * fe_values.JxW(q);
            }

          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              if (cell->at_boundary(face_no))
                {
                  fe_face_values.reinit(cell, face_no);

                  // We need the velocity, gradient(velocity), and pressure on
                  // the face
                  std::vector<Tensor<1, dim>> face_velocity(n_face_q_points);
                  std::vector<Tensor<2, dim>> face_grad_u(n_face_q_points);
                  std::vector<double> face_pressure(n_face_q_points);

                  fe_face_values[velocities].get_function_values(
                    present_solution, face_velocity);
                  fe_face_values[velocities].get_function_gradients(
                    present_solution, face_grad_u);
                  fe_face_values[pressure].get_function_values(present_solution,
                                                               face_pressure);

                  for (unsigned int qf = 0; qf < n_face_q_points; ++qf)
                    {
                      const Tensor<1, dim> &u_face = face_velocity[qf];
                      const Tensor<1, dim> &n_face =
                        fe_face_values.normal_vector(qf);

                      SymmetricTensor<2, dim> symgrad_u_face;

                      for (unsigned int i = 0; i < dim; ++i)
                        {
                          for (unsigned int j = 0; j < dim; ++j)
                            {
                              symgrad_u_face[i][j] =
                                0.5 *
                                (face_grad_u[qf][i][j] + face_grad_u[qf][j][i]);
                            }
                        }

                      SymmetricTensor<2, dim> stress_face =
                        -face_pressure[qf] *
                          Physics::Elasticity::StandardTensors<dim>::I +
                        2.0 * parameters.viscosity * symgrad_u_face;

                      Tensor<1, dim> traction = stress_face * n_face;

                      double integrand = u_face * traction;

                      // record inlet and outlet seperately
                      const types::boundary_id b_id =
                        cell->face(face_no)->boundary_id();

                      if (b_id == 0) // Inlet boundary
                        {
                          local_boundary_work_inlet +=
                            integrand * fe_face_values.JxW(qf);
                        }
                      else if (b_id == 1)
                        {
                          local_boundary_work_outlet +=
                            integrand * fe_face_values.JxW(qf);
                        }
                      // local_boundary_work += integrand *
                      // fe_face_values.JxW(qf);
                    }
                }
            }
        }

      // Global reduction
      double global_kinetic_energy = 0.0;
      double global_viscous_energy = 0.0;
      double global_divergence_residual = 0.0;
      // double global_boundary_work = 0.0;
      double global_boundary_work_inlet = 0.0;
      double global_boundary_work_outlet = 0.0;

      MPI_Allreduce(&local_ke,
                    &global_kinetic_energy,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_visc,
                    &global_viscous_energy,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_p_div_u,
                    &global_divergence_residual,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_boundary_work_inlet,
                    &global_boundary_work_inlet,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_boundary_work_outlet,
                    &global_boundary_work_outlet,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      // Output results
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          // --- BEGIN CHANGES ---
          // Modify the output columns to include inlet and outlet separately
          std::ofstream file("energy_estimates.txt",
                             time.current() == 0 ? std::ios::out
                                                 : std::ios::app);

          if (time.current() == 0)
            {
              file
                << "Time\tKinetic_Energy\tViscous_Dissipation\tPressure_Div_"
                   "Term"
                << "\tBoundary_Work_Inlet\tBoundary_Work_Outlet\n"; // <<<
                                                                    // CHANGED
            }

          file << time.current() << "\t" << global_kinetic_energy << "\t"
               << global_viscous_energy << "\t" << global_divergence_residual
               << "\t" << global_boundary_work_inlet << "\t" // <<< CHANGED
               << global_boundary_work_outlet << "\n";       // <<< CHANGED

          file.close();
        }
    }

    template class InsIM<2>;
    template class InsIM<3>;
  } // namespace MPI
} // namespace Fluid

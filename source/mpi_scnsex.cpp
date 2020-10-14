#include "mpi_scnsex.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SCnsEX<dim>::SCnsEX(parallel::distributed::Triangulation<dim> &tria,
                        const Parameters::AllParameters &parameters)
      : FluidSolver<dim>(tria, parameters)
    {
      AssertThrow(parameters.fluid_velocity_degree ==
                    parameters.fluid_pressure_degree,
                  ExcMessage("Velocity degree must the same as pressure!"));
    }

    template <int dim>
    void SCnsEX<dim>::initialize_system()
    {
      system_matrix.clear();

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

      // present_solution is ghosted because it is used in the
      // output and mesh refinement functions.
      present_solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      // intermediate solution is non-ghosted because the linear solver needs
      // a completely distributed vector.
      intermediate_solution.reinit(owned_partitioning, mpi_communicator);
      // evaluation_point is ghosted because it is used in the assembly.
      evaluation_point.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      // system_rhs is non-ghosted because it is only used in the linear
      // solver and residual evaluation.
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      fsi_acceleration.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      // Cell property
      setup_cell_property();

      if (initial_condition_field)
        {
          apply_initial_condition();
        }

      // Setup local matrices
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              local_matrices.initialize(cell, 1);
              const std::vector<std::shared_ptr<FullMatrix<double>>> m =
                local_matrices.get_data(cell);
              *(m[0]) = 0;
            }
        }

      stress = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        dim,
        std::vector<PETScWrappers::MPI::Vector>(
          dim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));

      // Hard-coded initial condition, only for VF cases!
      // apply_initial_condition();
    }

    template <int dim>
    void
    SCnsEX<dim>::set_hard_coded_boundary_condition_time(const unsigned int id,
                                                        const double time)
    {
      AssertThrow(
        parameters.fluid_dirichlet_bcs.find(id) !=
          parameters.fluid_dirichlet_bcs.end(),
        ExcMessage("Hard coded BC ID not included in parameters file!"));
      AssertThrow(this->hard_coded_boundary_values.find(id) !=
                    this->hard_coded_boundary_values.end(),
                  ExcMessage("Bundary condition is not hard coded!"));
      boundary_condition_time_limits[id] = time;
    }

    template <int dim>
    void SCnsEX<dim>::assemble(const bool assemble_system,
                               const bool assemble_velocity)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        gravity[i] = parameters.gravity[i];

      if (assemble_system)
        {
          if (assemble_velocity)
            {
              system_matrix.block(0, 0) = 0;
            }
          else
            {
              system_matrix.block(1, 1) = 0;
            }
        }
      else if (assemble_velocity && !hard_coded_boundary_values.empty())
        {
          system_matrix.block(0, 0) = 0;
        }

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
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      // For the linearized system, we create temporary storage for current
      // velocity
      // and gradient, current pressure, and present velocity. In practice, they
      // are
      // all obtained through their shape functions at quadrature points.

      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
      std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
      std::vector<double> current_pressure_values(n_q_points);
      std::vector<Tensor<1, dim>> current_pressure_gradients(n_q_points);
      std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
      std::vector<double> present_pressure_values(n_q_points);
      std::vector<double> sigma_pml(n_q_points);
      std::vector<Tensor<1, dim>> artificial_bf(n_q_points);

      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);
      std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

      // The parameters that is used in isentropic continuity equation:
      // heat capacity ratio and atmospheric pressure.
      const double cp_to_cv = 1.4;
      const double atm = 1013250;

      // Zero out sigma field and body force if their fields are not specified
      if (sigma_pml_field == nullptr)
        {
          for (auto &e : sigma_pml)
            {
              e = 0.0;
            }
        }

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto m = local_matrices.get_data(cell);
              fe_values.reinit(cell);

              local_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                evaluation_point, current_velocity_values);

              fe_values[velocities].get_function_gradients(
                evaluation_point, current_velocity_gradients);

              fe_values[pressure].get_function_gradients(
                evaluation_point, current_pressure_gradients);

              if (assemble_velocity)
                {
                  fe_values[velocities].get_function_values(
                    present_solution, present_velocity_values);
                }
              else // assemble pressure
                {
                  fe_values[pressure].get_function_values(
                    present_solution, present_pressure_values);

                  fe_values[pressure].get_function_values(
                    evaluation_point, current_pressure_values);
                }

              if (sigma_pml_field)
                {
                  sigma_pml_field->double_value_list(
                    fe_values.get_quadrature_points(), sigma_pml, 0);
                }
              if (body_force)
                {
                  body_force->tensor_value_list(
                    fe_values.get_quadrature_points(), artificial_bf);
                }

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const double rho = parameters.fluid_rho;
                  const double viscosity = parameters.viscosity;

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                      grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      double current_velocity_divergence =
                        trace(current_velocity_gradients[q]);
                      if (assemble_system)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              if (assemble_velocity)
                                {
                                  local_matrix(i, j) +=
                                    (viscosity * scalar_product(grad_phi_u[j],
                                                                grad_phi_u[i]) +
                                     rho * phi_u[i] * phi_u[j] /
                                       time.get_delta_t()) *
                                    fe_values.JxW(q);
                                  // PML attenuation
                                  local_matrix(i, j) += rho * sigma_pml[q] *
                                                        phi_u[j] * phi_u[i] *
                                                        fe_values.JxW(q);
                                }
                              else
                                {
                                  local_matrix(i, j) += phi_p[i] * phi_p[j] /
                                                        time.get_delta_t() /
                                                        atm * fe_values.JxW(q);
                                  // PML attenuation
                                  local_matrix(i, j) += sigma_pml[q] *
                                                        phi_p[j] * phi_p[i] /
                                                        atm * fe_values.JxW(q);
                                }
                            }
                        }

                      if (assemble_velocity)
                        {
                          local_rhs(i) +=
                            (rho * (present_velocity_values[q] * phi_u[i] /
                                      time.get_delta_t() -
                                    current_velocity_gradients[q] *
                                      current_velocity_values[q] * phi_u[i]) -
                             phi_u[i] * current_pressure_gradients[q] +
                             (gravity + artificial_bf[q]) * phi_u[i] * rho) *
                            fe_values.JxW(q);
                        }
                      else
                        {
                          local_rhs(i) +=
                            (-cp_to_cv * (atm + current_pressure_values[q]) *
                               current_velocity_divergence * phi_p[i] +
                             present_pressure_values[q] * phi_p[i] /
                               time.get_delta_t() -
                             phi_p[i] * current_pressure_gradients[q] *
                               current_velocity_values[q]) /
                            atm * fe_values.JxW(q);
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

              if (assemble_system)
                {
                  nonzero_constraints.distribute_local_to_global(
                    local_matrix,
                    local_rhs,
                    local_dof_indices,
                    system_matrix,
                    system_rhs,
                    false);
                  if (assemble_velocity)
                    {
                      *(m[0]) = local_matrix;
                    }
                }
              else
                {
                  // *(m[0]) is the local matrix stored when the system
                  // matrix is assembled
                  if (assemble_velocity && !hard_coded_boundary_values.empty())
                    {
                      nonzero_constraints.distribute_local_to_global(
                        *(m[0]),
                        local_rhs,
                        local_dof_indices,
                        system_matrix,
                        system_rhs,
                        false);
                    }
                  else
                    {
                      nonzero_constraints.distribute_local_to_global(
                        local_rhs, local_dof_indices, system_rhs);
                    }
                }
            }
        }

      if (assemble_system ||
          (assemble_velocity && !hard_coded_boundary_values.empty()))
        {
          system_matrix.compress(VectorOperation::add);
        }
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned int, double>
    SCnsEX<dim>::solve(const bool solve_for_velocity)
    {
      // This section includes the work done in the CG solver
      TimerOutput::Scope timer_section(timer, "Solve linear system");

      SolverControl solver_control(
        system_matrix.m(), 1e-6 * system_rhs.l2_norm(), true);

      GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
      PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

      nonzero_constraints.set_zero(intermediate_solution);

      // The solution vector must be non-ghosted
      if (solve_for_velocity)
        {
          PETScWrappers::PreconditionBoomerAMG preconditioner(
            system_matrix.block(0, 0));
          cg.solve(system_matrix.block(0, 0),
                   intermediate_solution.block(0),
                   system_rhs.block(0),
                   preconditioner);
        }
      else
        {
          PETScWrappers::PreconditionBoomerAMG preconditioner(
            system_matrix.block(1, 1));
          cg.solve(system_matrix.block(1, 1),
                   intermediate_solution.block(1),
                   system_rhs.block(1),
                   preconditioner);
        }

      nonzero_constraints.distribute(intermediate_solution);

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void SCnsEX<dim>::run_one_step(bool apply_nonzero_constraints,
                                   bool assemble_system)
    {
      (void)apply_nonzero_constraints;
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

      PETScWrappers::MPI::BlockVector increment(owned_partitioning,
                                                mpi_communicator);
      PETScWrappers::MPI::BlockVector last_solution(owned_partitioning,
                                                    mpi_communicator);

      while (relative_residual > parameters.fluid_tolerance &&
             current_residual > 1e-12)
        {
          AssertThrow(outer_iteration < parameters.fluid_max_iterations,
                      ExcMessage("Too many iterations!"));

          intermediate_solution = 0;

          // Since evaluation_point changes at every iteration,
          // we have to reassemble both the lhs and rhs of the system
          // before solving it.
          // If the Dirichlet BCs are time-dependent, nonzero_constraints
          // should be applied at the first iteration of every time step;
          // if they are time-independent, nonzero_constraints should be
          // applied only at the first iteration of the first time step.
          assemble(assemble_system && outer_iteration == 0, true);
          auto state_velocity = solve(true);
          evaluation_point.block(0) = intermediate_solution.block(0);

          assemble(assemble_system && outer_iteration == 0, false);
          auto state_pressure = solve(false);
          evaluation_point.block(1) = intermediate_solution.block(1);

          increment = evaluation_point;
          increment -= last_solution;
          current_residual = increment.l2_norm();

          if (outer_iteration == 0)
            {
              initial_residual = intermediate_solution.l2_norm();
            }
          relative_residual = current_residual / initial_residual;

          pcout << std::scientific << std::left << " ITR = " << std::setw(2)
                << outer_iteration << " ABS_RES = " << current_residual
                << " REL_RES = " << relative_residual
                << " VEL_ITR = " << std::setw(3) << state_velocity.first
                << " PRE_ITR = " << std::setw(3) << state_pressure.first
                << " VEL_RES = " << state_velocity.second
                << " PRE_RES = " << state_pressure.second << std::endl;
          outer_iteration++;
          last_solution = intermediate_solution;
        }
      // Newton iteration converges, update time and solution
      present_solution = evaluation_point;
      // Update stress for output
      update_stress();
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
    void SCnsEX<dim>::run()
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
      while (time.end() - time.current() > 1e-12)
        {
          for (auto b = boundary_condition_time_limits.begin();
               b != boundary_condition_time_limits.end();
               ++b)
            {
              if (b->second < time.current())
                {
                  hard_coded_boundary_values.erase(b->first);
                  b = boundary_condition_time_limits.erase(b);
                  if (b == boundary_condition_time_limits.end())
                    {
                      break;
                    }
                }
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
            }
          run_one_step(true, time.get_timestep() < 1 || success_load);
          success_load = false;
        }
    }
    template class SCnsEX<2>;
    template class SCnsEX<3>;
  } // namespace MPI
} // namespace Fluid

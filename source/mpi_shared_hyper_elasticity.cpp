#include "mpi_shared_hyper_elasticity.h"

namespace Internal
{
  using namespace dealii;

  template <int dim>
  void PointHistory<dim>::setup(const Parameters::AllParameters &parameters,
                                const unsigned int &mat_id)
  {
    if (parameters.solid_type == "NeoHookean")
      {
        Assert(parameters.C[mat_id - 1].size() >= 2, ExcInternalError());
        material.reset(new Solid::NeoHookean<dim>(parameters.C[mat_id - 1][0],
                                                  parameters.C[mat_id - 1][1],
                                                  parameters.solid_rho));
        update(parameters, Tensor<2, dim>());
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void PointHistory<dim>::update(const Parameters::AllParameters &parameters,
                                 const Tensor<2, dim> &Grad_u)
  {
    const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u);
    material->update_data(F);
    F_inv = invert(F);
    if (parameters.solid_type == "NeoHookean")
      {
        auto nh = std::dynamic_pointer_cast<Solid::NeoHookean<dim>>(material);
        Assert(nh, ExcInternalError());
        tau = nh->get_tau();
        Jc = nh->get_Jc();
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
    dPsi_vol_dJ = material->get_dPsi_vol_dJ();
    d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
  }
} // namespace Internal

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    template <int dim>
    SharedHyperElasticity<dim>::SharedHyperElasticity(
      Triangulation<dim> &tria, const Parameters::AllParameters &params)
      : SharedSolidSolver<dim>(tria, params)
    {
    }

    template <int dim>
    void SharedHyperElasticity<dim>::run_one_step(bool first_step)
    {
      double gamma = 0.5 + parameters.damping;
      double beta = gamma / 2;

      if (first_step)
        {
          // Solve for the initial acceleration
          assemble_system(true);
          this->solve(mass_matrix, previous_acceleration, system_rhs);
          this->output_results(time.get_timestep());
        }

      PETScWrappers::MPI::Vector predicted_displacement(current_displacement);
      PETScWrappers::MPI::Vector newton_update(current_displacement);
      PETScWrappers::MPI::Vector tmp(current_displacement);

      time.increment();

      pcout << std::endl
            << "Timestep " << time.get_timestep() << " @ " << time.current()
            << "s" << std::endl;

      // Reset the errors, iteration counter, and the solution increment
      newton_update = 0;
      unsigned int newton_iteration = 0;
      error_residual = 1.0;
      initial_error_residual = 1.0;
      normalized_error_residual = 1.0;
      error_update = 1.0;
      initial_error_update = 1.0;
      normalized_error_update = 1.0;
      const double dt = time.get_delta_t();

      // The prediction of the current displacement,
      // which is what we want to solve.
      predicted_displacement = previous_displacement;
      predicted_displacement.add(
        dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);

      pcout << std::string(100, '_') << std::endl;

      while (normalized_error_update > parameters.tol_d ||
             normalized_error_residual > parameters.tol_f)
        {
          AssertThrow(newton_iteration < parameters.solid_max_iterations,
                      ExcMessage("Too many Newton iterations!"));

          // Compute the displacement, velocity and acceleration
          current_acceleration = current_displacement;
          current_acceleration -= predicted_displacement;
          current_acceleration /= (beta * dt * dt);
          current_velocity = previous_velocity;
          current_velocity.add(dt * (1 - gamma),
                               previous_acceleration,
                               dt * gamma,
                               current_acceleration);

          // Assemble the system, and modify the RHS to account for
          // the time-discretization.
          assemble_system(false);
          mass_matrix.vmult(tmp, current_acceleration);
          system_rhs -= tmp;

          // Solve linear system
          const std::pair<unsigned int, double> lin_solver_output =
            this->solve(system_matrix, newton_update, system_rhs);

          // Error evaluation
          {
            // We should rule out the constrained components before evaluating
            // the norms of system_rhs and newton_update.
            error_residual = get_error(system_rhs);
            if (newton_iteration == 0)
              {
                initial_error_residual = error_residual;
              }
            normalized_error_residual = error_residual / initial_error_residual;

            error_update = get_error(newton_update);
            if (newton_iteration == 0)
              {
                initial_error_update = error_update;
              }
            normalized_error_update = error_update / initial_error_update;
          }

          current_displacement += newton_update;
          // Update the quadrature point history with the newest displacement
          update_qph(current_displacement);

          pcout << "Newton iteration = " << newton_iteration
                << ", CG itr = " << lin_solver_output.first << std::fixed
                << std::setprecision(3) << std::setw(7) << std::scientific
                << ", CG res = " << lin_solver_output.second
                << ", res_F = " << error_residual
                << ", res_U = " << error_update << std::endl;

          newton_iteration++;
        }

      // Once converged, update current acceleration and velocity again.
      current_acceleration = current_displacement;
      current_acceleration -= predicted_displacement;
      current_acceleration /= (beta * dt * dt);
      current_velocity = previous_velocity;
      current_velocity.add(dt * (1 - gamma),
                           previous_acceleration,
                           dt * gamma,
                           current_acceleration);
      // Update the previous values
      previous_acceleration = current_acceleration;
      previous_velocity = current_velocity;
      previous_displacement = current_displacement;

      pcout << std::string(100, '_') << std::endl
            << "Relative errors:" << std::endl
            << "Displacement:\t" << normalized_error_update << std::endl
            << "Force: \t\t" << normalized_error_residual << std::endl;

      // strain and stress
      update_strain_and_stress();

      if (time.time_to_output())
        {
          this->output_results(time.get_timestep());
        }
      if (time.time_to_save())
        {
          this->save_checkpoint(time.get_timestep());
        }
    }

    template <int dim>
    void SharedHyperElasticity<dim>::initialize_system()
    {
      SharedSolidSolver<dim>::initialize_system();
      setup_qph();
    }

    template <int dim>
    void SharedHyperElasticity<dim>::setup_qph()
    {
      const unsigned int n_q_points = volume_quad_formula.size();
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;
          unsigned int mat_id = cell->material_id();
          if (parameters.n_solid_parts == 1)
            mat_id = 1;
          quad_point_history.initialize(cell, n_q_points);
          const std::vector<std::shared_ptr<Internal::PointHistory<dim>>> lqph =
            quad_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              lqph[q]->setup(parameters, mat_id);
            }
        }
    }

    template <int dim>
    void SharedHyperElasticity<dim>::update_qph(
      const PETScWrappers::MPI::Vector &evaluation_point)
    {
      timer.enter_subsection("Update QPH data");

      // displacement gradient at quad points
      const unsigned int n_q_points = volume_quad_formula.size();
      FEValuesExtractors::Vector displacement(0);
      std::vector<Tensor<2, dim>> grad_u(volume_quad_formula.size());
      FEValues<dim> fe_values(
        fe, volume_quad_formula, update_values | update_gradients);

      Vector<double> tmp(evaluation_point);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;
          const std::vector<std::shared_ptr<Internal::PointHistory<dim>>> lqph =
            quad_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());

          fe_values.reinit(cell);
          fe_values[displacement].get_function_gradients(tmp, grad_u);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              lqph[q]->update(parameters, grad_u[q]);
            }
        }
      timer.leave_subsection();
    }

    template <int dim>
    double SharedHyperElasticity<dim>::compute_volume() const
    {
      const unsigned int n_q_points = volume_quad_formula.size();
      double volume = 0.0;
      FEValues<dim> fe_values(fe, volume_quad_formula, update_JxW_values);
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;
          fe_values.reinit(cell);
          const std::vector<std::shared_ptr<const Internal::PointHistory<dim>>>
            lqph = quad_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double det = lqph[q]->get_det_F();
              const double JxW = fe_values.JxW(q);
              volume += det * JxW;
            }
        }
      // Equivalent to MPI::Allreduce
      Utilities::MPI::sum(volume, mpi_communicator);
      Assert(volume > 0.0, ExcInternalError());
      return volume;
    }

    template <int dim>
    double SharedHyperElasticity<dim>::get_error(
      const PETScWrappers::MPI::Vector &v) const
    {
      Vector<double> tmp(v);
      constraints.distribute(tmp);
      return tmp.l2_norm();
    }

    template <int dim>
    void SharedHyperElasticity<dim>::assemble_system(bool initial_step)
    {
      timer.enter_subsection("Assemble tangent matrix");

      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_f_q_points = face_quad_formula.size();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      FEValuesExtractors::Vector displacement(0);
      double gamma = 0.5 + parameters.damping;
      double beta = gamma / 2;

      if (initial_step)
        {
          mass_matrix = 0.0;
        }
      system_matrix = 0.0;
      system_rhs = 0.0;

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_JxW_values);
      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_normal_vectors |
                                         update_JxW_values);

      std::vector<std::vector<Tensor<1, dim>>> phi(
        n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
      std::vector<std::vector<Tensor<2, dim>>> grad_phi(
        n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
      std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi(
        n_q_points, std::vector<SymmetricTensor<2, dim>>(dofs_per_cell));

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        {
          gravity[i] = parameters.gravity[i];
        }

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;

          auto p = cell_property.get_data(cell);
          Assert(p.size() == n_f_q_points * GeometryInfo<dim>::faces_per_cell,
                 ExcMessage("Wrong number of cell data!"));
          fe_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);

          local_mass = 0;
          local_matrix = 0;
          local_rhs = 0;

          const std::vector<std::shared_ptr<Internal::PointHistory<dim>>> lqph =
            quad_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const Tensor<2, dim> F_inv = lqph[q]->get_F_inv();
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi[q][k] = fe_values[displacement].value(k, q);
                  grad_phi[q][k] =
                    fe_values[displacement].gradient(k, q) * F_inv;
                  sym_grad_phi[q][k] = symmetrize(grad_phi[q][k]);
                }

              const SymmetricTensor<2, dim> tau = lqph[q]->get_tau();
              const SymmetricTensor<4, dim> Jc = lqph[q]->get_Jc();
              const double rho = lqph[q]->get_density();
              const double dt = time.get_delta_t();
              const double JxW = fe_values.JxW(q);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  for (unsigned int j = 0; j <= i; ++j)
                    {
                      if (initial_step)
                        {
                          local_mass(i, j) += rho * phi[q][i] * phi[q][j] * JxW;
                        }
                      else
                        {
                          const unsigned int component_j =
                            fe.system_to_component_index(j).first;
                          local_matrix(i, j) +=
                            (phi[q][i] * phi[q][j] * rho / (beta * dt * dt) +
                             sym_grad_phi[q][i] * Jc * sym_grad_phi[q][j]) *
                            JxW;
                          if (component_i == component_j)
                            {
                              local_matrix(i, j) +=
                                grad_phi[q][i][component_i] * tau *
                                grad_phi[q][j][component_j] * JxW;
                            }
                        }
                    }
                  local_rhs(i) -=
                    sym_grad_phi[q][i] * tau * JxW; // -internal force
                  // body force
                  local_rhs[i] += phi[q][i] * gravity * rho * fe_values.JxW(q);
                }
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                {
                  local_matrix(i, j) = local_matrix(j, i);
                  if (initial_step)
                    {
                      local_mass(i, j) = local_mass(j, i);
                    }
                }
            }

          // Neumann boundary conditions
          // If this is a stand-alone solid simulation, the Neumann boundary
          // type should be either Traction or Pressure; it this is a FSI
          // simulation, the Neumann boundary type must be FSI.

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            {
              unsigned int id = cell->face(face)->boundary_id();

              if (!cell->face(face)->at_boundary())
                {
                  // Not a Neumann boundary
                  continue;
                }

              if (parameters.simulation_type != "FSI" &&
                  parameters.solid_neumann_bcs.find(id) ==
                    parameters.solid_neumann_bcs.end())
                {
                  // Traction-free boundary, do nothing
                  continue;
                }

              fe_face_values.reinit(cell, face);

              Tensor<1, dim> traction;
              std::vector<double> prescribed_value;
              if (parameters.simulation_type != "FSI")
                {
                  // In stand-alone simulation, the boundary value is prescribed
                  // by the user.
                  prescribed_value = parameters.solid_neumann_bcs[id];
                }

              if (parameters.simulation_type != "FSI" &&
                  parameters.solid_neumann_bc_type == "Traction")
                {
                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      traction[i] = prescribed_value[i];
                    }
                }

              for (unsigned int q = 0; q < n_f_q_points; ++q)
                {
                  if (parameters.simulation_type != "FSI" &&
                      parameters.solid_neumann_bc_type == "Pressure")
                    {
                      // The normal is w.r.t. reference configuration!
                      traction = fe_face_values.normal_vector(q);
                      traction *= prescribed_value[0];
                    }
                  else if (parameters.simulation_type == "FSI")
                    {
                      traction = p[face * n_f_q_points + q]->fsi_traction;
                    }

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      const unsigned int component_j =
                        fe.system_to_component_index(j).first;
                      // +external force
                      local_rhs(j) += fe_face_values.shape_value(j, q) *
                                      traction[component_j] *
                                      fe_face_values.JxW(q);
                    }
                }
            }

          if (initial_step)
            {
              constraints.distribute_local_to_global(local_mass,
                                                     local_rhs,
                                                     local_dof_indices,
                                                     mass_matrix,
                                                     system_rhs);
            }
          else
            {
              constraints.distribute_local_to_global(local_matrix,
                                                     local_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
            }
        }

      if (initial_step)
        {
          mass_matrix.compress(VectorOperation::add);
        }
      else
        {
          system_matrix.compress(VectorOperation::add);
        }
      system_rhs.compress(VectorOperation::add);
      timer.leave_subsection();
    }

    template <int dim>
    void SharedHyperElasticity<dim>::update_strain_and_stress()
    {
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            {
              strain[i][j] = 0.0;
              stress[i][j] = 0.0;
            }
        }
      PETScWrappers::MPI::Vector surrounding_cells(locally_owned_scalar_dofs,
                                                   mpi_communicator);
      surrounding_cells = 0.0;
      // The strain and stress tensors are stored as 2D vectors of shape dim*dim
      // at cell and quadrature point level.
      std::vector<std::vector<Vector<double>>> cell_strain(
        dim,
        std::vector<Vector<double>>(dim,
                                    Vector<double>(scalar_fe.dofs_per_cell)));
      std::vector<std::vector<Vector<double>>> cell_stress(
        dim,
        std::vector<Vector<double>>(dim,
                                    Vector<double>(scalar_fe.dofs_per_cell)));
      std::vector<std::vector<Vector<double>>> quad_strain(
        dim,
        std::vector<Vector<double>>(
          dim, Vector<double>(volume_quad_formula.size())));
      std::vector<std::vector<Vector<double>>> quad_stress(
        dim,
        std::vector<Vector<double>>(
          dim, Vector<double>(volume_quad_formula.size())));

      // Displacement gradients at quadrature points.
      std::vector<Tensor<2, dim>> current_displacement_gradients(
        volume_quad_formula.size());

      // The projection matrix from quadrature points to the dofs.
      FullMatrix<double> qpt_to_dof(scalar_fe.dofs_per_cell,
                                    volume_quad_formula.size());
      FETools::compute_projection_from_quadrature_points_matrix(
        scalar_fe, volume_quad_formula, volume_quad_formula, qpt_to_dof);

      const FEValuesExtractors::Vector displacements(0);

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      auto cell = dof_handler.begin_active();
      auto scalar_cell = scalar_dof_handler.begin_active();
      Vector<double> local_sorrounding_cells(scalar_fe.dofs_per_cell);
      local_sorrounding_cells = 1.0;

      Vector<double> localized_current_displacement(current_displacement);

      for (; cell != dof_handler.end(); ++cell, ++scalar_cell)
        {
          if (cell->subdomain_id() == this_mpi_process)
            {
              fe_values.reinit(cell);
              const std::vector<std::shared_ptr<Internal::PointHistory<dim>>>
                lqph = quad_point_history.get_data(cell);

              for (unsigned int q = 0; q < volume_quad_formula.size(); ++q)
                {
                  const SymmetricTensor<2, dim> tau = lqph[q]->get_tau();
                  const Tensor<2, dim> F = invert(lqph[q]->get_F_inv());
                  const double J = lqph[q]->get_det_F();
                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          quad_strain[i][j][q] = F[i][j];
                          quad_stress[i][j][q] = tau[i][j] / J;
                        }
                    }
                }

              for (unsigned int i = 0; i < dim; ++i)
                {
                  for (unsigned int j = 0; j < dim; ++j)
                    {
                      qpt_to_dof.vmult(cell_strain[i][j], quad_strain[i][j]);
                      qpt_to_dof.vmult(cell_stress[i][j], quad_stress[i][j]);
                      scalar_cell->distribute_local_to_global(cell_strain[i][j],
                                                              strain[i][j]);
                      scalar_cell->distribute_local_to_global(cell_stress[i][j],
                                                              stress[i][j]);
                    }
                }
              scalar_cell->distribute_local_to_global(local_sorrounding_cells,
                                                      surrounding_cells);
            }
        }
      surrounding_cells.compress(VectorOperation::add);

      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            {
              strain[i][j].compress(VectorOperation::add);
              stress[i][j].compress(VectorOperation::add);
              const unsigned int local_begin =
                surrounding_cells.local_range().first;
              const unsigned int local_end =
                surrounding_cells.local_range().second;
              for (unsigned int k = local_begin; k < local_end; ++k)
                {
                  strain[i][j][k] /= surrounding_cells[k];
                  stress[i][j][k] /= surrounding_cells[k];
                }
              strain[i][j].compress(VectorOperation::insert);
              stress[i][j].compress(VectorOperation::insert);
            }
        }
    }

    template class SharedHyperElasticity<2>;
    template class SharedHyperElasticity<3>;
  } // namespace MPI
} // namespace Solid

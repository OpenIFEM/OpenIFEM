#include "mpi_shared_linear_elasticity.h"

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    template <int dim>
    SharedLinearElasticity<dim>::SharedLinearElasticity(
      Triangulation<dim> &tria, const Parameters::AllParameters &parameters)
      : SharedSolidSolver<dim>(tria, parameters)
    {
      material.resize(parameters.n_solid_parts, LinearElasticMaterial<dim>());
      for (unsigned int i = 0; i < parameters.n_solid_parts; ++i)
        {
          LinearElasticMaterial<dim> tmp(parameters.E[i],
                                         parameters.nu[i],
                                         parameters.solid_rho,
                                         parameters.eta[i]);
          material[i] = tmp;
        }
    }

    template <int dim>
    void SharedLinearElasticity<dim>::assemble_system(const bool is_initial)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      double alpha = -parameters.damping;
      double gamma = 0.5 - alpha;
      double beta = pow((1 + alpha), 2) / 4;

      if (is_initial)
        {
          mass_matrix = 0;
          system_matrix = 0;
          stiffness_matrix = 0;
          damping_matrix = 0;
        }
      system_rhs = 0;

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

      FEFaceValues<dim> fe_face_values(
        fe,
        face_quad_formula,
        update_values | update_quadrature_points | update_normal_vectors |
          update_JxW_values);

      SymmetricTensor<4, dim> elasticity;
      SymmetricTensor<4, dim> viscosity;
      const double rho = material[0].get_density();
      const double dt = time.get_delta_t();

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_f_q_points = face_quad_formula.size();

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_stiffness(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_damping(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      Vector<double> localized_displacement(current_displacement);

      // The symmetric gradients of the displacement shape functions at a
      // certain point. There are dofs_per_cell shape functions so the size is
      // dofs_per_cell.
      std::vector<SymmetricTensor<2, dim>> symmetric_grad_phi(dofs_per_cell);
      // The shape functions at a certain point.
      std::vector<Tensor<1, dim>> phi(dofs_per_cell);
      // A "viewer" to describe the nodal dofs as a vector.
      FEValuesExtractors::Vector displacements(0);

      //********** TO BE FIXED LATER **************//
      // Temporarily disable fsi_stress_row calculation //

      /*std::vector<std::vector<Tensor<1, dim>>> fsi_stress_rows_values(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          fsi_stress_rows_values[d].resize(n_f_q_points);
        }*/
      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        {
          gravity[i] = parameters.gravity[i];
        }

      // fsi_vel_diff_lag  at quadrature points (used to
      // calculate penalty force for OpenIFEM-SABLE coupling)
      std::vector<Tensor<1, dim>> fsi_vel_diff(n_q_points);

      // Loop over cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          // Only operates on the locally owned cells
          if (cell->subdomain_id() == this_mpi_process)
            {
              auto p = cell_property.get_data(cell);
              int mat_id = cell->material_id();
              if (material.size() == 1)
                mat_id = 1;
              elasticity = material[mat_id - 1].get_elasticity();
              viscosity = material[mat_id - 1].get_viscosity();
              local_mass = 0;
              local_matrix = 0;
              local_stiffness = 0;
              local_damping = 0;
              local_rhs = 0;

              fe_values.reinit(cell);

              fe_values[displacements].get_function_values(fsi_vel_diff_lag,
                                                           fsi_vel_diff);

              // Loop over quadrature points
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // Loop over the dofs once, to calculate the grad_ph_u
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      symmetric_grad_phi[k] =
                        fe_values[displacements].symmetric_gradient(k, q);
                      phi[k] = fe_values[displacements].value(k, q);
                    }
                  // Loop over the dofs again, to assemble
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      if (is_initial)
                        {

                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              local_mass[i][j] +=
                                rho * phi[i] * phi[j] * fe_values.JxW(q);
                              local_matrix[i][j] +=
                                (symmetric_grad_phi[i] * viscosity *
                                   symmetric_grad_phi[j] * gamma * dt *
                                   (1 + alpha) +
                                 symmetric_grad_phi[i] * elasticity *
                                   symmetric_grad_phi[j] * beta * dt * dt *
                                   (1 + alpha)) *
                                fe_values.JxW(q);
                              local_stiffness[i][j] +=
                                symmetric_grad_phi[i] * elasticity *
                                symmetric_grad_phi[j] * fe_values.JxW(q);
                              local_damping[i][j] +=
                                symmetric_grad_phi[i] * viscosity *
                                symmetric_grad_phi[j] * fe_values.JxW(q);
                            }
                        }
                      local_rhs[i] += phi[i] * gravity * rho * fe_values.JxW(q);

                      // add penalty force based on Eulerian-Lagrangian velocity
                      // difference (only for OpenIFEM-SABLE coupling)
                      if (parameters.simulation_type == "FSI")
                        {
                          local_rhs[i] +=
                            phi[i] * fsi_vel_diff[q] * fe_values.JxW(q);
                        }
                    }
                }

              cell->get_dof_indices(local_dof_indices);

              // Traction or Pressure
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   ++face)
                {
                  if (cell->face(face)->at_boundary())
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

                      std::vector<double> value;
                      if (parameters.simulation_type != "FSI")
                        {
                          // In stand-alone simulation, the boundary value
                          // is prescribed by the user.
                          value = parameters.solid_neumann_bcs[id];
                        }
                      Tensor<1, dim> traction;
                      if (parameters.simulation_type != "FSI" &&
                          parameters.solid_neumann_bc_type == "Traction")
                        {
                          for (unsigned int i = 0; i < dim; ++i)
                            {
                              traction[i] = value[i];
                            }
                        }

                      //********** TO BE FIXED LATER **************//
                      // Temporarily disable fsi_stress_row calculation //

                      // Get FSI stress values on face quadrature points
                      /*std::vector<Tensor<2, dim>> fsi_stress(n_f_q_points);
                      if (parameters.simulation_type == "FSI")
                        {
                          std::vector<Point<dim>> vertex_displacement(
                            GeometryInfo<dim>::vertices_per_face);
                          for (unsigned int v = 0;
                               v < GeometryInfo<dim>::vertices_per_face;
                               ++v)
                            {
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  vertex_displacement[v][d] =
                                    localized_displacement(
                                      cell->face(face)->vertex_dof_index(v, d));
                                }
                              cell->face(face)->vertex(v) +=
                                vertex_displacement[v];
                            }
                          fe_face_values.reinit(cell, face);
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              fe_face_values[displacements].get_function_values(
                                fsi_stress_rows[d], fsi_stress_rows_values[d]);
                            }
                          for (unsigned int v = 0;
                               v < GeometryInfo<dim>::vertices_per_face;
                               ++v)
                            {
                              cell->face(face)->vertex(v) -=
                                vertex_displacement[v];
                            }
                          for (unsigned int q = 0; q < n_f_q_points; ++q)
                            {
                              for (unsigned int d1 = 0; d1 < dim; ++d1)
                                {
                                  for (unsigned int d2 = 0; d2 < dim; ++d2)
                                    {
                                      fsi_stress[q][d1][d2] =
                                        fsi_stress_rows_values[d1][q][d2];
                                    }
                                }
                            } // End looping face quadrature points
                        }
                      else
                        {
                          fe_face_values.reinit(cell, face);
                        }*/

                      fe_face_values.reinit(cell, face);

                      for (unsigned int q = 0; q < n_f_q_points; ++q)
                        {
                          if (parameters.simulation_type != "FSI" &&
                              parameters.solid_neumann_bc_type == "Pressure")
                            {
                              // The normal is w.r.t. reference
                              // configuration!
                              traction = fe_face_values.normal_vector(q);
                              traction *= value[0];
                            }
                          else if (parameters.simulation_type == "FSI")
                            {
                              //********** TO BE FIXED LATER **************//
                              // Temporarily disable fsi_stress_row calculation
                              // //
                              /*traction =
                                fsi_stress[q] *
                                fe_face_values.normal_vector(q);*/
                              if (p[face]->fsi_traction.size() != 0)
                                traction = p[face]->fsi_traction[q];
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
                }

              // create lumped mass matrix

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  double sum = 0;
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      sum = sum + local_mass[i][j];
                    }
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      if (i == j)
                        {
                          local_mass[i][j] = sum;
                        }
                      else
                        {
                          local_mass[i][j] = 0;
                        }
                    }
                }

              local_matrix.add(1, local_mass);

              // Now distribute local data to the system, and apply the
              // hanging node constraints at the same time.
              if (is_initial)
                {
                  constraints.distribute_local_to_global(
                    local_matrix, local_dof_indices, system_matrix);
                  constraints.distribute_local_to_global(
                    local_mass, local_dof_indices, mass_matrix);
                  constraints.distribute_local_to_global(
                    local_stiffness, local_dof_indices, stiffness_matrix);
                  constraints.distribute_local_to_global(
                    local_damping, local_dof_indices, damping_matrix);
                }
              constraints.distribute_local_to_global(
                local_rhs, local_dof_indices, system_rhs);
            }
        }
      // Synchronize with other processors.
      if (is_initial)
        {
          system_matrix.compress(VectorOperation::add);
          mass_matrix.compress(VectorOperation::add);
          stiffness_matrix.compress(VectorOperation::add);
          damping_matrix.compress(VectorOperation::add);

          // OpenIFEM-SABLE coupling: save system_matrix diagonal for
          // application of added mass effect
          if (parameters.simulation_type == "FSI")
            {
              std::pair<int, int> range = system_matrix.local_range();
              for (int i = range.first; i < range.second; i++)
                system_matrix_diagonal.push_back(system_matrix.el(i, i));
            }
        }
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    void SharedLinearElasticity<dim>::run_one_step(bool first_step)
    {
      std::cout.precision(6);
      std::cout.width(12);

      double alpha = -parameters.damping;
      double gamma = 0.5 - alpha;
      double beta = pow((1 - alpha), 2) / 4;

      if (first_step)
        {
          // Compute the mass matrix for the initial acceleration, \f$ Ma_n = F
          // \f$ and the system matrices. This is only done once even in FSI
          // mode.
          assemble_system(true);

          // Save nodal mass in a vector
          std::pair<int, int> range = mass_matrix.local_range();
          for (int i = range.first; i < range.second; i++)
            {
              nodal_mass[i] = mass_matrix.el(i, i);
            }
          nodal_mass.compress(VectorOperation::insert);

          calculate_KE_and_momemtum();

          // apply added mass effect
          if (parameters.simulation_type == "FSI")
            {
              std::pair<int, int> range = mass_matrix.local_range();
              for (int i = range.first; i < range.second; i++)
                mass_matrix.add(i, i, added_mass_effect[i]);
              mass_matrix.compress(VectorOperation::add);
            }

          this->solve(mass_matrix, previous_acceleration, system_rhs);
          this->output_results(time.get_timestep());
        }

      else if (parameters.simulation_type == "FSI")
        assemble_system(false);

      const double dt = time.get_delta_t();

      PETScWrappers::MPI::Vector tmp1(locally_owned_dofs, mpi_communicator);
      PETScWrappers::MPI::Vector tmp2(locally_owned_dofs, mpi_communicator);
      PETScWrappers::MPI::Vector tmp3(locally_owned_dofs, mpi_communicator);
      PETScWrappers::MPI::Vector tmp4(locally_owned_dofs, mpi_communicator);
      PETScWrappers::MPI::Vector tmp5(locally_owned_dofs, mpi_communicator);

      time.increment();
      pcout << std::string(91, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;

      // Modify the RHS
      tmp1 = system_rhs;
      tmp2 = previous_displacement;
      tmp2.add((1 + alpha) * dt,
               previous_velocity,
               (0.5 - beta) * dt * dt * (1 + alpha),
               previous_acceleration);
      stiffness_matrix.vmult(tmp3, tmp2);

      tmp4 = previous_velocity;
      tmp4.add((1 + alpha) * (1 - gamma) * dt, previous_acceleration);
      damping_matrix.vmult(tmp5, tmp4);
      tmp1 -= tmp3;
      tmp1 -= tmp5;

      // apply added mass effect
      if (parameters.simulation_type == "FSI")
        {
          std::pair<int, int> range = system_matrix.local_range();
          for (unsigned int i = 0; i < system_matrix_diagonal.size(); i++)
            {
              int index = i + range.first;
              system_matrix.set(index,
                                index,
                                system_matrix_diagonal[i] +
                                  added_mass_effect[index]);
            }
          system_matrix.compress(VectorOperation::insert);
        }

      auto state = this->solve(system_matrix, current_acceleration, tmp1);

      // update the current velocity
      // \f$ v_{n+1} = v_n + (1-\gamma)\Delta{t}a_n + \gamma\Delta{t}a_{n+1}
      // \f$
      current_velocity = previous_velocity;
      current_velocity.add(dt * (1 - gamma),
                           previous_acceleration,
                           dt * gamma,
                           current_acceleration);

      // update the current displacement
      current_displacement = previous_displacement;
      current_displacement.add(dt, previous_velocity);
      current_displacement.add(dt * dt * (0.5 - beta),
                               previous_acceleration,
                               dt * dt * beta,
                               current_acceleration);

      // update the previous values
      previous_acceleration = current_acceleration;
      previous_velocity = current_velocity;
      previous_displacement = current_displacement;

      pcout << std::scientific << std::left << " CG iteration: " << std::setw(3)
            << state.first << " CG residual: " << state.second << std::endl;

      // strain and stress
      update_strain_and_stress();

      calculate_KE_and_momemtum();

      if (time.time_to_output())
        {
          this->output_results(time.get_timestep());
        }

      if (parameters.simulation_type == "Solid" && time.time_to_refine())
        {
          this->refine_mesh(parameters.global_refinements[1],
                            parameters.global_refinements[1] + 3);
          tmp1.reinit(locally_owned_dofs, mpi_communicator);
          tmp2.reinit(locally_owned_dofs, mpi_communicator);
          tmp3.reinit(locally_owned_dofs, mpi_communicator);
          assemble_system(true);
        }

      if (parameters.simulation_type == "Solid" && time.time_to_save())
        {
          this->save_checkpoint(time.get_timestep());
        }
    }

    template <int dim>
    void SharedLinearElasticity<dim>::update_strain_and_stress()
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

      SymmetricTensor<4, dim> elasticity;
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
              fe_values[displacements].get_function_gradients(
                localized_current_displacement, current_displacement_gradients);
              int mat_id = cell->material_id();
              if (parameters.n_solid_parts == 1)
                mat_id = 1;
              elasticity = material[mat_id - 1].get_elasticity();

              for (unsigned int q = 0; q < volume_quad_formula.size(); ++q)
                {
                  SymmetricTensor<2, dim> tmp_strain, tmp_stress;
                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          tmp_strain[i][j] =
                            (current_displacement_gradients[q][i][j] +
                             current_displacement_gradients[q][j][i]) /
                            2;
                          quad_strain[i][j][q] = tmp_strain[i][j];
                        }
                    }
                  tmp_stress = elasticity * tmp_strain;
                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          quad_stress[i][j][q] = tmp_stress[i][j];
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

    template class SharedLinearElasticity<2>;
    template class SharedLinearElasticity<3>;
  } // namespace MPI
} // namespace Solid

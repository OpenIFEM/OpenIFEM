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
      double beta = pow((1 - alpha), 2) / 4;

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

      std::vector<std::vector<Tensor<1, dim>>> fsi_stress_rows_values(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          fsi_stress_rows_values[d].resize(n_f_q_points);
        }
      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        {
          gravity[i] = parameters.gravity[i];
        }

      // Loop over cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          // Only operates on the locally owned cells
          if (cell->subdomain_id() == this_mpi_process)
            {
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
                                (rho * phi[i] * phi[j] + // mass matrix part
                                 symmetric_grad_phi[i] * viscosity *
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
                    }
                }

              cell->get_dof_indices(local_dof_indices);

              // Traction or Pressure
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   ++face)
                {
                  if (cell->face(face)->at_boundary())
                    // if (cell->face(face)->at_boundary() &&
                    // cell->neighbor_index(face) == -1)
                    {
                      unsigned int id = cell->face(face)->boundary_id();
                      if (!cell->face(face)->at_boundary())
                        // if (parameters.solid_neumann_bcs.find(id) ==
                        // parameters.solid_neumann_bcs.end())
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

                      // Get FSI stress values on face quadrature points
                      std::vector<Tensor<2, dim>> fsi_stress(n_f_q_points);
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
                        }

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
                              traction =
                                fsi_stress[q] * fe_face_values.normal_vector(q);
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
              /*
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
              */

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

          update_strain_and_stress();
          this->solve(mass_matrix, previous_acceleration, system_rhs);
          this->output_results(time.get_timestep());

          // compute KE
          PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
          mass_matrix.vmult(tmp, current_velocity);
          double K = 0.5 * (current_velocity * tmp);

          stiffness_matrix.vmult(tmp, current_displacement);
          double U = 0.5 * (current_displacement * tmp);

          total_external_work = 0.0; // Initialize at t=0
          total_external_work_v = 0.0;

          if (this_mpi_process == 0)
            {
              std::ofstream out("energy_estimates.csv");
              out << "time,kinetic_energy,strain_energy,external_work\n";
              out << time.current() << "," << K << "," << U << ","
                  << total_external_work << "\n";
            }

          compute_and_write_energy();
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

      // Compute external work increment
      PETScWrappers::MPI::Vector delta_u = current_displacement;
      delta_u -= previous_displacement;
      double delta_W = system_rhs * delta_u;
      total_external_work += delta_W;
      compute_and_write_energy();

      // update the previous values
      previous_acceleration = current_acceleration;

      // compute_solid_energy();

      previous_velocity = current_velocity;

      previous_displacement = current_displacement;

      pcout << std::scientific << std::left << " CG iteration: " << std::setw(3)
            << state.first << " CG residual: " << state.second << std::endl;

      // strain and stress
      update_strain_and_stress();

      if (time.time_to_output())
        {
          this->output_results(time.get_timestep());

          // compute KE
          PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
          mass_matrix.vmult(tmp, current_velocity);
          double K = 0.5 * (current_velocity * tmp);
          stiffness_matrix.vmult(tmp, current_displacement);
          double U = 0.5 * (current_displacement * tmp);

          if (this_mpi_process == 0)
            {
              std::ofstream out("energy_estimates.csv", std::ios::app);
              out << time.current() << "," << K << "," << U << ","
                  << total_external_work << "\n";
            }
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

      Vector<double> localized_velocity(current_velocity);

      std::vector<Tensor<2, dim>> current_velocity_gradients(
        volume_quad_formula.size());

      std::vector<SymmetricTensor<2, dim>> sym_grad_v(
        volume_quad_formula.size());

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

    template <int dim>
    void SharedLinearElasticity<dim>::compute_and_write_energy()
    {
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points |
                                update_normal_vectors | update_JxW_values);

      // const unsigned int dofs_per_cell = fe.dofs_per_cell;

      const unsigned int n_q_points = volume_quad_formula.size();

      Vector<double> localized_velocity(current_velocity);
      std::vector<Tensor<2, dim>> grad_u(n_q_points);
      std::vector<SymmetricTensor<2, dim>> sym_grad_u(n_q_points);
      std::vector<Tensor<1, dim>> vel(n_q_points);

      const FEValuesExtractors::Vector displacements(0);

      SymmetricTensor<4, dim> elasticity;

      Vector<double> local_disp(current_displacement);

      auto cell = dof_handler.begin_active();

      double local_ke = 0.0;
      double ke = 0.0;
      double total_strain_energy = 0.0;
      double local_strain_energy = 0.0;

      for (; cell != dof_handler.end(); ++cell)
        {
          if (cell->subdomain_id() == this_mpi_process)
            {
              fe_values.reinit(cell);
              fe_values[displacements].get_function_values(localized_velocity,
                                                           vel);

              fe_values[displacements].get_function_gradients(local_disp,
                                                              grad_u);

              // Material parameters
              int mat_id = cell->material_id();
              if (parameters.n_solid_parts == 1)
                mat_id = 1; // single material fallback
              elasticity = material[mat_id - 1].get_elasticity();

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  local_ke += 0.5 * parameters.solid_rho *
                              vel[q].norm_square() * fe_values.JxW(q);
                  SymmetricTensor<2, dim> eps_q;
                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = 0; j < dim; ++j)
                      eps_q[i][j] = 0.5 * (grad_u[q][i][j] + grad_u[q][j][i]);
                  // stress at quadrature point
                  const SymmetricTensor<2, dim> sigma_q = elasticity * eps_q;

                  // (1/2) sigma : eps
                  double w_q = 0.5 * (sigma_q * eps_q);

                  // multiply by volume element
                  w_q *= fe_values.JxW(q);

                  local_strain_energy += w_q;
                }
            }
        }

      MPI_Allreduce(&local_strain_energy,
                    &total_strain_energy,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_ke, &ke, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      double local_incremental_work = 0.0;
      double incremental_work = 0.0;

      FEFaceValues<dim> fe_face_values(
        fe,
        face_quad_formula,
        update_values | update_quadrature_points | update_normal_vectors |
          update_JxW_values);

      const unsigned int n_face_q_points = face_quad_formula.size();

      // Copy old and new displacements to CPU vectors:
      Vector<double> disp_old(previous_displacement);
      Vector<double> disp_new(current_displacement);
      std::vector<Tensor<1, dim>> vals_old(n_face_q_points),
        vals_new(n_face_q_points);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->subdomain_id() == this_mpi_process)
            {
              for (unsigned int face_n = 0;
                   face_n < GeometryInfo<dim>::faces_per_cell;
                   ++face_n)
                {
                  if (cell->face(face_n)->at_boundary())
                    {
                      const unsigned int id = cell->face(face_n)->boundary_id();
                      if (parameters.solid_neumann_bcs.find(id) ==
                          parameters.solid_neumann_bcs.end())
                        {
                          continue;
                        }
                      fe_face_values.reinit(cell, face_n);
                      fe_face_values[displacements].get_function_values(
                        disp_old, vals_old);
                      fe_face_values[displacements].get_function_values(
                        disp_new, vals_new);

                      const std::vector<double> &val =
                        parameters.solid_neumann_bcs[id];

                      for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {

                          Tensor<1, dim> traction;

                          if (parameters.simulation_type != "FSI")
                            {
                              if (parameters.solid_neumann_bc_type ==
                                  "Pressure")
                                {
                                  traction = fe_face_values.normal_vector(q);
                                  traction *=
                                    val[0]; // assume val[0] is the scalar
                                }

                              else if (parameters.solid_neumann_bc_type ==
                                       "Traction")
                                {
                                  for (unsigned int d = 0; d < dim; ++d)
                                    traction[d] = val[d];
                                }
                            }

                          Tensor<1, dim> delta_u;
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              delta_u[d] = vals_new[q][d] - vals_old[q][d];
                            }

                          const double t_dot_delta_u =
                            (traction[0] * delta_u[0] +
                             (dim > 1 ? traction[1] * delta_u[1] : 0.0) +
                             (dim > 2 ? traction[2] * delta_u[2] : 0.0));

                          local_incremental_work +=
                            t_dot_delta_u * fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      MPI_Allreduce(&local_incremental_work,
                    &incremental_work,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      total_external_work_v += incremental_work;

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream energy_csv;

          if (time.current() == 0)
            {
              energy_csv.open("energy_verify.csv");
              energy_csv << "time"
                         << "kinetic_energy"
                         << "potential_energy"
                         << "external_work"
                         << "\n";
            }
          else
            {
              energy_csv.open("energy_verify.csv", std::ios_base::app);
            }

          energy_csv << std::scientific << time.current() << "," << ke << ","
                     << total_strain_energy << "," << total_external_work_v
                     << ","
                     << "\n";
          energy_csv.close();
        }
    }

    template class SharedLinearElasticity<2>;
    template class SharedLinearElasticity<3>;
  } // namespace MPI
} // namespace Solid

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
          LinearElasticMaterial<dim> tmp(
            parameters.E[i], parameters.nu[i], parameters.solid_rho);
          material[i] = tmp;
        }
    }

    template <int dim>
    void SharedLinearElasticity<dim>::assemble_system(const bool is_initial)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      double alpha = 0.95;
      double beta = pow((2 - alpha), 2) / 4;

      system_matrix = 0;
      stiffness_matrix = 0;
      system_rhs = 0;

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

      FEValues<dim> scalar_fe_values(scalar_fe,
                                     volume_quad_formula,
                                     update_values | update_gradients |
                                       update_quadrature_points);

      FEFaceValues<dim> fe_face_values(
        fe,
        face_quad_formula,
        update_values | update_quadrature_points | update_normal_vectors |
          update_JxW_values);

      SymmetricTensor<4, dim> elasticity;
      const double rho = material[0].get_density();
      const double dt = time.get_delta_t();

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_f_q_points = face_quad_formula.size();

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_stiffness(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      Vector<double> localized_displacement(current_displacement);
      std::vector<std::vector<Vector<double>>> localized_strain_rate(dim);
      for (unsigned int i = 0; i < dim; ++i)
        {
          localized_strain_rate[i].resize(dim);
          for (unsigned int j = 0; j < dim; ++j)

            {
              localized_strain_rate[i][j] = strain[i][j];
              Vector<double> localized_previous_strain_component(
                previous_strain[i][j]);
              localized_strain_rate[i][j] -=
                localized_previous_strain_component;
              localized_strain_rate[i][j] /= time.get_delta_t();
            }
        }

      // The symmetric gradients of the displacement shape functions at a
      // certain point. There are dofs_per_cell shape functions so the size is
      // dofs_per_cell.
      std::vector<SymmetricTensor<2, dim>> symmetric_grad_phi(dofs_per_cell);
      // The shape functions at a certain point.
      std::vector<Tensor<1, dim>> phi(dofs_per_cell);
      // A "viewer" to describe the nodal dofs as a vector.
      FEValuesExtractors::Vector displacements(0);

      // Strain rate used for viscous damping
      std::vector<Tensor<1, dim>> strain_rate_divergence(n_q_points);
      std::vector<Tensor<1, dim>> strain_rate_components(n_q_points);

      // Loop over cells
      for (auto cell = dof_handler.begin_active(),
                scalar_cell = scalar_dof_handler.begin_active();
           cell != dof_handler.end();
           ++cell, ++scalar_cell)
        {
          // Only operates on the locally owned cells
          if (cell->subdomain_id() == this_mpi_process)
            {
              auto p = cell_property.get_data(cell);
              int mat_id = cell->material_id();
              if (material.size() == 1)
                mat_id = 1;
              elasticity = material[mat_id - 1].get_elasticity();
              Assert(p.size() == GeometryInfo<dim>::faces_per_cell,
                     ExcMessage("Wrong number of cell data!"));
              local_matrix = 0;
              local_stiffness = 0;
              local_rhs = 0;

              fe_values.reinit(cell);
              scalar_fe_values.reinit(scalar_cell);

              // Compute the divergence of strain rate
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  strain_rate_divergence[q] = 0;
                }
              for (unsigned int i = 0; i < dim; ++i)
                {
                  for (unsigned int j = 0; j < dim; ++j)
                    {

                      scalar_fe_values.get_function_gradients(
                        localized_strain_rate[i][j], strain_rate_components);
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                          strain_rate_divergence[q][i] +=
                            strain_rate_components[q][j];
                        }
                    }
                }

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
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          if (is_initial)
                            {
                              local_matrix[i][j] +=
                                rho * phi[i] * phi[j] * fe_values.JxW(q);
                            }
                          else
                            {
                              local_matrix[i][j] +=
                                (rho * phi[i] * phi[j] +
                                 symmetric_grad_phi[i] * elasticity *
                                   symmetric_grad_phi[j] * beta * dt * dt) *
                                fe_values.JxW(q);
                              local_stiffness[i][j] +=
                                symmetric_grad_phi[i] * elasticity *
                                symmetric_grad_phi[j] * fe_values.JxW(q);
                            }
                        }
                      // zero body force, viscous damping
                      Tensor<1, dim> gravity;
                      local_rhs[i] +=
                        (phi[i] * gravity * rho - parameters.damping * phi[i] *
                                                    strain_rate_divergence[q]) *
                        fe_values.JxW(q);
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

                      // Get FSI stress values on face quadrature points
                      std::vector<Tensor<2, dim>> fsi_stress(n_f_q_points);
                      if (parameters.simulation_type == "FSI")
                        {
                          Assert(
                            parameters.solid_degree == 1,
                            ExcMessage(
                              "FSI traction only supports 1st order solid!"));
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
                          for (unsigned int v = 0;
                               v < GeometryInfo<dim>::vertices_per_face;
                               ++v)
                            {
                              cell->face(face)->vertex(v) -=
                                vertex_displacement[v];
                            }
                          for (unsigned int q = 0; q < n_f_q_points; ++q)
                            {
                              for (unsigned int v = 0;
                                   v < GeometryInfo<dim>::vertices_per_face;
                                   ++v)
                                {
                                  // shape_value() has the size of
                                  // dof_per_cells, even for fe_face_values. So
                                  // we have to loop over the cell vertices to
                                  // locate where we are
                                  unsigned int function_no;
                                  for (unsigned int cell_v = 0;
                                       cell_v <
                                       GeometryInfo<dim>::vertices_per_cell;
                                       ++cell_v)
                                    {
                                      // Get the corresponding cell vertex
                                      if (cell->face(face)->vertex_index(v) ==
                                          cell->vertex_index(cell_v))
                                        // Get the dof number
                                        {
                                          types::global_dof_index v_dof =
                                            cell->vertex_dof_index(cell_v, 0);
                                          for (unsigned d = 0;
                                               d < local_dof_indices.size();
                                               ++d)
                                            {
                                              if (local_dof_indices[d] == v_dof)
                                                function_no = d;
                                            }
                                          break;
                                        }
                                    }
                                  fsi_stress[q] +=
                                    fe_face_values.shape_value(function_no, q) *
                                    p[face]->fsi_stress[v];
                                }
                            }
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

              // Now distribute local data to the system, and apply the
              // hanging node constraints at the same time.
              constraints.distribute_local_to_global(local_matrix,
                                                     local_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
              constraints.distribute_local_to_global(
                local_stiffness, local_dof_indices, stiffness_matrix);
            }
        }
      // Synchronize with other processors.
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
      stiffness_matrix.compress(VectorOperation::add);
    }

    template <int dim>
    void SharedLinearElasticity<dim>::run_one_step(bool first_step)
    {
      std::cout.precision(6);
      std::cout.width(12);

      double alpha = 0.95;
      double gamma = 1.5 - alpha;
      double beta = pow((2 - alpha), 2) / 4;

      if (first_step)
        {
          // Neet to compute the initial acceleration, \f$ Ma_n = F \f$,
          // at this point set system_matrix to mass_matrix.
          assemble_system(true);
          this->solve(system_matrix, previous_acceleration, system_rhs);
          // Update the system_matrix
          assemble_system(false);
          this->output_results(time.get_timestep());
        }

      else
        assemble_system(false);

      const double dt = time.get_delta_t();

      PETScWrappers::MPI::Vector tmp1(locally_owned_dofs, mpi_communicator);
      PETScWrappers::MPI::Vector tmp2(locally_owned_dofs, mpi_communicator);

      PETScWrappers::MPI::Vector tmp3(locally_owned_dofs, mpi_communicator);

      time.increment();
      pcout << std::string(91, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;

      // Modify the RHS
      tmp1 = system_rhs;
      tmp2 = previous_displacement;
      tmp2.add(
        dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);
      stiffness_matrix.vmult(tmp3, tmp2);
      tmp1 -= tmp3;

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

      current_displacement *= alpha;
      current_displacement.add(1 - alpha, previous_displacement);
      current_velocity *= alpha;
      current_velocity.add(1 - alpha, previous_velocity);

      // update the previous values
      previous_acceleration = current_acceleration;
      previous_velocity = current_velocity;
      previous_displacement = current_displacement;

      pcout << std::scientific << std::left << " CG iteration: " << std::setw(3)
            << state.first << " CG residual: " << state.second << std::endl;

      // strain and stress
      update_strain_and_stress();

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
          assemble_system(false);
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
              previous_strain[i][j] = strain[i][j];
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

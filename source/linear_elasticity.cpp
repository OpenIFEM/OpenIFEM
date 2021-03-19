#include "linear_elasticity.h"

namespace Solid
{
  using namespace dealii;

  template <int dim>
  LinearElasticity<dim>::LinearElasticity(
    Triangulation<dim> &tria, const Parameters::AllParameters &parameters)
    : SolidSolver<dim>(tria, parameters)
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
  void LinearElasticity<dim>::assemble(bool is_initial, bool assemble_matrix)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    double gamma = 0.5 + parameters.damping;
    double beta = gamma / 2;

    if (assemble_matrix)
      {
        system_matrix = 0;
        stiffness_matrix = 0;
      }
    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
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

    // The symmetric gradients of the displacement shape functions at a certain
    // point.
    // There are dofs_per_cell shape functions so the size is dofs_per_cell.
    std::vector<SymmetricTensor<2, dim>> symmetric_grad_phi(dofs_per_cell);
    // The shape functions at a certain point.
    std::vector<Tensor<1, dim>> phi(dofs_per_cell);
    // A "viewer" to describe the nodal dofs as a vector.
    FEValuesExtractors::Vector displacements(0);

    // Loop over cells
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
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
                if (assemble_matrix)
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
                  }
                // body force
                Tensor<1, dim> gravity;
                for (unsigned int i = 0; i < dim; ++i)
                  {
                    gravity[i] = parameters.gravity[i];
                  }
                local_rhs[i] += phi[i] * gravity * rho * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);

        // Neumann boundary conditions
        // If this is a stand-alone solid simulation, the Neumann boundary type
        // should be either Traction or Pressure;
        // it this is a FSI simulation, the Neumann boundary type must be FSI.

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            unsigned int id = cell->face(face)->boundary_id();

            if (!cell->face(face)->at_boundary() ||
                parameters.solid_dirichlet_bcs.find(id) !=
                  parameters.solid_dirichlet_bcs.end())
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
                    // TODO:
                    // here and FSI, the normal is w.r.t. reference
                    // configuration,
                    // should be changed to current config.
                    traction = fe_face_values.normal_vector(q);
                    traction *= prescribed_value[0];
                  }
                else if (parameters.simulation_type == "FSI")
                  {
                    traction = p[face]->fsi_traction;
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

        if (assemble_matrix)
          {
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
        else
          {
            constraints.distribute_local_to_global(
              local_rhs, local_dof_indices, system_rhs);
          }
      }
  }

  template <int dim>
  void LinearElasticity<dim>::assemble_system(bool is_initial)
  {
    assemble(is_initial, true);
  }

  template <int dim>
  void LinearElasticity<dim>::assemble_rhs()
  {
    // In case of assembling rhs only, the first boolean does not matter.
    assemble(false, false);
  }

  template <int dim>
  void LinearElasticity<dim>::run_one_step(bool first_step)
  {
    std::cout.precision(6);
    std::cout.width(12);

    double gamma = 0.5 + parameters.damping;
    double beta = gamma / 2;

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

    const double dt = time.get_delta_t();

    // Time loop
    time.increment();
    std::cout << std::string(91, '*') << std::endl
              << "Time step = " << time.get_timestep()
              << ", at t = " << std::scientific << time.current() << std::endl;

    // In FSI application we have to update the RHS
    if (parameters.simulation_type == "FSI")
      {
        assemble_rhs();
      }
    // Modify the RHS
    Vector<double> tmp1(system_rhs);
    auto tmp2 = previous_displacement;
    tmp2.add(
      dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);
    Vector<double> tmp3(dof_handler.n_dofs());
    stiffness_matrix.vmult(tmp3, tmp2);
    tmp1 -= tmp3;

    auto state = this->solve(system_matrix, current_acceleration, tmp1);

    // update the current velocity
    // \f$ v_{n+1} = v_n + (1-\gamma)\Delta{t}a_n + \gamma\Delta{t}a_{n+1}
    // \f$
    current_velocity = previous_velocity;
    current_velocity.add(dt * (1 - gamma), previous_acceleration);
    current_velocity.add(dt * gamma, current_acceleration);

    // update the current displacement
    current_displacement = previous_displacement;
    current_displacement.add(dt, previous_velocity);
    current_displacement.add(dt * dt * (0.5 - beta), previous_acceleration);
    current_displacement.add(dt * dt * beta, current_acceleration);

    // update the previous values
    previous_acceleration = current_acceleration;
    previous_velocity = current_velocity;
    previous_displacement = current_displacement;

    std::cout << std::scientific << std::left
              << " CG iteration: " << std::setw(3) << state.first
              << " CG residual: " << state.second << std::endl;

    update_strain_and_stress();

    if (time.time_to_output())
      {
        this->output_results(time.get_timestep());
      }

    if (time.time_to_refine())
      {
        this->refine_mesh(1, 4);
        assemble_system(false);
      }
  }

  template <int dim>
  void LinearElasticity<dim>::update_strain_and_stress()
  {
    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          {
            strain[i][j] = 0.0;
            stress[i][j] = 0.0;
          }
      }
    std::vector<int> surrounding_cells(scalar_dof_handler.n_dofs(), 0);
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
      std::vector<Vector<double>>(dim,
                                  Vector<double>(volume_quad_formula.size())));
    std::vector<std::vector<Vector<double>>> quad_stress(
      dim,
      std::vector<Vector<double>>(dim,
                                  Vector<double>(volume_quad_formula.size())));

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
    std::vector<types::global_dof_index> dof_indices(scalar_fe.dofs_per_cell);

    for (unsigned int i = 0; i < cellwise_stress.size(); ++i)
      {
        cellwise_stress[i].reinit(triangulation.n_active_cells());
      }

    for (; cell != dof_handler.end(); ++cell, ++scalar_cell)
      {
        std::vector<double> tmp_cell_stress(6, 0.0);

        scalar_cell->get_dof_indices(dof_indices);
        fe_values.reinit(cell);
        fe_values[displacements].get_function_gradients(
          current_displacement, current_displacement_gradients);
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

            tmp_cell_stress[0] += (0.25 * quad_stress[0][0][q]);
            tmp_cell_stress[1] += (0.25 * quad_stress[0][1][q]);
            tmp_cell_stress[2] += (0.25 * quad_stress[1][1][q]);

            if (dim == 3)
              {
                tmp_cell_stress[3] += (0.25 * quad_stress[0][2][q]);
                tmp_cell_stress[4] += (0.25 * quad_stress[1][2][q]);
                tmp_cell_stress[5] += (0.25 * quad_stress[2][2][q]);
              }
          }

        for (unsigned int i = 0; i < cellwise_stress.size(); ++i)
          {
            cellwise_stress[i][cell->active_cell_index()] = tmp_cell_stress[i];
          }

        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                qpt_to_dof.vmult(cell_strain[i][j], quad_strain[i][j]);
                qpt_to_dof.vmult(cell_stress[i][j], quad_stress[i][j]);
                for (unsigned int k = 0; k < scalar_fe.dofs_per_cell; ++k)
                  {
                    strain[i][j][dof_indices[k]] += cell_strain[i][j][k];
                    stress[i][j][dof_indices[k]] += cell_stress[i][j][k];
                    if (i == 0 && j == 0)
                      surrounding_cells[dof_indices[k]]++;
                  }
              }
          }
      }

    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          {
            for (unsigned int k = 0; k < scalar_dof_handler.n_dofs(); ++k)
              {
                strain[i][j][k] /= surrounding_cells[k];
                stress[i][j][k] /= surrounding_cells[k];
              }
          }
      }
  }

  template class LinearElasticity<2>;
  template class LinearElasticity<3>;
} // namespace Solid

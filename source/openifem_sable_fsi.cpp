#include "openifem_sable_fsi.h"
#include <complex>
#include <iostream>

template <int dim>
OpenIFEM_Sable_FSI<dim>::~OpenIFEM_Sable_FSI()
{
  timer.print_summary();
}

template <int dim>
OpenIFEM_Sable_FSI<dim>::OpenIFEM_Sable_FSI(Fluid::SableWrap<dim> &f,
                                            Solid::SolidSolver<dim> &s,
                                            const Parameters::AllParameters &p,
                                            bool use_dirichlet_bc)
  : FSI<dim>(f, s, p, use_dirichlet_bc), sable_solver(f)
{
  assert(use_dirichlet_bc == false);
}

// Dirichlet bcs are applied to artificial fluid cells, so fluid nodes should
// be marked as artificial or real. Meanwhile, additional body force is
// applied to the artificial fluid quadrature points. To accomodate these two
// settings, we define indicator at quadrature points, but only when all
// of the vertices of a fluid cell are found to be in solid domain,
// set the indicators at all quadrature points to be 1.
template <int dim>
void OpenIFEM_Sable_FSI<dim>::update_indicator()
{
  TimerOutput::Scope timer_section(timer, "Update indicator");

  cell_partially_inside_solid.clear();
  cell_nodes_inside_solid.clear();
  cell_nodes_inside_solid.clear();
  for (unsigned int i = 0; i < sable_solver.triangulation.n_active_cells(); i++)
    cell_partially_inside_solid.push_back(false);

  // set condition for the indicator field
  // cell is aritifical if nodes_inside_solid > min_nodes_inside
  unsigned int min_nodes_inside = GeometryInfo<dim>::vertices_per_cell - 1;
  if (parameters.indicator_field_condition == "CompletelyInsideSolid")
    {
      min_nodes_inside = GeometryInfo<dim>::vertices_per_cell - 1;
    }
  else if (parameters.indicator_field_condition == "PartiallyInsideSolid")
    min_nodes_inside = 0;

  FEValues<dim> fe_values(sable_solver.fe,
                          sable_solver.volume_quad_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  move_solid_mesh(true);
  int cell_count = 0;
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      auto p = sable_solver.cell_property.get_data(f_cell);
      unsigned int inside_count = 0;
      std::vector<int> inside_nodes;
      std::vector<int> outside_nodes;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
            {
              ++inside_count;
              inside_nodes.push_back(v);
            }
          else
            outside_nodes.push_back(v);
        }
      p[0]->indicator = (inside_count > min_nodes_inside ? 1 : 0);

      // cell is partially inside the solid
      if ((inside_count > min_nodes_inside) &&
          (inside_count < GeometryInfo<dim>::vertices_per_cell))
        {
          // modify indicator for partially convered cells
          p[0]->indicator =
            double(inside_count) / double(GeometryInfo<dim>::vertices_per_cell);
          cell_partially_inside_solid[cell_count] = true;
          // store local node ids which are inside and outside the solid
          cell_nodes_inside_solid.insert({cell_count, inside_nodes});
          cell_nodes_outside_solid.insert({cell_count, outside_nodes});
        }
      cell_count += 1;

      // Update quadrature points based indicator field
      /*fe_values.reinit(f_cell);
      auto q_points = fe_values.get_quadrature_points();
      unsigned int inside_count_qpoint = 0;

      for (unsigned int q = 0; q < q_points.size(); q++)
        {
          if (point_in_solid(solid_solver.dof_handler, q_points[q]))
            {
              ++inside_count_qpoint;
            }
        }
      p[0]->indicator_qpoint =
        double(inside_count_qpoint) / double(q_points.size());*/
      Point<dim> l1 = f_cell->vertex(0);
      Point<dim> u1 = f_cell->vertex(3);
      // Get uppder and lower corner point for the solid box
      Point<dim> l2(solid_box[0], solid_box[2]);
      Point<dim> u2(solid_box[1], solid_box[3]);
      p[0]->indicator_qpoint = 0;
      // check if rectangles overlap
      if ((l2(0) >= u1(0)) || (l1(0) >= u2(0)))
        continue;
      if ((l2(1) >= u1(1)) || (l1(1) >= u2(1)))
        continue;
      double intersection_area =
        (std::min(u1(0), u2(0)) - std::max(l1(0), l2(0))) *
        (std::min(u1(1), u2(1)) - std::max(l1(1), l2(1)));
      double total_area = abs(u1(0) - l1(0)) * abs(u1(1) - l1(1));
      p[0]->indicator_qpoint = intersection_area / total_area;
    }
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::update_indicator_qpoints()
{
  TimerOutput::Scope timer_section(timer,
                                   "Update quadrature point based indicator");

  move_solid_mesh(true);

  FEValues<dim> fe_values(sable_solver.fe,
                          sable_solver.volume_quad_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  int cell_count = 0;
  cell_nodes_inside_solid.clear();
  cell_nodes_outside_solid.clear();
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      fe_values.reinit(f_cell);
      auto q_points = fe_values.get_quadrature_points();
      auto p = sable_solver.cell_property.get_data(f_cell);
      unsigned int inside_count = 0;

      for (unsigned int q = 0; q < q_points.size(); q++)
        {
          if (point_in_solid(solid_solver.dof_handler, q_points[q]))
            {
              ++inside_count;
            }
        }
      p[0]->indicator = double(inside_count) / double(q_points.size());
      // check which cell nodes are inside cells to calculate velocity bc
      std::vector<int> inside_nodes;
      std::vector<int> outside_nodes;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
            {
              inside_nodes.push_back(v);
            }
          else
            outside_nodes.push_back(v);
        }
      // store local node ids which are inside and outside the solid
      cell_nodes_inside_solid.insert({cell_count, inside_nodes});
      cell_nodes_outside_solid.insert({cell_count, outside_nodes});

      cell_count += 1;

      // p[0]->indicator_qpoint = p[0]->indicator;
      Point<dim> l1 = f_cell->vertex(0);
      Point<dim> u1 = f_cell->vertex(3);
      // Get uppder and lower corner point for the solid box
      Point<dim> l2(solid_box[0], solid_box[2]);
      Point<dim> u2(solid_box[1], solid_box[3]);
      p[0]->indicator_qpoint = 0;
      // check if rectangles overlap
      if ((l2(0) >= u1(0)) || (l1(0) >= u2(0)))
        continue;
      if ((l2(1) >= u1(1)) || (l1(1) >= u2(1)))
        continue;
      double intersection_area =
        (std::min(u1(0), u2(0)) - std::max(l1(0), l2(0))) *
        (std::min(u1(1), u2(1)) - std::max(l1(1), l2(1)));
      double total_area = abs(u1(0) - l1(0)) * abs(u1(1) - l1(1));
      p[0]->indicator_qpoint = intersection_area / total_area;
    }

  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::find_fluid_bc()
{
  TimerOutput::Scope timer_section(timer, "Find fluid BC");
  move_solid_mesh(true);

  // The nonzero Dirichlet BCs (to set the velocity) and zero Dirichlet
  // BCs (to set the velocity increment) for the artificial fluid domain.
  AffineConstraints<double> inner_nonzero, inner_zero;
  inner_nonzero.clear();
  inner_zero.clear();
  // inner_nonzero.reinit(fluid_solver.locally_relevant_dofs);
  // inner_zero.reinit(fluid_solver.locally_relevant_dofs);
  BlockVector<double> tmp_fsi_acceleration;
  tmp_fsi_acceleration.reinit(sable_solver.dofs_per_block);
  BlockVector<double> tmp_fsi_velocity;
  tmp_fsi_velocity.reinit(sable_solver.dofs_per_block);

  Vector<double> localized_solid_velocity(solid_solver.current_velocity);
  Vector<double> localized_solid_acceleration(
    solid_solver.current_acceleration);
  std::vector<std::vector<Vector<double>>> localized_stress(
    dim, std::vector<Vector<double>>(dim));
  for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = 0; j < dim; ++j)
        {
          localized_stress[i][j] = solid_solver.stress[i][j];
        }
    }

  const std::vector<Point<dim>> &unit_points =
    sable_solver.fe.get_unit_support_points();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<SymmetricTensor<2, dim>> sym_grad_v(unit_points.size());
  std::vector<double> p(unit_points.size());
  std::vector<Tensor<2, dim>> grad_v(unit_points.size());
  std::vector<Tensor<1, dim>> v(unit_points.size());
  std::vector<Tensor<1, dim>> dv(unit_points.size());

  MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
  Quadrature<dim> dummy_q(unit_points);
  FEValues<dim> dummy_fe_values(mapping,
                                sable_solver.fe,
                                dummy_q,
                                update_quadrature_points | update_values |
                                  update_gradients);
  std::vector<types::global_dof_index> dof_indices(
    sable_solver.fe.dofs_per_cell);
  std::vector<unsigned int> dof_touched(sable_solver.dof_handler.n_dofs(), 0);

  const std::vector<Point<dim>> &scalar_unit_points =
    sable_solver.scalar_fe.get_unit_support_points();
  Quadrature<dim> scalar_dummy_q(scalar_unit_points);
  FEValues<dim> scalar_fe_values(sable_solver.scalar_fe,
                                 scalar_dummy_q,
                                 update_values | update_quadrature_points |
                                   update_JxW_values | update_gradients);
  std::vector<types::global_dof_index> scalar_dof_indices(
    sable_solver.scalar_fe.dofs_per_cell);
  std::vector<unsigned int> scalar_dof_touched(
    sable_solver.scalar_dof_handler.n_dofs(), 0);
  std::vector<double> f_stress_component(scalar_unit_points.size());
  std::vector<std::vector<double>> f_cell_stress =
    std::vector<std::vector<double>>(
      sable_solver.fsi_stress.size(),
      std::vector<double>(scalar_unit_points.size()));

  for (auto scalar_cell = sable_solver.scalar_dof_handler.begin_active();
       scalar_cell != sable_solver.scalar_dof_handler.end();
       ++scalar_cell)
    {
      auto ptr = sable_solver.cell_property.get_data(scalar_cell);
      if (ptr[0]->indicator == 0)
        continue;
      scalar_cell->get_dof_indices(scalar_dof_indices);
      scalar_fe_values.reinit(scalar_cell);

      int stress_index = 0;
      // get fluid stress at support points
      for (unsigned int i = 0; i < dim; i++)
        {
          for (unsigned int j = i; j < dim; j++)
            {
              scalar_fe_values.get_function_values(sable_solver.stress[i][j],
                                                   f_stress_component);
              f_cell_stress[stress_index] = f_stress_component;
              stress_index++;
            }
        }

      for (unsigned int i = 0; i < scalar_unit_points.size(); ++i)
        {
          // Skip the already-set dofs.
          if (scalar_dof_touched[scalar_dof_indices[i]] != 0)
            continue;
          auto scalar_support_points = scalar_fe_values.get_quadrature_points();
          scalar_dof_touched[scalar_dof_indices[i]] = 1;
          if (!point_in_solid(solid_solver.scalar_dof_handler,
                              scalar_support_points[i]))
            continue;
          Utils::GridInterpolator<dim, Vector<double>> scalar_interpolator(
            solid_solver.scalar_dof_handler, scalar_support_points[i]);
          stress_index = 0;
          for (unsigned int j = 0; j < dim; j++)
            {
              for (unsigned int k = j; k < dim; k++)
                {
                  Vector<double> s_stress_component(1);
                  scalar_interpolator.point_value(solid_solver.stress[j][k],
                                                  s_stress_component);
                  // When node-based SABLE stress is used
                  if (parameters.fsi_force_calculation_option == "NodeBased")
                    {
                      sable_solver
                        .fsi_stress[stress_index][scalar_dof_indices[i]] =
                        f_cell_stress[stress_index][i] - s_stress_component[0];
                    }
                  else
                    {
                      // When using cell-wise  SABLE stress is used
                      sable_solver
                        .fsi_stress[stress_index][scalar_dof_indices[i]] =
                        -s_stress_component[0];
                    }
                  stress_index++;
                }
            }
        }
    }

  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      // Use is_artificial() instead of !is_locally_owned() because ghost
      // elements must be taken care of to set correct Dirichlet BCs!
      if (f_cell->is_artificial())
        {
          continue;
        }
      // Now skip the ghost elements because it's not store in cell property.
      if (!use_dirichlet_bc)
        {
          auto ptr = sable_solver.cell_property.get_data(f_cell);
          if (ptr[0]->indicator == 0)
            continue;

          // auto hints = cell_hints.get_data(f_cell);
          dummy_fe_values.reinit(f_cell);
          f_cell->get_dof_indices(dof_indices);
          auto support_points = dummy_fe_values.get_quadrature_points();
          // Fluid velocity at support points
          dummy_fe_values[velocities].get_function_values(
            sable_solver.present_solution, v);
          // Fluid velocity increment at support points
          dummy_fe_values[velocities].get_function_values(
            sable_solver.solution_increment, dv);
          // Fluid velocity gradient at support points
          dummy_fe_values[velocities].get_function_gradients(
            sable_solver.present_solution, grad_v);
          // Fluid symmetric velocity gradient at support points
          dummy_fe_values[velocities].get_function_symmetric_gradients(
            sable_solver.present_solution, sym_grad_v);
          // Fluid pressure at support points
          dummy_fe_values[pressure].get_function_values(
            sable_solver.present_solution, p);
          // Loop over the support points to calculate fsi acceleration.
          for (unsigned int i = 0; i < unit_points.size(); ++i)
            {
              // Skip the already-set dofs.
              if (dof_touched[dof_indices[i]] != 0)
                continue;
              auto base_index = sable_solver.fe.system_to_base_index(i);
              const unsigned int i_group = base_index.first.first;
              Assert(
                i_group < 2,
                ExcMessage("There should be only 2 groups of finite element!"));
              if (i_group == 1)
                continue; // skip the pressure dofs
              // Same as fluid_solver.fe.system_to_base_index(i).first.second;
              const unsigned int index =
                sable_solver.fe.system_to_component_index(i).first;
              Assert(index < dim,
                     ExcMessage("Vector component should be less than dim!"));
              dof_touched[dof_indices[i]] = 1;
              if (!point_in_solid(solid_solver.dof_handler, support_points[i]))
                continue;
              /*Utils::CellLocator<dim, DoFHandler<dim>> locator(
                solid_solver.dof_handler, support_points[i], *(hints[i]));
              *(hints[i]) = locator.search();*/
              Utils::GridInterpolator<dim, Vector<double>> interpolator(
                solid_solver.dof_handler, support_points[i]);
              if (!interpolator.found_cell())
                {
                  std::stringstream message;
                  message << "Cannot find point in solid: " << support_points[i]
                          << std::endl;
                  AssertThrow(interpolator.found_cell(),
                              ExcMessage(message.str()));
                }
              // Solid acceleration at fluid unit point
              Vector<double> solid_acc(dim);
              Vector<double> solid_vel(dim);
              interpolator.point_value(localized_solid_acceleration, solid_acc);
              interpolator.point_value(localized_solid_velocity, solid_vel);
              Tensor<1, dim> vs;
              for (int j = 0; j < dim; ++j)
                {
                  vs[j] = solid_vel[j];
                }
              // Fluid total acceleration at support points
              Tensor<1, dim> fluid_acc =
                (vs - v[i]) / time.get_delta_t() + grad_v[i] * v[i];
              //(dv[i]) / time.get_delta_t() + grad_v[i] * v[i];
              auto line = dof_indices[i];
              // Note that we are setting the value of the constraint to the
              // velocity delta!
              tmp_fsi_acceleration(line) =
                (fluid_acc[index] - solid_acc[index]);
              tmp_fsi_velocity(line) = vs[index];
            }
        }
      // Dirichlet BCs
      if (use_dirichlet_bc)
        {
        }
    }
  tmp_fsi_acceleration.compress(VectorOperation::insert);
  sable_solver.fsi_acceleration = tmp_fsi_acceleration;
  tmp_fsi_velocity.compress(VectorOperation::insert);
  sable_solver.fsi_velocity = tmp_fsi_velocity;

  // distribute solution to the nodes which are outside solid and belongs to
  // cell which is partially inside the solid
  int cell_count = 0;
  std::vector<int> vertex_visited(sable_solver.triangulation.n_vertices(), 0);
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      if (cell_partially_inside_solid[cell_count])
        {
          // get average solution from the nodes which are inside the solid
          std::vector<double> solution_vec(dim, 0);
          std::vector<int> nodes_inside = cell_nodes_inside_solid[cell_count];
          for (unsigned int i = 0; i < nodes_inside.size(); i++)
            {
              for (unsigned int j = 0; j < dim; j++)
                {
                  int vertex_dof_index =
                    f_cell->vertex_dof_index(nodes_inside[i], j);
                  solution_vec[j] +=
                    sable_solver.fsi_velocity[vertex_dof_index] /
                    nodes_inside.size();
                }
            }

          // distribute solution to the nodes which are outside the solid
          std::vector<int> nodes_outside = cell_nodes_outside_solid[cell_count];
          for (unsigned int i = 0; i < nodes_outside.size(); i++)
            {
              int vertex_index = f_cell->vertex_index(nodes_outside[i]);
              vertex_visited[vertex_index] += 1;
              for (unsigned int j = 0; j < dim; j++)
                {
                  int vertex_dof_index =
                    f_cell->vertex_dof_index(nodes_outside[i], j);
                  sable_solver.fsi_velocity[vertex_dof_index] *=
                    (vertex_visited[vertex_index] - 1);
                  sable_solver.fsi_velocity[vertex_dof_index] +=
                    solution_vec[j];
                  // average the solution if the corresponding node is visited
                  // more than once
                  sable_solver.fsi_velocity[vertex_dof_index] /=
                    vertex_visited[vertex_index];
                }
            }
        }
      cell_count += 1;
    }
  if (use_dirichlet_bc)
    {
    }
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::find_fluid_bc_qpoints()
{
  TimerOutput::Scope timer_section(timer,
                                   "Find fluid BC based on quadrature points");
  move_solid_mesh(true);

  FEValues<dim> fe_values(sable_solver.fe,
                          sable_solver.volume_quad_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FEValues<dim> scalar_fe_values(sable_solver.scalar_fe,
                                 sable_solver.volume_quad_formula,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);

  sable_solver.system_rhs = 0;
  sable_solver.fsi_force = 0;
  sable_solver.fsi_force_acceleration_part = 0;
  sable_solver.fsi_force_stress_part = 0;

  const unsigned int dofs_per_cell = sable_solver.fe.dofs_per_cell;
  const unsigned int u_dofs = sable_solver.fe.base_element(0).dofs_per_cell;
  const unsigned int p_dofs = sable_solver.fe.base_element(1).dofs_per_cell;
  const unsigned int n_q_points = sable_solver.volume_quad_formula.size();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
              ExcMessage("Wrong partitioning of dofs!"));

  Vector<double> local_rhs(dofs_per_cell);
  Vector<double> local_rhs_acceleration_part(dofs_per_cell);
  Vector<double> local_rhs_stress_part(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<2, dim>> grad_v(n_q_points);
  std::vector<Tensor<1, dim>> v(n_q_points);
  std::vector<Tensor<1, dim>> dv(n_q_points);

  auto scalar_f_cell = sable_solver.scalar_dof_handler.begin_active();
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end(),
            scalar_f_cell != sable_solver.scalar_dof_handler.end();
       ++f_cell, ++scalar_f_cell)
    {
      auto ptr = sable_solver.cell_property.get_data(f_cell);
      const double ind = ptr[0]->indicator;
      const double ind_qpoint = ptr[0]->indicator_qpoint;
      auto s = sable_solver.cell_wise_stress.get_data(f_cell);

      if (ind == 0)
        continue;
      /*const double rho_bar =
        parameters.solid_rho * ind + s[0]->eulerian_density * (1 - ind);*/
      const double rho_bar = parameters.solid_rho * ind_qpoint +
                             s[0]->eulerian_density * (1 - ind_qpoint);

      fe_values.reinit(f_cell);
      scalar_fe_values.reinit(scalar_f_cell);

      local_rhs = 0;
      local_rhs_acceleration_part = 0;
      local_rhs_stress_part = 0;

      // Fluid velocity at support points
      fe_values[velocities].get_function_values(sable_solver.present_solution,
                                                v);
      // Fluid velocity increment at support points
      fe_values[velocities].get_function_values(sable_solver.solution_increment,
                                                dv);
      // Fluid velocity gradient at support points
      fe_values[velocities].get_function_gradients(
        sable_solver.present_solution, grad_v);

      auto q_points = fe_values.get_quadrature_points();
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          if (!point_in_solid(solid_solver.dof_handler, q_points[q]))
            continue;

          Utils::GridInterpolator<dim, Vector<double>> interpolator(
            solid_solver.dof_handler, q_points[q]);
          if (!interpolator.found_cell())
            {
              std::stringstream message;
              message << "Cannot find point in solid: " << q_points[q]
                      << std::endl;
              AssertThrow(interpolator.found_cell(), ExcMessage(message.str()));
            }
          // Solid acceleration at fluid unit point
          Vector<double> solid_acc(dim);
          Vector<double> solid_vel(dim);
          interpolator.point_value(solid_solver.current_acceleration,
                                   solid_acc);
          interpolator.point_value(solid_solver.current_velocity, solid_vel);
          Tensor<1, dim> vs;
          Tensor<1, dim> solid_acc_tensor;
          for (int j = 0; j < dim; ++j)
            {
              vs[j] = solid_vel[j];
              solid_acc_tensor[j] = solid_acc[j];
            }

          // Fluid total acceleration at support points
          Tensor<1, dim> fluid_acc_tensor =
            (vs - v[q]) / time.get_delta_t() + grad_v[q] * v[q];
          //(dv[q]) / time.get_delta_t() + grad_v[q] * v[q];
          // calculate FSI acceleration
          Tensor<1, dim> fsi_acc_tensor;
          fsi_acc_tensor = fluid_acc_tensor;
          fsi_acc_tensor -= solid_acc_tensor;

          SymmetricTensor<2, dim> f_cell_stress;
          int count = 0;
          for (unsigned int k = 0; k < dim; k++)
            {
              for (unsigned int m = k; m < dim; m++)
                {
                  f_cell_stress[k][m] = s[0]->cell_stress[count];
                  count++;
                }
            }

          // Create the scalar interpolator for stresses based on the
          // existing interpolator
          auto s_cell = interpolator.get_cell();
          TriaActiveIterator<DoFCellAccessor<dim, dim, false>> scalar_s_cell(
            &solid_solver.triangulation,
            s_cell->level(),
            s_cell->index(),
            &solid_solver.scalar_dof_handler);
          Utils::GridInterpolator<dim, Vector<double>> scalar_interpolator(
            solid_solver.scalar_dof_handler, q_points[q], {}, scalar_s_cell);

          SymmetricTensor<2, dim> s_cell_stress;
          for (unsigned int k = 0; k < dim; k++)
            {
              for (unsigned int m = k; m < dim; m++)
                {

                  Vector<double> s_stress_component(1);
                  scalar_interpolator.point_value(solid_solver.stress[k][m],
                                                  s_stress_component);
                  s_cell_stress[k][m] = s_stress_component[0];
                }
            }
          // calculate FSI stress
          SymmetricTensor<2, dim> fsi_stress_tensor;
          fsi_stress_tensor = f_cell_stress;
          fsi_stress_tensor -= s_cell_stress;

          // assemble FSI force
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              div_phi_u[k] = fe_values[velocities].divergence(k, q);
              grad_phi_u[k] = fe_values[velocities].gradient(k, q);
              phi_u[k] = fe_values[velocities].value(k, q);
              phi_p[k] = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {

              local_rhs(i) +=
                (scalar_product(grad_phi_u[i], fsi_stress_tensor) +
                 rho_bar * fsi_acc_tensor * phi_u[i]) *
                fe_values.JxW(q);
              local_rhs_acceleration_part(i) +=
                (rho_bar * fsi_acc_tensor * phi_u[i]) * fe_values.JxW(q);
              local_rhs_stress_part(i) +=
                (scalar_product(grad_phi_u[i], fsi_stress_tensor)) *
                fe_values.JxW(q);
            }
        }

      f_cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          sable_solver.system_rhs[local_dof_indices[i]] += local_rhs(i);
          sable_solver.fsi_force[local_dof_indices[i]] += local_rhs(i);
          sable_solver.fsi_force_acceleration_part[local_dof_indices[i]] +=
            local_rhs_acceleration_part(i);
          sable_solver.fsi_force_stress_part[local_dof_indices[i]] +=
            local_rhs_stress_part(i);
        }
    }

  // Interpolate velocity to the nodes inside Lagrangian solid
  const std::vector<Point<dim>> &unit_points =
    sable_solver.fe.get_unit_support_points();

  BlockVector<double> tmp_fsi_velocity;
  tmp_fsi_velocity.reinit(sable_solver.dofs_per_block);

  MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
  Quadrature<dim> dummy_q(unit_points);
  FEValues<dim> dummy_fe_values(mapping,
                                sable_solver.fe,
                                dummy_q,
                                update_quadrature_points | update_values |
                                  update_gradients);
  std::vector<types::global_dof_index> dof_indices(
    sable_solver.fe.dofs_per_cell);
  std::vector<unsigned int> dof_touched(sable_solver.dof_handler.n_dofs(), 0);

  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {

      auto ptr = sable_solver.cell_property.get_data(f_cell);
      if (ptr[0]->indicator == 0)
        continue;

      dummy_fe_values.reinit(f_cell);
      f_cell->get_dof_indices(dof_indices);
      auto support_points = dummy_fe_values.get_quadrature_points();
      // Loop over the support points to calculate fsi acceleration.
      for (unsigned int i = 0; i < unit_points.size(); ++i)
        {
          // Skip the already-set dofs.
          if (dof_touched[dof_indices[i]] != 0)
            continue;
          auto base_index = sable_solver.fe.system_to_base_index(i);
          const unsigned int i_group = base_index.first.first;
          Assert(
            i_group < 2,
            ExcMessage("There should be only 2 groups of finite element!"));
          if (i_group == 1)
            continue; // skip the pressure dofs
          // Same as fluid_solver.fe.system_to_base_index(i).first.second;
          const unsigned int index =
            sable_solver.fe.system_to_component_index(i).first;
          Assert(index < dim,
                 ExcMessage("Vector component should be less than dim!"));
          dof_touched[dof_indices[i]] = 1;
          if (!point_in_solid(solid_solver.dof_handler, support_points[i]))
            continue;
          Utils::GridInterpolator<dim, Vector<double>> interpolator(
            solid_solver.dof_handler, support_points[i]);
          if (!interpolator.found_cell())
            {
              std::stringstream message;
              message << "Cannot find point in solid: " << support_points[i]
                      << std::endl;
              AssertThrow(interpolator.found_cell(), ExcMessage(message.str()));
            }
          // Solid velocity at fluid unit point
          Vector<double> solid_vel_nodal(dim);
          interpolator.point_value(solid_solver.current_velocity,
                                   solid_vel_nodal);
          auto line = dof_indices[i];
          tmp_fsi_velocity(line) = solid_vel_nodal[index];
        }
    }

  tmp_fsi_velocity.compress(VectorOperation::insert);
  sable_solver.fsi_velocity = tmp_fsi_velocity;

  // distribute solution to the nodes which are outside solid and belongs to
  // cell which is partially inside the solid
  int cell_count = 0;
  std::vector<int> vertex_visited(sable_solver.triangulation.n_vertices(), 0);
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      auto ptr = sable_solver.cell_property.get_data(f_cell);
      std::vector<int> nodes_inside = cell_nodes_inside_solid[cell_count];
      if (nodes_inside.size() > 0 && ptr[0]->indicator != 0)
        {
          // get average solution from the nodes which are inside the solid
          std::vector<double> solution_vec(dim, 0);
          for (unsigned int i = 0; i < nodes_inside.size(); i++)
            {
              for (unsigned int j = 0; j < dim; j++)
                {
                  int vertex_dof_index =
                    f_cell->vertex_dof_index(nodes_inside[i], j);
                  solution_vec[j] +=
                    sable_solver.fsi_velocity[vertex_dof_index] /
                    nodes_inside.size();
                }
            }

          // distribute solution to the nodes which are outside the solid
          std::vector<int> nodes_outside = cell_nodes_outside_solid[cell_count];
          for (unsigned int i = 0; i < nodes_outside.size(); i++)
            {
              int vertex_index = f_cell->vertex_index(nodes_outside[i]);
              vertex_visited[vertex_index] += 1;
              for (unsigned int j = 0; j < dim; j++)
                {
                  int vertex_dof_index =
                    f_cell->vertex_dof_index(nodes_outside[i], j);
                  sable_solver.fsi_velocity[vertex_dof_index] *=
                    (vertex_visited[vertex_index] - 1);
                  sable_solver.fsi_velocity[vertex_dof_index] +=
                    solution_vec[j];
                  // average the solution if the corresponding node is visited
                  // more than once
                  sable_solver.fsi_velocity[vertex_dof_index] /=
                    vertex_visited[vertex_index];
                }
            }
        }
      cell_count += 1;
    }

  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::find_solid_bc()
{
  TimerOutput::Scope timer_section(timer, "Find solid BC");
  // Must use the updated solid coordinates
  move_solid_mesh(true);
  // Fluid FEValues to do interpolation
  FEValues<dim> fe_values(
    sable_solver.fe, sable_solver.volume_quad_formula, update_values);
  // Solid FEFaceValues to get the normal at face center
  FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                   solid_solver.face_quad_formula,
                                   update_quadrature_points |
                                     update_normal_vectors);
  // Get Eulerian cell size
  // Note: Only works for unifrom, strucutred mesh
  auto e_cell = sable_solver.triangulation.begin_active();
  double h = abs(e_cell->vertex(0)(0) - e_cell->vertex(1)(0));

  const unsigned int n_f_q_points = solid_solver.face_quad_formula.size();

  for (auto s_cell = solid_solver.dof_handler.begin_active();
       s_cell != solid_solver.dof_handler.end();
       ++s_cell)
    {
      auto ptr = solid_solver.cell_property.get_data(s_cell);
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          // Current face is at boundary and without Dirichlet bc.
          if (s_cell->face(f)->at_boundary())
            {
              ptr[f]->fsi_traction.clear();
              fe_face_values.reinit(s_cell, f);

              for (unsigned int q = 0; q < n_f_q_points; ++q)
                {
                  Point<dim> q_point = fe_face_values.quadrature_point(q);
                  Tensor<1, dim> normal = fe_face_values.normal_vector(q);
                  // hard coded parameter value to scale the distance along
                  // the face normal
                  double beta = parameters.solid_traction_extension_scale;
                  double d = h * beta;
                  // Find a point at a distance d from q_point along the face
                  // normal
                  Point<dim> q_point_extension;
                  for (unsigned int i = 0; i < dim; i++)
                    q_point_extension(i) = q_point(i) + d * normal[i];
                  // Get interpolated solution from the fluid
                  Vector<double> value(dim + 1);
                  Utils::GridInterpolator<dim, BlockVector<double>>
                    interpolator(sable_solver.dof_handler, q_point_extension);
                  interpolator.point_value(sable_solver.present_solution,
                                           value);
                  // Create the scalar interpolator for stresses based on the
                  // existing interpolator
                  auto f_cell = interpolator.get_cell();
                  // If the quadrature point is outside background mesh
                  if (f_cell->index() == -1)
                    {
                      Tensor<1, dim> zero_tensor;
                      ptr[f]->fsi_traction.push_back(zero_tensor);
                      continue;
                    }
                  // get cell-wise stress from SABLE
                  auto ptr_f = sable_solver.cell_wise_stress.get_data(f_cell);
                  TriaActiveIterator<DoFCellAccessor<dim, dim, false>>
                    scalar_f_cell(&sable_solver.triangulation,
                                  f_cell->level(),
                                  f_cell->index(),
                                  &sable_solver.scalar_dof_handler);
                  Utils::GridInterpolator<dim, Vector<double>>
                    scalar_interpolator(sable_solver.scalar_dof_handler,
                                        q_point,
                                        {},
                                        scalar_f_cell);
                  SymmetricTensor<2, dim> viscous_stress;
                  int count = 0;
                  for (unsigned int i = 0; i < dim; i++)
                    {
                      for (unsigned int j = i; j < dim; j++)
                        {
                          // Interpolate stress from nodal stress field
                          if (parameters.traction_calculation_option ==
                              "NodeBased")
                            {
                              Vector<double> stress_component(1);
                              scalar_interpolator.point_value(
                                sable_solver.stress[i][j], stress_component);
                              viscous_stress[i][j] = stress_component[0];
                            }
                          else
                            {
                              // Get cell-wise stress
                              viscous_stress[i][j] =
                                ptr_f[0]->cell_stress_no_bgmat[count];
                            }
                          count++;
                        }
                    }
                  // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                  SymmetricTensor<2, dim> stress =
                    -value[dim] * Physics::Elasticity::StandardTensors<dim>::I +
                    viscous_stress;
                  ptr[f]->fsi_traction.push_back(stress * normal);
                }
            }
        }
    }
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::run()
{
  // global refinement in sable solver is not possible as it would change the
  // order of nodes
  solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
  sable_solver.setup_dofs();
  sable_solver.make_constraints();
  sable_solver.initialize_system();
  solid_solver.setup_dofs();
  solid_solver.initialize_system();

  std::cout << "Number of fluid active cells and dofs: ["
            << sable_solver.triangulation.n_active_cells() << ", "
            << sable_solver.dof_handler.n_dofs() << "]" << std::endl
            << "Number of solid active cells and dofs: ["
            << solid_solver.triangulation.n_active_cells() << ", "
            << solid_solver.dof_handler.n_dofs() << "]" << std::endl;

  // Time loop.
  // use_nonzero_constraints is set to true only at the first time step,
  // which means nonzero_constraints will be applied at the first iteration
  // in the first time step only, and never be used again.
  // This corresponds to time-independent Dirichlet BCs.
  bool first_step = true;
  while (sable_solver.is_comm_active)
    {
      // send initial solution
      if (time.current() == 0)
        {
          sable_solver.run_one_step();
        }
      // get dt from Sable
      sable_solver.get_dt_sable();
      time.set_delta_t(sable_solver.time.get_delta_t());
      solid_solver.time.set_delta_t(sable_solver.time.get_delta_t());
      time.increment();
      find_solid_bc();
      solid_solver.run_one_step(first_step);
      // indicator field
      update_solid_box();
      if (parameters.fsi_force_criteria == "Nodes")
        {
          update_indicator();
          find_fluid_bc();
        }
      else
        {
          update_indicator_qpoints();
          find_fluid_bc_qpoints();
        }
      sable_solver.send_fsi_force(sable_solver.sable_no_nodes);
      sable_solver.send_indicator(sable_solver.sable_no_ele,
                                  sable_solver.sable_no_nodes);
      // send_indicator_field
      sable_solver.run_one_step();
      first_step = false;
    }
}

template class OpenIFEM_Sable_FSI<2>;
template class OpenIFEM_Sable_FSI<3>;
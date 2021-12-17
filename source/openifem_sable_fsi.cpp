#include "openifem_sable_fsi.h"
#include <complex>
#include <iostream>

template <int dim>
OpenIFEM_Sable_FSI<dim>::~ OpenIFEM_Sable_FSI()
{
  timer.print_summary();
}

template <int dim>
OpenIFEM_Sable_FSI<dim>::OpenIFEM_Sable_FSI(Fluid::SableWrap<dim> &f,
                                            Solid::SolidSolver<dim> &s,
                                            const Parameters::AllParameters &p,
                                            bool use_dirichlet_bc)
  :FSI<dim>(f, s, p, use_dirichlet_bc), sable_solver(f)
{
  assert(use_dirichlet_bc==false);
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
  for(unsigned int i=0; i<sable_solver.triangulation.n_active_cells();i++)
    cell_partially_inside_solid.push_back(false);

  // set condition for the indicator field
  // cell is aritifical if nodes_inside_solid > min_nodes_inside
  unsigned int min_nodes_inside = GeometryInfo<dim>::vertices_per_cell -1;
  if(parameters.indicator_field_condition == "CompletelyInsideSolid")
  {
    min_nodes_inside =  GeometryInfo<dim>::vertices_per_cell -1;
  }
  else if(parameters.indicator_field_condition == "PartiallyInsideSolid")
    min_nodes_inside = (dim == 2 ? 1 : 3);

  move_solid_mesh(true);
  int cell_count =0;
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
      p[0]->indicator =
        (inside_count >  min_nodes_inside ? 1 : 0);

      // cell is partially inside the solid  
      if((inside_count > min_nodes_inside) && (inside_count < GeometryInfo<dim>::vertices_per_cell))
      {
        cell_partially_inside_solid[cell_count]=true;
        //store local node ids which are inside and outside the solid
        cell_nodes_inside_solid.insert({cell_count, inside_nodes});
        cell_nodes_outside_solid.insert({cell_count, outside_nodes});
      }
      cell_count += 1;  
    }
  move_solid_mesh(false);
}

// This function interpolates the solid velocity into the fluid solver,
// as the Dirichlet boundary conditions for artificial fluid vertices
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
  //inner_nonzero.reinit(fluid_solver.locally_relevant_dofs);
  //inner_zero.reinit(fluid_solver.locally_relevant_dofs);
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
  std::vector<unsigned int> scalar_dof_touched(sable_solver.scalar_dof_handler.n_dofs(), 0);
  std::vector<double> f_stress_component(scalar_unit_points.size());  
  std::vector<std::vector<double>> f_cell_stress = std::vector<std::vector<double>>(sable_solver.fsi_stress.size(),std::vector<double>(scalar_unit_points.size()));

  for (auto scalar_cell = sable_solver.scalar_dof_handler.begin_active(); scalar_cell != sable_solver.scalar_dof_handler.end();
       ++scalar_cell)
  {
    auto ptr = sable_solver.cell_property.get_data(scalar_cell);
    if (ptr[0]->indicator == 0)
      continue;
    scalar_cell->get_dof_indices(scalar_dof_indices);
    scalar_fe_values.reinit(scalar_cell);

    int stress_index=0;
    // get fluid stress at support points
    for (unsigned int i = 0; i < dim; i++)
    {
      for (unsigned int j = i; j < dim; j++)
        {
          scalar_fe_values.get_function_values(sable_solver.stress[i][j], f_stress_component);
          f_cell_stress[stress_index]= f_stress_component;
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
      stress_index=0;
      for (unsigned int j = 0; j < dim; j++)
      {
        for (unsigned int k = j; k < dim; k++)
        {
          Vector<double> s_stress_component(1);
          scalar_interpolator.point_value(sable_solver.stress[j][k],
                                                      s_stress_component);
          sable_solver.fsi_stress[stress_index][scalar_dof_indices[i]]= f_cell_stress[stress_index][i] - s_stress_component[0];
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

          //auto hints = cell_hints.get_data(f_cell);
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
              Assert(i_group < 2,
                     ExcMessage(
                       "There should be only 2 groups of finite element!"));
              if (i_group == 1)
                continue; // skip the pressure dofs
              // Same as fluid_solver.fe.system_to_base_index(i).first.second;
              const unsigned int index =
                sable_solver.fe.system_to_component_index(i).first;
              Assert(index < dim,
                     ExcMessage("Vector component should be less than dim!"));
              dof_touched[dof_indices[i]] = 1;
              if (!point_in_solid(solid_solver.dof_handler,
                                  support_points[i]))
                continue;
              /*Utils::CellLocator<dim, DoFHandler<dim>> locator(
                solid_solver.dof_handler, support_points[i], *(hints[i]));
              *(hints[i]) = locator.search();*/
              Utils::GridInterpolator<dim, Vector<double>> interpolator(
                solid_solver.dof_handler, support_points[i]);
              if (!interpolator.found_cell())
                {
                  std::stringstream message;
                  message
                    << "Cannot find point in solid: " << support_points[i]
                    << std::endl;
                  AssertThrow(interpolator.found_cell(),
                              ExcMessage(message.str()));
                }
              // Solid acceleration at fluid unit point
              Vector<double> solid_acc(dim);
              Vector<double> solid_vel(dim);
              interpolator.point_value(localized_solid_acceleration,
                                       solid_acc);
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
                parameters.solid_rho*(fluid_acc[index] - solid_acc[index]);
              tmp_fsi_velocity(line) =
                vs[index];  

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

  // distribute solution to the nodes which are outside solid and belongs to cell which is partially inside the solid
  int cell_count=0;
  std::vector<int> vertex_visited(sable_solver.triangulation.n_vertices(),0); 
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end();
       ++f_cell)
    {
      if(cell_partially_inside_solid[cell_count])
      {
        // get average solution from the nodes which are inside the solid  
        std::vector<double> solution_vec(dim, 0);
        std::vector<int> nodes_inside = cell_nodes_inside_solid[cell_count];
        for(unsigned int i=0; i< nodes_inside.size(); i++)
        {
          for(unsigned int j=0; j< dim; j++)
          {
            int vertex_dof_index = f_cell->vertex_dof_index(nodes_inside[i],j);
            solution_vec[j] += sable_solver.fsi_velocity[vertex_dof_index]/nodes_inside.size(); 
          }  
        }

        // distribute solution to the nodes which are outside the solid
        std::vector<int> nodes_outside = cell_nodes_outside_solid[cell_count];
        for(unsigned int i=0; i< nodes_outside.size(); i++)
        {
          int vertex_index = f_cell->vertex_index(nodes_outside[i]);
          vertex_visited[vertex_index] += 1;
          for(unsigned int j=0; j< dim; j++)
          {
            int vertex_dof_index = f_cell->vertex_dof_index(nodes_outside[i],j);
            sable_solver.fsi_velocity[vertex_dof_index] *= (vertex_visited[vertex_index]-1);
            sable_solver.fsi_velocity[vertex_dof_index] += solution_vec[j];
            // average the solution if the corresponding node is visited more than once
            sable_solver.fsi_velocity[vertex_dof_index] /= vertex_visited[vertex_index];
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
void OpenIFEM_Sable_FSI<dim>::run()
{
  // global refinement in sable solver is not possible as it would change the order of nodes 
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
      //send initial solution
      if(time.current()==0)
      {
        sable_solver.run_one_step(true);
      }  
      //get dt from Sable
      sable_solver.get_dt_sable();
      time.set_delta_t(sable_solver.time.get_delta_t());
      solid_solver.time.set_delta_t(sable_solver.time.get_delta_t());
      time.increment();
      find_solid_bc();
      solid_solver.run_one_step(first_step);
      //indicator field
      update_solid_box();
      update_indicator();
      find_fluid_bc();
      sable_solver.send_fsi_force(sable_solver.sable_no_nodes, sable_solver.sable_no_nodes_one_dir);
      sable_solver.send_indicator(sable_solver.sable_no_ele, sable_solver.sable_no_nodes, sable_solver.sable_no_nodes_one_dir);
      //send_indicator_field
      sable_solver.run_one_step(false);
      first_step=false;
    }
}

template class OpenIFEM_Sable_FSI<2>;
template class OpenIFEM_Sable_FSI<3>;

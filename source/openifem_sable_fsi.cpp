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
  move_solid_mesh(true);
  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto p = fluid_solver.cell_property.get_data(f_cell);
      int inside_count = 0;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (!point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
            {
              break;
            }
          ++inside_count;
        }
      p[0]->indicator =
        (inside_count == GeometryInfo<dim>::vertices_per_cell ? 1 : 0);
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
              auto line = dof_indices[i];
              // Note that we are setting the value of the constraint to the
              // velocity delta!
              tmp_fsi_acceleration(line) =
                parameters.solid_rho*(fluid_acc[index] - solid_acc[index]);
            }
        }
      // Dirichlet BCs
      if (use_dirichlet_bc)
        {
          
        }
    }
  tmp_fsi_acceleration.compress(VectorOperation::insert);
  sable_solver.fsi_acceleration = tmp_fsi_acceleration;
  if (use_dirichlet_bc)
    {
      
    }
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::run()
{
  
  sable_solver.triangulation.refine_global(parameters.global_refinements[0]);
  sable_solver.setup_dofs();
  sable_solver.make_constraints();
  sable_solver.initialize_system();
  solid_solver.setup_dofs();
  solid_solver.initialize_system();

  std::cout << "Number of fluid active cells and dofs: ["
            << fluid_solver.triangulation.n_active_cells() << ", "
            << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
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
      sable_solver.send_indicator(sable_solver.sable_no_ele);
      //send_indicator_field
      sable_solver.run_one_step(false);
      first_step=false;
    }
}

template class OpenIFEM_Sable_FSI<2>;
template class OpenIFEM_Sable_FSI<3>;

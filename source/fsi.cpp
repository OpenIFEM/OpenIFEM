#include "fsi.h"
#include <complex>
#include <iostream>

template <int dim>
FSI<dim>::~FSI()
{
  timer.print_summary();
}

template <int dim>
FSI<dim>::FSI(Fluid::FluidSolver<dim> &f,
              Solid::SolidSolver<dim> &s,
              const Parameters::AllParameters &p,
              bool use_dirichlet_bc)
  : fluid_solver(f),
    solid_solver(s),
    parameters(p),
    time(parameters.end_time,
         parameters.time_step,
         parameters.output_interval,
         parameters.refinement_interval,
         parameters.save_interval),
    timer(std::cout, TimerOutput::never, TimerOutput::wall_times),
    use_dirichlet_bc(use_dirichlet_bc)
{
  solid_box.reinit(2 * dim);
}

template <int dim>
void FSI<dim>::move_solid_mesh(bool move_forward)
{
  TimerOutput::Scope timer_section(timer, "Move solid mesh");
  std::vector<bool> vertex_touched(solid_solver.triangulation.n_vertices(),
                                   false);
  for (auto cell = solid_solver.dof_handler.begin_active();
       cell != solid_solver.dof_handler.end();
       ++cell)
    {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (!vertex_touched[cell->vertex_index(v)])
            {
              vertex_touched[cell->vertex_index(v)] = true;
              Point<dim> vertex_displacement;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  vertex_displacement[d] = solid_solver.current_displacement(
                    cell->vertex_dof_index(v, d));
                }
              if (move_forward)
                {
                  cell->vertex(v) += vertex_displacement;
                }
              else
                {
                  cell->vertex(v) -= vertex_displacement;
                }
            }
        }
    }
}

template <int dim>
void FSI<dim>::update_solid_box()
{
  move_solid_mesh(true);
  solid_box = 0;
  for (unsigned int i = 0; i < dim; ++i)
    {
      solid_box(2 * i) =
        solid_solver.triangulation.get_vertices().begin()->operator()(i);
      solid_box(2 * i + 1) =
        solid_solver.triangulation.get_vertices().begin()->operator()(i);
    }
  for (auto v = solid_solver.triangulation.get_vertices().begin();
       v != solid_solver.triangulation.get_vertices().end();
       ++v)
    {
      for (unsigned int i = 0; i < dim; ++i)
        {
          if ((*v)(i) < solid_box(2 * i))
            solid_box(2 * i) = (*v)(i);
          else if ((*v)(i) > solid_box(2 * i + 1))
            solid_box(2 * i + 1) = (*v)(i);
        }
    }
  move_solid_mesh(false);
}

template <int dim>
bool FSI<dim>::point_in_solid(const DoFHandler<dim> &df,
                              const Point<dim> &point)
{
  // Check whether the point is in the solid box first.
  for (unsigned int i = 0; i < dim; ++i)
    {
      if (point(i) < solid_box(2 * i) || point(i) > solid_box(2 * i + 1))
        return false;
    }
  for (auto cell = df.begin_active(); cell != df.end(); ++cell)
    {
      if (cell->point_inside(point))
        {
          return true;
        }
    }
  return false;
}

template <int dim>
void FSI<dim>::update_solid_displacement()
{
  move_solid_mesh(true);
  auto displacement = solid_solver.current_displacement;
  std::vector<bool> vertex_touched(solid_solver.dof_handler.n_dofs(), false);
  for (auto cell : solid_solver.dof_handler.active_cell_iterators())
    {
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (!vertex_touched[cell->vertex_index(v)] &&
              !solid_solver.constraints.is_constrained(cell->vertex_index(v)))
            {
              vertex_touched[cell->vertex_index(v)] = true;
              Point<dim> point = cell->vertex(v);
              Vector<double> tmp(dim + 1);
              VectorTools::point_value(fluid_solver.dof_handler,
                                       fluid_solver.present_solution,
                                       point,
                                       tmp);
              for (unsigned int d = 0; d < dim; ++d)
                {
                  displacement[cell->vertex_dof_index(v, d)] +=
                    tmp[d] * time.get_delta_t();
                }
            }
        }
    }
  move_solid_mesh(false);
  solid_solver.current_displacement = displacement;
}

// Dirichlet bcs are applied to artificial fluid cells, so fluid nodes should
// be marked as artificial or real. Meanwhile, additional body force is
// applied to the artificial fluid quadrature points. To accommodate these two
// settings, we define indicator at quadrature points, but only when all
// of the vertices of a fluid cell are found to be in solid domain,
// set the indicators at all quadrature points to be 1.
template <int dim>
void FSI<dim>::update_indicator()
{
  TimerOutput::Scope timer_section(timer, "Update indicator");
  move_solid_mesh(true);
  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto p = fluid_solver.cell_property.get_data(f_cell);
      auto center = f_cell->center();
      p[0]->indicator = point_in_solid(solid_solver.dof_handler, center);
    }
  move_solid_mesh(false);
}

// This function interpolates the solid velocity into the fluid solver,
// as the Dirichlet boundary conditions for artificial fluid vertices
template <int dim>
void FSI<dim>::find_fluid_bc()
{
  TimerOutput::Scope timer_section(timer, "Find fluid BC");
  move_solid_mesh(true);

  // The nonzero Dirichlet BCs (to set the velocity) and zero Dirichlet
  // BCs (to set the velocity increment) for the artificial fluid domain.
  AffineConstraints<double> inner_nonzero, inner_zero;
  inner_nonzero.clear();
  inner_zero.clear();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<SymmetricTensor<2, dim>> sym_grad_v(1);
  std::vector<double> p(1);
  std::vector<Tensor<2, dim>> grad_v(1);
  std::vector<Tensor<1, dim>> v(1);
  std::vector<Tensor<1, dim>> dv(1);

  // Cell center in unit coordinate system
  Point<dim> unit_center;
  for (unsigned int i = 0; i < dim; ++i)
    {
      unit_center[i] = 0.5;
    }
  Quadrature<dim> quad(unit_center);
  MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
  FEValues<dim> fe_values(mapping,
                          fluid_solver.fe,
                          quad,
                          update_quadrature_points | update_values |
                            update_gradients);

  const std::vector<Point<dim>> &unit_points =
    fluid_solver.fe.get_unit_support_points();
  Quadrature<dim> dummy_q(unit_points);
  FEValues<dim> dummy_fe_values(
    mapping, fluid_solver.fe, dummy_q, update_quadrature_points);
  std::vector<types::global_dof_index> dof_indices(
    fluid_solver.fe.dofs_per_cell);

  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto ptr = fluid_solver.cell_property.get_data(f_cell);
      ptr[0]->fsi_acceleration = 0;
      ptr[0]->fsi_stress = 0;
      if (!use_dirichlet_bc && ptr[0]->indicator == 1)
        {
          fe_values.reinit(f_cell);
          // Fluid velocity increment at cell center
          fe_values[velocities].get_function_values(
            fluid_solver.solution_increment, dv);
          // Fluid velocity gradient at cell center
          fe_values[velocities].get_function_gradients(
            fluid_solver.present_solution, grad_v);
          // Fluid symmetric velocity gradient at cell center
          fe_values[velocities].get_function_symmetric_gradients(
            fluid_solver.present_solution, sym_grad_v);
          // Fluid pressure at cell center
          fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                  p);
          // Real coordinates of fluid cell center
          auto point = fe_values.get_quadrature_points()[0];
          // Solid acceleration at fluid cell center
          Vector<double> solid_acc(dim);
          VectorTools::point_value(solid_solver.dof_handler,
                                   solid_solver.current_acceleration,
                                   point,
                                   solid_acc);
          // Fluid total acceleration at cell center
          Tensor<1, dim> fluid_acc =
            dv[0] / time.get_delta_t() + grad_v[0] * v[0];
          (void)fluid_acc;
          // FSI acceleration term:
          for (unsigned int i = 0; i < dim; ++i)
            {
              ptr[0]->fsi_acceleration[i] =
                (parameters.solid_rho - parameters.fluid_rho) *
                (parameters.gravity[i] - solid_acc[i]);
            }
        }
      // Dirichlet BCs
      if (use_dirichlet_bc)
        {
          dummy_fe_values.reinit(f_cell);
          f_cell->get_dof_indices(dof_indices);
          auto support_points = dummy_fe_values.get_quadrature_points();
          // Loop over the support points to set Dirichlet BCs.
          for (unsigned int i = 0; i < unit_points.size(); ++i)
            {
              auto base_index = fluid_solver.fe.system_to_base_index(i);
              const unsigned int i_group = base_index.first.first;
              Assert(
                i_group < 2,
                ExcMessage("There should be only 2 groups of finite element!"));
              if (i_group == 1)
                continue; // skip the pressure dofs
              bool inside = true;
              for (unsigned int d = 0; d < dim; ++d)
                if (std::abs(unit_points[i][d]) < 1e-5)
                  {
                    inside = false;
                    break;
                  }
              if (inside)
                continue; // skip the in-cell support point
              // Same as fluid_solver.fe.system_to_base_index(i).first.second;
              const unsigned int index =
                fluid_solver.fe.system_to_component_index(i).first;
              Assert(index < dim,
                     ExcMessage("Vector component should be less than dim!"));
              if (!point_in_solid(solid_solver.dof_handler, support_points[i]))
                continue;
              Vector<double> fluid_velocity(dim);
              VectorTools::point_value(solid_solver.dof_handler,
                                       solid_solver.current_velocity,
                                       support_points[i],
                                       fluid_velocity);
              auto line = dof_indices[i];
              inner_nonzero.add_line(line);
              inner_zero.add_line(line);
              // Note that we are setting the value of the constraint to the
              // velocity delta!
              inner_nonzero.set_inhomogeneity(
                line,
                fluid_velocity[index] - fluid_solver.present_solution(line));
            }
        }
    }
  if (use_dirichlet_bc)
    {
      inner_nonzero.close();
      inner_zero.close();
      fluid_solver.nonzero_constraints.merge(
        inner_nonzero,
        AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
      fluid_solver.zero_constraints.merge(
        inner_zero,
        AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    }
  move_solid_mesh(false);
}

template <int dim>
void FSI<dim>::find_solid_bc()
{
  TimerOutput::Scope timer_section(timer, "Find solid BC");
  // Must use the updated solid coordinates
  move_solid_mesh(true);
  // Fluid FEValues to do interpolation
  FEValues<dim> fe_values(
    fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);
  // Solid FEFaceValues to get the normal at face center
  Point<dim - 1> unit_face_center;
  for (unsigned int i = 0; i < dim - 1; ++i)
    {
      unit_face_center[i] = 0.5;
    }
  Quadrature<dim - 1> center_quad(unit_face_center);
  FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                   unit_face_center,
                                   update_quadrature_points |
                                     update_normal_vectors);

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
              fe_face_values.reinit(s_cell, f);
              Point<dim> q_point = fe_face_values.quadrature_point(0);
              Tensor<1, dim> normal = fe_face_values.normal_vector(0);
              // Get interpolated solution from the fluid
              Vector<double> value(dim + 1);
              Utils::GridInterpolator<dim, BlockVector<double>> interpolator(
                fluid_solver.dof_handler, q_point);
              interpolator.point_value(fluid_solver.present_solution, value);
              // Create the scalar interpolator for stresses based on the
              // existing interpolator
              auto f_cell = interpolator.get_cell();
              TriaActiveIterator<DoFCellAccessor<dim, dim, false>>
                scalar_f_cell(&fluid_solver.triangulation,
                              f_cell->level(),
                              f_cell->index(),
                              &fluid_solver.scalar_dof_handler);
              Utils::GridInterpolator<dim, Vector<double>> scalar_interpolator(
                fluid_solver.scalar_dof_handler, q_point, {}, scalar_f_cell);
              SymmetricTensor<2, dim> viscous_stress;
              for (unsigned int i = 0; i < dim; i++)
                {
                  for (unsigned int j = i; j < dim; j++)
                    {
                      Vector<double> stress_component(1);
                      scalar_interpolator.point_value(fluid_solver.stress[i][j],
                                                      stress_component);
                      viscous_stress[i][j] = stress_component[0];
                    }
                }
              // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
              SymmetricTensor<2, dim> stress =
                -value[dim] * Physics::Elasticity::StandardTensors<dim>::I +
                viscous_stress;
              ptr[f]->fsi_traction = stress * normal;
            }
        }
    }
  move_solid_mesh(false);
}

template <int dim>
void FSI<dim>::refine_mesh(const unsigned int min_grid_level,
                           const unsigned int max_grid_level)
{
  TimerOutput::Scope timer_section(timer, "Refine mesh");
  move_solid_mesh(true);
  std::vector<Point<dim>> solid_boundary_points;
  for (auto s_cell : solid_solver.dof_handler.active_cell_iterators())
    {
      bool is_boundary = false;
      Point<dim> point;
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        {
          if (s_cell->face(face)->at_boundary())
            {
              point = s_cell->face(face)->center();
              is_boundary = true;
              break;
            }
        }
      if (is_boundary)
        {
          solid_boundary_points.push_back(point);
        }
    }
  for (auto f_cell : fluid_solver.dof_handler.active_cell_iterators())
    {
      auto center = f_cell->center();
      double dist = 1000;
      for (auto point : solid_boundary_points)
        {
          dist = std::min(center.distance(point), dist);
        }
      if (dist < f_cell->diameter())
        f_cell->set_refine_flag();
      else
        f_cell->set_coarsen_flag();
    }
  move_solid_mesh(false);
  if (fluid_solver.triangulation.n_levels() > max_grid_level)
    {
      for (auto cell = fluid_solver.triangulation.begin_active(max_grid_level);
           cell != fluid_solver.triangulation.end();
           ++cell)
        {
          cell->clear_refine_flag();
        }
    }

  for (auto cell = fluid_solver.triangulation.begin_active(min_grid_level);
       cell != fluid_solver.triangulation.end_active(min_grid_level);
       ++cell)
    {
      cell->clear_coarsen_flag();
    }

  BlockVector<double> buffer(fluid_solver.present_solution);
  SolutionTransfer<dim, BlockVector<double>> solution_transfer(
    fluid_solver.dof_handler);

  fluid_solver.triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(buffer);

  fluid_solver.triangulation.execute_coarsening_and_refinement();

  fluid_solver.setup_dofs();
  fluid_solver.make_constraints();
  fluid_solver.initialize_system();

  solution_transfer.interpolate(buffer, fluid_solver.present_solution);
  fluid_solver.nonzero_constraints.distribute(fluid_solver.present_solution);
}

template <int dim>
void FSI<dim>::run()
{
  solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
  solid_solver.setup_dofs();
  solid_solver.initialize_system();
  fluid_solver.triangulation.refine_global(parameters.global_refinements[0]);
  fluid_solver.setup_dofs();
  fluid_solver.make_constraints();
  fluid_solver.initialize_system();

  std::cout << "Number of fluid active cells and dofs: ["
            << fluid_solver.triangulation.n_active_cells() << ", "
            << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
            << "Number of solid active cells and dofs: ["
            << solid_solver.triangulation.n_active_cells() << ", "
            << solid_solver.dof_handler.n_dofs() << "]" << std::endl;

  bool first_step = true;
  if (parameters.refinement_interval < parameters.end_time)
    {
      refine_mesh(parameters.global_refinements[0],
                  parameters.global_refinements[0] + 1);
      refine_mesh(parameters.global_refinements[0],
                  parameters.global_refinements[0] + 1);
    }
  while (time.end() - time.current() > 1e-12)
    {
      find_solid_bc();
      {
        TimerOutput::Scope timer_section(timer, "Run solid solver");
        solid_solver.run_one_step(first_step);
      }
      update_solid_box();
      update_indicator();
      fluid_solver.make_constraints();
      if (!first_step)
        {
          fluid_solver.nonzero_constraints.clear();
          fluid_solver.nonzero_constraints.copy_from(
            fluid_solver.zero_constraints);
        }
      find_fluid_bc();
      {
        TimerOutput::Scope timer_section(timer, "Run fluid solver");
        fluid_solver.run_one_step(true);
      }
      first_step = false;
      time.increment();
      if (time.time_to_refine())
        {
          refine_mesh(parameters.global_refinements[0],
                      parameters.global_refinements[0] + 1);
        }
    }
}

template class FSI<2>;
template class FSI<3>;

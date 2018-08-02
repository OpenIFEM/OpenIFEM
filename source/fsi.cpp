#include "fsi.h"
#include <complex>
#include <iostream>

template <int dim>
FSI<dim>::FSI(Fluid::FluidSolver<dim> &f,
              Solid::SolidSolver<dim> &s,
              const Parameters::AllParameters &p)
  : fluid_solver(f),
    solid_solver(s),
    parameters(p),
    time(parameters.end_time,
         parameters.time_step,
         parameters.output_interval,
         parameters.refinement_interval,
         parameters.save_interval)
{
}

template <int dim>
void FSI<dim>::move_solid_mesh(bool move_forward)
{
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
bool FSI<dim>::point_in_mesh(const DoFHandler<dim> &df, const Point<dim> &point)
{
  for (auto cell = df.begin_active(); cell != df.end(); ++cell)
    {
      if (cell->point_inside(point))
        {
          return true;
        }
    }
  return false;
}

// Dirichlet bcs are applied to artificial fluid cells, so fluid nodes should
// be marked as artificial or real. Meanwhile, additional body force is
// applied to the artificial fluid quadrature points. To accomodate these two
// settings, we define indicator at quadrature points, but only when all
// of the vertices of a fluid cell are found to be in solid domain,
// set the indicators at all quadrature points to be 1.
template <int dim>
void FSI<dim>::update_indicator()
{
  move_solid_mesh(true);
  FEValues<dim> fe_values(fluid_solver.fe,
                          fluid_solver.volume_quad_formula,
                          update_quadrature_points);
  const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      fe_values.reinit(f_cell);
      auto p = fluid_solver.cell_property.get_data(f_cell);
      bool is_solid = true;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          Point<dim> point = f_cell->vertex(v);
          if (!point_in_mesh(solid_solver.dof_handler, point))
            {
              is_solid = false;
              break;
            }
        }
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          p[q]->indicator = is_solid;
        }
    }
  move_solid_mesh(false);
}

// This function interpolates the solid velocity into the fluid solver,
// as the Dirichlet boundary conditions for artificial fluid vertices
template <int dim>
void FSI<dim>::find_fluid_bc()
{
  move_solid_mesh(true);

  // The nonzero Dirichlet BCs (to set the velocity) and zero Dirichlet
  // BCs (to set the velocity increment) for the artificial fluid domain.
  AffineConstraints<double> inner_nonzero, inner_zero;
  inner_nonzero.clear();
  inner_zero.clear();

  const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
  FEValues<dim> fe_values(fluid_solver.fe,
                          fluid_solver.volume_quad_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<SymmetricTensor<2, dim>> sym_grad_v(n_q_points);
  std::vector<double> p(n_q_points);
  std::vector<Tensor<2, dim>> grad_v(n_q_points);
  std::vector<Tensor<1, dim>> v(n_q_points);
  std::vector<Tensor<1, dim>> dv(n_q_points);

  const std::vector<Point<dim>> &unit_points =
    fluid_solver.fe.get_unit_support_points();
  Quadrature<dim> dummy_q(unit_points.size());
  MappingQGeneric<dim> mapping(1);
  FEValues<dim> dummy_fe_values(
    mapping, fluid_solver.fe, dummy_q, update_quadrature_points);
  std::vector<types::global_dof_index> dof_indices(
    fluid_solver.fe.dofs_per_cell);

  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto ptr = fluid_solver.cell_property.get_data(f_cell);
      if (ptr[0]->indicator != 1)
        {
          continue;
        }
      fe_values.reinit(f_cell);
      dummy_fe_values.reinit(f_cell);
      f_cell->get_dof_indices(dof_indices);
      auto support_points = dummy_fe_values.get_quadrature_points();
      // Fluid velocity increment
      fe_values[velocities].get_function_values(fluid_solver.solution_increment,
                                                dv);
      // Fluid velocity gradient
      fe_values[velocities].get_function_gradients(
        fluid_solver.present_solution, grad_v);
      // Fluid symmetric velocity gradient
      fe_values[velocities].get_function_symmetric_gradients(
        fluid_solver.present_solution, sym_grad_v);
      // Fluid pressure
      fe_values[pressure].get_function_values(fluid_solver.present_solution, p);
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
          // if (!point_in_mesh(solid_solver.dof_handler, support_points[i]))
          // continue;
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
            line, fluid_velocity[index] - fluid_solver.present_solution(line));
        }
      // Loop over all quadrature points to set FSI forces.
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Point<dim> point = fe_values.quadrature_point(q);
          if (!ptr[q]->indicator)
            continue;
          // acceleration: Dv^f/Dt - Dv^s/Dt
          Tensor<1, dim> fluid_acc =
            dv[q] / time.get_delta_t() + grad_v[q] * v[q];
          Vector<double> solid_acc(dim);
          VectorTools::point_value(solid_solver.dof_handler,
                                   solid_solver.current_acceleration,
                                   point,
                                   solid_acc);
          for (unsigned int i = 0; i < dim; ++i)
            {
              ptr[q]->fsi_acceleration[i] =
                -parameters.gravity[i]; // fluid_acc[i] - solid_acc[i];
            }
          // stress: sigma^f - sigma^s
          SymmetricTensor<2, dim> solid_sigma;
          for (unsigned int i = 0; i < dim; ++i)
            {
              for (unsigned int j = 0; j < dim; ++j)
                {
                  Vector<double> sigma_ij(1);
                  VectorTools::point_value(solid_solver.dg_dof_handler,
                                           solid_solver.stress[i][j],
                                           point,
                                           sigma_ij);
                  solid_sigma[i][j] = sigma_ij[0];
                }
            }
          ptr[q]->fsi_stress = 0;
          /*
          -p[q] * Physics::Elasticity::StandardTensors<dim>::I +
          parameters.viscosity * sym_grad_v[q];// - solid_sigma;
          */
        }
    }
  inner_nonzero.close();
  inner_zero.close();
  fluid_solver.nonzero_constraints.merge(
    inner_nonzero,
    AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
  fluid_solver.zero_constraints.merge(
    inner_zero,
    AffineConstraints<double>::MergeConflictBehavior::left_object_wins);

  move_solid_mesh(false);
}

template <int dim>
void FSI<dim>::find_solid_bc()
{
  // Must use the updated solid coordinates
  move_solid_mesh(true);
  // Fluid FEValues to do interpolation
  FEValues<dim> fe_values(
    fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);
  // Solid FEFaceValues to get the normal
  FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                   solid_solver.face_quad_formula,
                                   update_quadrature_points |
                                     update_normal_vectors);

  const unsigned int n_face_q_points = solid_solver.face_quad_formula.size();

  for (auto s_cell = solid_solver.dof_handler.begin_active();
       s_cell != solid_solver.dof_handler.end();
       ++s_cell)
    {
      auto ptr = solid_solver.cell_property.get_data(s_cell);
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          // Current face is at boundary and without Dirichlet bc.
          if (s_cell->face(f)->at_boundary() &&
              parameters.solid_dirichlet_bcs.find(
                s_cell->face(f)->boundary_id()) ==
                parameters.solid_dirichlet_bcs.end())
            {
              fe_face_values.reinit(s_cell, f);
              for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                  Point<dim> q_point = fe_face_values.quadrature_point(q);
                  Tensor<1, dim> normal = fe_face_values.normal_vector(q);
                  Vector<double> value(dim + 1);
                  Utils::GridInterpolator<dim, BlockVector<double>>
                    interpolator(fluid_solver.dof_handler, q_point);
                  interpolator.point_value(fluid_solver.present_solution,
                                           value);
                  std::vector<Tensor<1, dim>> gradient(dim + 1,
                                                       Tensor<1, dim>());
                  interpolator.point_gradient(fluid_solver.present_solution,
                                              gradient);
                  SymmetricTensor<2, dim> sym_deformation;
                  for (unsigned int i = 0; i < dim; ++i)
                    {
                      for (unsigned int j = 0; j < dim; ++j)
                        {
                          sym_deformation[i][j] =
                            (gradient[i][j] + gradient[j][i]) / 2;
                        }
                    }
                  // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                  SymmetricTensor<2, dim> stress =
                    -value[dim] * Physics::Elasticity::StandardTensors<dim>::I +
                    parameters.viscosity * sym_deformation;
                  ptr[f * n_face_q_points + q]->fsi_traction = stress * normal;
                }
            }
        }
    }
  move_solid_mesh(false);
}

template <int dim>
void FSI<dim>::refine_mesh(const unsigned int min_grid_level,
                           const unsigned int max_grid_level)
{
  move_solid_mesh(true);
  for (auto f_cell : fluid_solver.dof_handler.active_cell_iterators())
    {
      auto center = f_cell->center();
      double dist = 1000;
      for (auto s_cell : solid_solver.dof_handler.active_cell_iterators())
        {
          dist = std::min(center.distance(s_cell->center()), dist);
        }
      if (dist < 0.1)
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
  solid_solver.triangulation.refine_global(parameters.global_refinement + 1);
  solid_solver.setup_dofs();
  solid_solver.initialize_system();
  fluid_solver.triangulation.refine_global(parameters.global_refinement);
  fluid_solver.setup_dofs();
  fluid_solver.make_constraints();
  fluid_solver.initialize_system();

  std::cout << "  Number of fluid active cells: "
            << fluid_solver.triangulation.n_active_cells() << std::endl
            << "  Number of solid active cells: "
            << solid_solver.triangulation.n_active_cells() << std::endl;

  bool first_step = true;
  refine_mesh(parameters.global_refinement, parameters.global_refinement + 2);
  while (time.end() - time.current() > 1e-12)
    {
      find_solid_bc();
      solid_solver.run_one_step(first_step);
      update_indicator();
      fluid_solver.make_constraints();
      if (!first_step)
        {
          fluid_solver.nonzero_constraints.clear();
          fluid_solver.nonzero_constraints.copy_from(
            fluid_solver.zero_constraints);
        }
      find_fluid_bc();
      fluid_solver.run_one_step(true);
      first_step = false;
      time.increment();
      if (time.time_to_refine())
        refine_mesh(parameters.global_refinement,
                    parameters.global_refinement + 2);
    }
}

template class FSI<2>;
template class FSI<3>;

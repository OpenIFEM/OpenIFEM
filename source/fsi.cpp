#include "fsi.h"
#include <iostream>

template <int dim>
FSI<dim>::FSI(Fluid::NavierStokes<dim> &f,
              Solid::LinearElasticSolver<dim> &s,
              const Parameters::AllParameters &p)
  : fluid_solver(f),
    solid_solver(s),
    parameters(p),
    time(parameters.end_time,
         parameters.time_step,
         parameters.output_interval,
         parameters.refinement_interval)
{
  std::cout << "  Number of fluid active cells: "
            << fluid_solver.triangulation.n_active_cells() << std::endl
            << "  Number of solid active cells: "
            << solid_solver.triangulation.n_active_cells() << std::endl;
}

template <int dim>
void FSI<dim>::initialize_system()
{
  fluid_solver.setup_dofs();
  fluid_solver.make_constraints();
  fluid_solver.initialize_system();
  solid_solver.setup_dofs();
  solid_solver.initialize_system();
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

template <int dim>
void FSI<dim>::update_indicator()
{
  move_solid_mesh(true);

  // NOTE: The indicator is defined at quadrature points, so its value should be
  // determined by whether the quadrature point is in solid mesh or not.
  // However,
  // this is not stable. Our current strategy is that only when all the
  // quadrature
  // points in a fluid cell are found in solid, then set the indicators to 1.
  const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
  FEValues<dim> fe_values(fluid_solver.fe,
                          fluid_solver.volume_quad_formula,
                          update_quadrature_points);
  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto p = fluid_solver.cell_property.get_data(f_cell);
      fe_values.reinit(f_cell);
      bool is_solid = true;
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Point<dim> q_point = fe_values.quadrature_point(q);
          if (!point_in_mesh(solid_solver.dof_handler, q_point))
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

template <int dim>
void FSI<dim>::find_fluid_fsi()
{
  FEValues<dim> fe_values(fluid_solver.fe,
                          fluid_solver.volume_quad_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  const double dt = time.get_delta_t();
  const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<SymmetricTensor<2, dim>> sym_grad_v(n_q_points);
  std::vector<Tensor<2, dim>> grad_v(n_q_points);
  std::vector<Tensor<1, dim>> v(n_q_points);
  std::vector<Tensor<1, dim>> dv(n_q_points);
  std::vector<double> p(n_q_points);

  for (auto f_cell = fluid_solver.dof_handler.begin_active();
       f_cell != fluid_solver.dof_handler.end();
       ++f_cell)
    {
      auto ptr = fluid_solver.cell_property.get_data(f_cell);
      fe_values.reinit(f_cell);
      // Fluid symmetric velocity gradient
      fe_values[velocities].get_function_symmetric_gradients(
        fluid_solver.present_solution, sym_grad_v);
      // Fluid pressure
      fe_values[pressure].get_function_values(fluid_solver.present_solution, p);
      // Fluid velocity
      fe_values[velocities].get_function_values(fluid_solver.present_solution,
                                                v);
      // Fluid velocity gradient
      fe_values[velocities].get_function_gradients(
        fluid_solver.present_solution, grad_v);
      // Fluid velocity increment
      fe_values[velocities].get_function_values(fluid_solver.solution_increment,
                                                dv);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          if (ptr[q]->indicator == 0)
            {
              continue; // FSI force is only applied to artificial fluid cells
            }
          double mu = ptr[q]->get_mu();
          // Solid part
          Point<dim> q_point = fe_values.quadrature_point(q);
          Vector<double> tmp(dim);
          Utils::GridInterpolator<dim, Vector<double>> interpolator(
            solid_solver.dof_handler, q_point);
          interpolator.point_value(solid_solver.current_acceleration, tmp);
          Tensor<1, dim> solid_acc;
          for (unsigned int i = 0; i < dim; ++i)
            {
              solid_acc[i] = tmp[i];
            }
          Utils::GridInterpolator<dim, Vector<double>> dg_interpolator(
            solid_solver.dg_dof_handler, q_point);
          SymmetricTensor<2, dim> solid_sigma;
          for (unsigned int i = 0; i < dim; ++i)
            {
              for (unsigned int j = 0; j < dim; ++j)
                {
                  Vector<double> sigma_ij(1);
                  dg_interpolator.point_value(solid_solver.stress[i][j],
                                              sigma_ij);
                  solid_sigma[i][j] = sigma_ij[0];
                }
            }
          // Fluid part
          SymmetricTensor<2, dim> fluid_sigma =
            -p[q] * Physics::Elasticity::StandardTensors<dim>::I +
            mu * sym_grad_v[q];
          Tensor<1, dim> fluid_acc = dv[q] / dt + grad_v[q] * v[q];
          // FSI force
          ptr[q]->fsi_stress = fluid_sigma - solid_sigma;
          ptr[q]->fsi_acceleration = fluid_acc - solid_acc;
        }
    }
}

template <int dim>
void FSI<dim>::find_solid_bc()
{
  // Fluid finite element extractor
  const FEValuesExtractors::Vector v(0);
  const FEValuesExtractors::Scalar p(dim);

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
      auto p = solid_solver.cell_property.get_data(s_cell);
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
                  // Note that we are using undeformed quadrature points and
                  // normal vectors.
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
                  p[f * n_face_q_points + q]->fsi_traction = stress * normal;
                }
            }
        }
    }
}

template <int dim>
void FSI<dim>::run()
{
  std::ofstream ofs("FSI_force.txt", std::ofstream::out);

  fluid_solver.triangulation.refine_global(parameters.global_refinement);
  solid_solver.triangulation.refine_global(parameters.global_refinement);
  initialize_system();
  bool first_step = true;
  while (time.end() - time.current() > 1e-12)
    {
      find_solid_bc();
      solid_solver.run_one_step(first_step);
      update_indicator();
      find_fluid_fsi();
      fluid_solver.run_one_step(first_step);
      first_step = false;
      time.increment();
    }
}

template class FSI<2>;
template class FSI<3>;

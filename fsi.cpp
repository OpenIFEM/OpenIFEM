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
void FSI<dim>::update_indicator()
{
  move_solid_mesh(true);

  for (auto f_cell = fluid_solver.triangulation.begin_active();
       f_cell != fluid_solver.triangulation.end();
       ++f_cell)
    {
      Point<dim> center = f_cell->center();
      bool is_solid = false;
      for (auto s_cell = solid_solver.triangulation.begin_active();
           s_cell != solid_solver.triangulation.end();
           ++s_cell)
        {
          if (s_cell->point_inside(center))
            {
              is_solid = true;
            }
        }
      auto p = fluid_solver.cell_property.get_data(f_cell);
      p[0]->indicator = (is_solid ? 1.0 : 0.0);
    }

  move_solid_mesh(false);
}

template <int dim>
void FSI<dim>::find_solid_bc(std::vector<double> &pressure)
{
  pressure.clear();

  // Fluid finite element extractor
  const FEValuesExtractors::Vector v(0);
  const FEValuesExtractors::Scalar p(dim);

  // Fluid FEValue to do interpolation
  FEValues<dim> fe_values(
    fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);

  for (auto s_cell = solid_solver.dof_handler.begin_active();
       s_cell != solid_solver.dof_handler.end();
       ++s_cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          // Current face is at boundary and without Dirichlet bc.
          if (s_cell->face(f)->at_boundary() &&
              parameters.solid_dirichlet_bcs.find(
                s_cell->face(f)->boundary_id()) ==
                parameters.solid_dirichlet_bcs.end())
            {
              Point<dim> center = s_cell->face(f)->center();
              Vector<double> value(dim + 1);
              // This function finds the fe value at an arbitrary point,
              // if the point is not in the field, an exception will be thrown.
              VectorTools::point_value(fluid_solver.dof_handler,
                                       fluid_solver.present_solution,
                                       center,
                                       value);
              // For solid, the pressure bc is defined as negative
              // FIXME: I forgot the pressure we get the fluid solver has been
              // normalized..
              pressure.push_back(-value[dim]);
            }
        }
    }
}

template <int dim>
void FSI<dim>::run()
{
  fluid_solver.triangulation.refine_global(parameters.global_refinement);
  initialize_system();
  bool first_step = true;
  while (time.end() - time.current() > 1e-12)
    {
      std::vector<double> pressure;
      find_solid_bc(pressure);
      solid_solver.fluid_pressure = pressure;
      solid_solver.run_one_step(first_step);
      update_indicator();
      fluid_solver.run_one_step(first_step);

      first_step = false;
      time.increment();
    }
}

template class FSI<2>;
template class FSI<3>;

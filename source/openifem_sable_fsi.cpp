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

template <int dim>
std::pair<bool, const typename DoFHandler<dim>::active_cell_iterator>
OpenIFEM_Sable_FSI<dim>::point_in_solid_new(const DoFHandler<dim> &df,
                                            const Point<dim> &point)
{
  // Check whether the point is in the solid box first.
  for (unsigned int i = 0; i < dim; ++i)
    {
      if (point(i) < solid_box(2 * i) || point(i) > solid_box(2 * i + 1))

        return {false, {}};
    }
  for (auto cell = df.begin_active(); cell != df.end(); ++cell)
    {
      if (cell->point_inside(point))
        {
          return {true, cell};
        }
    }
  return {false, {}};
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

      if (inside_count == 0)
        {
          p[0]->indicator = 0;
          p[0]->exact_indicator = 0;
          continue;
        }

      if (inside_count == GeometryInfo<dim>::vertices_per_cell)
        {
          p[0]->indicator = 1;
          p[0]->exact_indicator = 1;
          continue;
        }

      // update exact indicator field
      // initialize it to zero
      p[0]->exact_indicator = 0;

      /**** Old way *****/
      // get upper and lower corner for the Eulerian cell
      /*Point<dim> l_eu = f_cell->vertex(0);
      Point<dim> u_eu;
      if (dim == 2)
        u_eu = f_cell->vertex(3);
      else
        u_eu = f_cell->vertex(7);
      // get eulerian cell size
      double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));
      double total_cell_area = pow(h, dim);
      int tot_sample = 0;
      for (auto s_cell = solid_solver.dof_handler.begin_active();
           s_cell != solid_solver.dof_handler.end();
           ++s_cell)
        {
          // create bounding box for the Lagrangian element
          Point<dim> l_lag = s_cell->vertex(0);
          Point<dim> u_lag = s_cell->vertex(0);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              for (unsigned int i = 0; i < dim; i++)
                {
                  if (s_cell->vertex(v)(i) < l_lag(i))
                    l_lag(i) = s_cell->vertex(v)(i);
                  else if (s_cell->vertex(v)(i) > u_lag(i))
                    u_lag(i) = s_cell->vertex(v)(i);
                }
            }

          bool intersection = true;
          for (unsigned int i = 0; i < dim; i++)
            {
              if ((l_lag(i) >= u_eu(i)) || (l_eu(i) >= u_lag(i)))
                {
                  intersection = false;
                  break;
                }
            }
          if (!intersection)
            continue;

          Point<dim> l_int;
          Point<dim> u_int;
          for (unsigned int i = 0; i < dim; i++)
            {
              l_int(i) = std::max(l_eu(i), l_lag(i));
              u_int(i) = std::min(u_eu(i), u_lag(i));
            }

          double intersection_area = 1.0;
          for (unsigned i = 0; i < dim; i++)
            {
              intersection_area *= abs(u_int(i) - l_int(i));
            }
          // if intersection area equals Eulerian cell area assign indicator as
          // 1
          if (abs(intersection_area - total_cell_area) < 1e-10)
            {
              p[0]->exact_indicator = 1;
              continue;
            }
          // sample points in intersection area
          int sample_count = 0;
          int sample_inside = 0;
          // set distance to sample equally spaced points from starting point
          double dh = 0.1 * h;
          // determine no. of samples in each direction
          std::vector<int> n_sample_points(dim, 0);
          for (unsigned int i = 0; i < dim; i++)
            {
              double n_samples = abs(u_int(i) - l_int(i)) / (dh);
              n_sample_points[i] = int(std::round(n_samples)) + 1;
            }
          tot_sample += n_sample_points[0] + n_sample_points[1];
          // create starting point for sampling at lower corner of intersection
          // box
          Point<dim> start_point_l(l_int);
          for (int i = 0; i < n_sample_points[0]; i++)
            {
              Point<dim> sample_point = start_point_l;
              sample_point(0) = start_point_l(0) + i * dh;
              for (int j = 0; j < n_sample_points[1]; j++)
                {
                  sample_point(1) = start_point_l(1) + j * dh;
                  if (dim == 2)
                    {
                      sample_count += 1;
                      if (s_cell->point_inside(sample_point))
                        sample_inside += 1;
                    }
                  else
                    {
                      for (int k = 0; k < n_sample_points[2]; k++)
                        {
                          sample_point(2) = start_point_l(2) + k * dh;
                          sample_count += 1;
                          if (s_cell->point_inside(sample_point))
                            sample_inside += 1;
                        }
                    }
                }
            }

          // create starting point for sampling at upper corner of intersection
          // box
          Point<dim> start_point_u(u_int);
          for (int i = 0; i < n_sample_points[0]; i++)
            {
              Point<dim> sample_point = start_point_u;
              sample_point(0) = start_point_u(0) - i * dh;
              for (int j = 0; j < n_sample_points[1]; j++)
                {
                  sample_point(1) = start_point_u(1) - j * dh;
                  if (dim == 2)
                    {
                      sample_count += 1;
                      if (s_cell->point_inside(sample_point))
                        sample_inside += 1;
                    }
                  else
                    {
                      for (int k = 0; k < n_sample_points[2]; k++)
                        {
                          sample_point(2) = start_point_l(2) - k * dh;
                          sample_count += 1;
                          if (s_cell->point_inside(sample_point))
                            sample_inside += 1;
                        }
                    }
                }
            }

          p[0]->exact_indicator +=
            (intersection_area / total_cell_area) *
            (double(sample_inside) / double(sample_count));
          }*/
      /**** Old way *****/
      /**** New way *****/
      // get upper and lower corner for the Eulerian cell
      Point<dim> l_eu = f_cell->vertex(0);
      Point<dim> u_eu;
      if (dim == 2)
        u_eu = f_cell->vertex(3);
      else
        u_eu = f_cell->vertex(7);
      // get eulerian cell size
      double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));
      double total_cell_area = pow(h, dim);
      // check if cell intersects with solid box
      bool intersection = true;
      for (unsigned int i = 0; i < dim; i++)
        {
          if ((solid_box(2 * i) >= u_eu(i)) ||
              (l_eu(i) >= solid_box(2 * i + 1)))
            {
              intersection = false;
              break;
            }
        }
      if (!intersection)
        continue;
      // sample points
      int n = 10;
      int sample_count = pow((n + 1), dim);
      double dh = h / double(n);
      std::vector<Point<dim>> sample_points;

      for (int i = 0; i < n + 1; i++)
        {
          for (int j = 0; j < n + 1; j++)
            {
              Point<dim> sample;
              sample[0] = l_eu[0] + dh * i;
              sample[1] = l_eu[1] + dh * j;

              if (dim == 3)
                {
                  for (int k = 0; k < n + 1; k++)
                    {
                      sample[2] = l_eu[2] + dh * h;
                    }
                }

              bool inside_box = true;
              for (unsigned int d = 0; d < dim; d++)
                {
                  if ((sample[d] < solid_box[2 * d]) ||
                      (sample[d] > solid_box[2 * d + 1]))
                    {
                      inside_box = false;
                      break;
                    }
                }
              if (!inside_box)
                continue;
              sample_points.push_back(sample);
            }
        }

      for (auto s_cell = solid_solver.dof_handler.begin_active();
           s_cell != solid_solver.dof_handler.end();
           ++s_cell)
        {

          // create bounding box for the Lagrangian element
          Point<dim> l_lag = s_cell->vertex(0);
          Point<dim> u_lag = s_cell->vertex(0);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              for (unsigned int i = 0; i < dim; i++)
                {
                  if (s_cell->vertex(v)(i) < l_lag(i))
                    l_lag(i) = s_cell->vertex(v)(i);
                  else if (s_cell->vertex(v)(i) > u_lag(i))
                    u_lag(i) = s_cell->vertex(v)(i);
                }
            }

          bool intersection = true;
          for (unsigned int i = 0; i < dim; i++)
            {
              if ((l_lag(i) >= u_eu(i)) || (l_eu(i) >= u_lag(i)))
                {
                  intersection = false;
                  break;
                }
            }
          if (!intersection)
            continue;

          Point<dim> l_int;
          Point<dim> u_int;
          for (unsigned int i = 0; i < dim; i++)
            {
              l_int(i) = std::max(l_eu(i), l_lag(i));
              u_int(i) = std::min(u_eu(i), u_lag(i));
            }

          double intersection_area = 1.0;
          for (unsigned i = 0; i < dim; i++)
            {
              intersection_area *= abs(u_int(i) - l_int(i));
            }
          // if intersection area equals Eulerian cell area assign indicator as
          // 1
          if (abs(intersection_area - total_cell_area) < 1e-10)
            {
              p[0]->exact_indicator = 1;
              continue;
            }

          int sample_inside = 0;
          for (unsigned int s = 0; s < sample_points.size(); s++)
            {
              auto sample = sample_points[s];
              bool inside_box = true;
              for (unsigned int d = 0; d < dim; d++)
                {
                  if ((sample[d] < l_int[d]) || (sample[d] > u_int[d]))
                    {
                      inside_box = false;
                      break;
                    }
                }
              if (!inside_box)
                continue;
              if (s_cell->point_inside(sample))
                sample_inside += 1;
            }

          p[0]->exact_indicator +=
            (double(sample_inside) / double(sample_count));
        }
      /**** New way *****/
      // if the exact indicator is greater than one then round it off to 1
      if (p[0]->exact_indicator > 1.0)
        p[0]->exact_indicator = 1.0;
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

      // check which cell nodes are inside cells to calculate velocity bc
      std::vector<int> inside_nodes;
      std::vector<int> outside_nodes;
      int inside_count = 0;
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
            {
              inside_nodes.push_back(v);
              ++inside_count;
            }
          else
            outside_nodes.push_back(v);
        }
      // store local node ids which are inside and outside the solid
      cell_nodes_inside_solid.insert({cell_count, inside_nodes});
      cell_nodes_outside_solid.insert({cell_count, outside_nodes});

      auto q_points = fe_values.get_quadrature_points();
      auto p = sable_solver.cell_property.get_data(f_cell);
      unsigned int inside_qpoint = 0;

      if (inside_count == 0)
        {
          p[0]->indicator = 0;
          p[0]->exact_indicator = 0;
          continue;
        }

      if (inside_count == GeometryInfo<dim>::vertices_per_cell)
        {
          p[0]->indicator = 1;
          p[0]->exact_indicator = 1;
          continue;
        }

      for (unsigned int q = 0; q < q_points.size(); q++)
        {
          if (point_in_solid(solid_solver.dof_handler, q_points[q]))
            {
              ++inside_qpoint;
            }
        }
      if (parameters.indicator_field_condition == "CompletelyInsideSolid")
        {
          p[0]->indicator = (inside_qpoint == q_points.size() ? 1 : 0);
        }
      else
        {
          // parameters.indicator_field_condition == "PartiallyInsideSolid"
          p[0]->indicator = double(inside_qpoint) / double(q_points.size());
        }

      cell_count += 1;

      // update exact indicator field
      // initialize it to zero
      p[0]->exact_indicator = 0;

      /**** Old way *****/
      // get upper and lower corner for the Eulerian cell
      /*Point<dim> l_eu = f_cell->vertex(0);
      Point<dim> u_eu;
      if (dim == 2)
        u_eu = f_cell->vertex(3);
      else
        u_eu = f_cell->vertex(7);
      // get eulerian cell size
      double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));
      double total_cell_area = pow(h, dim);

      for (auto s_cell = solid_solver.dof_handler.begin_active();
           s_cell != solid_solver.dof_handler.end();
           ++s_cell)
        {
          // create bounding box for the Lagrangian element
          Point<dim> l_lag = s_cell->vertex(0);
          Point<dim> u_lag = s_cell->vertex(0);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              for (unsigned int i = 0; i < dim; i++)
                {
                  if (s_cell->vertex(v)(i) < l_lag(i))
                    l_lag(i) = s_cell->vertex(v)(i);
                  else if (s_cell->vertex(v)(i) > u_lag(i))
                    u_lag(i) = s_cell->vertex(v)(i);
                }
            }

          bool intersection = true;
          for (unsigned int i = 0; i < dim; i++)
            {
              if ((l_lag(i) >= u_eu(i)) || (l_eu(i) >= u_lag(i)))
                {
                  intersection = false;
                  break;
                }
            }
          if (!intersection)
            continue;

          Point<dim> l_int;
          Point<dim> u_int;
          for (unsigned int i = 0; i < dim; i++)
            {
              l_int(i) = std::max(l_eu(i), l_lag(i));
              u_int(i) = std::min(u_eu(i), u_lag(i));
            }

          double intersection_area = 1.0;
          for (unsigned i = 0; i < dim; i++)
            {
              intersection_area *= abs(u_int(i) - l_int(i));
            }
          // if intersection area equals Eulerian cell area assign indicator as
          // 1
          if (abs(intersection_area - total_cell_area) < 1e-10)
            {
              p[0]->exact_indicator = 1;
              continue;
            }
          // sample points in intersection area
          int sample_count = 0;
          int sample_inside = 0;
          // set distance to sample equally spaced points from starting point
          double dh = 0.01 * h;
          // determine no. of samples in each direction
          std::vector<int> n_sample_points(dim, 0);
          for (unsigned int i = 0; i < dim; i++)
            {
              double n_samples = abs(u_int(i) - l_int(i)) / (dh);
              n_sample_points[i] = int(std::round(n_samples)) + 1;
            }

          // create starting point for sampling at lower corner of intersection
          // box
          Point<dim> start_point_l(l_int);
          for (int i = 0; i < n_sample_points[0]; i++)
            {
              Point<dim> sample_point = start_point_l;
              sample_point(0) = start_point_l(0) + i * dh;
              for (int j = 0; j < n_sample_points[1]; j++)
                {
                  sample_point(1) = start_point_l(1) + j * dh;
                  if (dim == 2)
                    {
                      sample_count += 1;
                      if (s_cell->point_inside(sample_point))
                        sample_inside += 1;
                    }
                  else
                    {
                      for (int k = 0; k < n_sample_points[2]; k++)
                        {
                          sample_point(2) = start_point_l(2) + k * dh;
                          sample_count += 1;
                          if (s_cell->point_inside(sample_point))
                            sample_inside += 1;
                        }
                    }
                }
            }

          // create starting point for sampling at upper corner of intersection
          // box
          Point<dim> start_point_u(u_int);
          for (int i = 0; i < n_sample_points[0]; i++)
            {
              Point<dim> sample_point = start_point_u;
              sample_point(0) = start_point_u(0) - i * dh;
              for (int j = 0; j < n_sample_points[1]; j++)
                {
                  sample_point(1) = start_point_u(1) - j * dh;
                  if (dim == 2)
                    {
                      sample_count += 1;
                      if (s_cell->point_inside(sample_point))
                        sample_inside += 1;
                    }
                  else
                    {
                      for (int k = 0; k < n_sample_points[2]; k++)
                        {
                          sample_point(2) = start_point_l(2) - k * dh;
                          sample_count += 1;
                          if (s_cell->point_inside(sample_point))
                            sample_inside += 1;
                        }
                    }
                }
            }

          p[0]->exact_indicator +=
            (intersection_area / total_cell_area) *
            (double(sample_inside) / double(sample_count));
          }  */
      /**** Old way *****/
      /**** New way *****/
      // get upper and lower corner for the Eulerian cell
      Point<dim> l_eu = f_cell->vertex(0);
      Point<dim> u_eu;
      if (dim == 2)
        u_eu = f_cell->vertex(3);
      else
        u_eu = f_cell->vertex(7);
      // get eulerian cell size
      double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));
      double total_cell_area = pow(h, dim);
      // check if cell intersects with solid box
      bool intersection = true;
      for (unsigned int i = 0; i < dim; i++)
        {
          if ((solid_box(2 * i) >= u_eu(i)) ||
              (l_eu(i) >= solid_box(2 * i + 1)))
            {
              intersection = false;
              break;
            }
        }
      if (!intersection)
        continue;
      // sample points
      int n = 10;
      int sample_count = pow((n + 1), dim);
      double dh = h / double(n);
      std::vector<Point<dim>> sample_points;

      for (int i = 0; i < n + 1; i++)
        {
          for (int j = 0; j < n + 1; j++)
            {
              Point<dim> sample;
              sample[0] = l_eu[0] + dh * i;
              sample[1] = l_eu[1] + dh * j;

              if (dim == 3)
                {
                  for (int k = 0; k < n + 1; k++)
                    {
                      sample[2] = l_eu[2] + dh * h;
                    }
                }

              bool inside_box = true;
              for (unsigned int d = 0; d < dim; d++)
                {
                  if ((sample[d] < solid_box[2 * d]) ||
                      (sample[d] > solid_box[2 * d + 1]))
                    {
                      inside_box = false;
                      break;
                    }
                }
              if (!inside_box)
                continue;
              sample_points.push_back(sample);
            }
        }

      for (auto s_cell = solid_solver.dof_handler.begin_active();
           s_cell != solid_solver.dof_handler.end();
           ++s_cell)
        {

          // create bounding box for the Lagrangian element
          Point<dim> l_lag = s_cell->vertex(0);
          Point<dim> u_lag = s_cell->vertex(0);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              for (unsigned int i = 0; i < dim; i++)
                {
                  if (s_cell->vertex(v)(i) < l_lag(i))
                    l_lag(i) = s_cell->vertex(v)(i);
                  else if (s_cell->vertex(v)(i) > u_lag(i))
                    u_lag(i) = s_cell->vertex(v)(i);
                }
            }

          bool intersection = true;
          for (unsigned int i = 0; i < dim; i++)
            {
              if ((l_lag(i) >= u_eu(i)) || (l_eu(i) >= u_lag(i)))
                {
                  intersection = false;
                  break;
                }
            }
          if (!intersection)
            continue;

          Point<dim> l_int;
          Point<dim> u_int;
          for (unsigned int i = 0; i < dim; i++)
            {
              l_int(i) = std::max(l_eu(i), l_lag(i));
              u_int(i) = std::min(u_eu(i), u_lag(i));
            }

          double intersection_area = 1.0;
          for (unsigned i = 0; i < dim; i++)
            {
              intersection_area *= abs(u_int(i) - l_int(i));
            }
          // if intersection area equals Eulerian cell area assign indicator as
          // 1
          if (abs(intersection_area - total_cell_area) < 1e-10)
            {
              p[0]->exact_indicator = 1;
              continue;
            }

          int sample_inside = 0;
          for (unsigned int s = 0; s < sample_points.size(); s++)
            {
              auto sample = sample_points[s];
              bool inside_box = true;
              for (unsigned int d = 0; d < dim; d++)
                {
                  if ((sample[d] < l_int[d]) || (sample[d] > u_int[d]))
                    {
                      inside_box = false;
                      break;
                    }
                }
              if (!inside_box)
                continue;
              if (s_cell->point_inside(sample))
                sample_inside += 1;
            }

          p[0]->exact_indicator +=
            (double(sample_inside) / double(sample_count));
        }
      /**** New way *****/
      // if the exact indicator is greater than one then round it off to 1
      if (p[0]->exact_indicator > 1.0)
        p[0]->exact_indicator = 1.0;
    }

  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::setup_cell_hints()
{
  unsigned int n_unit_points = sable_solver.fe.get_unit_support_points().size();
  for (auto cell = sable_solver.triangulation.begin_active();
       cell != sable_solver.triangulation.end();
       ++cell)
    {
      if (!cell->is_artificial())
        {
          cell_hints.initialize(cell, n_unit_points);
          const std::vector<
            std::shared_ptr<typename DoFHandler<dim>::active_cell_iterator>>
            hints = cell_hints.get_data(cell);
          Assert(hints.size() == n_unit_points,
                 ExcMessage("Wrong number of cell hints!"));
          for (unsigned int v = 0; v < n_unit_points; ++v)
            {
              // Initialize the hints with the begin iterators!
              *(hints[v]) = solid_solver.dof_handler.begin_active();
            }
        }
    }
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
  // inner_nonzero.reinit(sable_solver.locally_relevant_dofs);
  // inner_zero.reinit(sable_solver.locally_relevant_dofs);
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

          auto hints = cell_hints.get_data(f_cell);
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
              // Same as sable_solver.fe.system_to_base_index(i).first.second;
              const unsigned int index =
                sable_solver.fe.system_to_component_index(i).first;
              Assert(index < dim,
                     ExcMessage("Vector component should be less than dim!"));
              dof_touched[dof_indices[i]] = 1;
              if (!point_in_solid(solid_solver.dof_handler, support_points[i]))
                continue;
              Utils::CellLocator<dim, DoFHandler<dim>> locator(
                solid_solver.dof_handler, support_points[i], *(hints[i]));
              *(hints[i]) = locator.search();
              Utils::GridInterpolator<dim, Vector<double>> interpolator(
                solid_solver.dof_handler, support_points[i], {}, *(hints[i]));
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
              // apply explicit Eulerian penalty
              fluid_acc += parameters.penalty_scale_factor[1] *
                           ((vs - v[i]) / time.get_delta_t());
              //(dv[i]) / time.get_delta_t() + grad_v[i] * v[i];
              auto line = dof_indices[i];
              // Note that we are setting the value of the constraint to the
              // velocity delta!
              tmp_fsi_acceleration(line) =
                (fluid_acc[index] - solid_acc[index]);
              // add penalty force based on the velocity difference between
              // Lagrangian solid and SABLE, calculated from previous time step
              /*tmp_fsi_acceleration(line) +=
                sable_solver.fsi_vel_diff_eul(line) / time.get_delta_t();*/
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
  sable_solver.fsi_penalty_force = 0;

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
  Vector<double> local_penalty_force(dofs_per_cell);

  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<2, dim>> grad_v(n_q_points);
  std::vector<Tensor<1, dim>> v(n_q_points);
  std::vector<Tensor<1, dim>> dv(n_q_points);
  std::vector<Tensor<1, dim>> fsi_vel_diff(n_q_points);

  auto scalar_f_cell = sable_solver.scalar_dof_handler.begin_active();
  for (auto f_cell = sable_solver.dof_handler.begin_active();
       f_cell != sable_solver.dof_handler.end(),
            scalar_f_cell != sable_solver.scalar_dof_handler.end();
       ++f_cell, ++scalar_f_cell)
    {
      auto ptr = sable_solver.cell_property.get_data(f_cell);
      const double ind = ptr[0]->indicator;
      const double ind_exact = ptr[0]->exact_indicator;
      auto s = sable_solver.cell_wise_stress.get_data(f_cell);

      if (ind == 0)
        continue;
      /*const double rho_bar =
        parameters.solid_rho * ind + s[0]->eulerian_density * (1 - ind);*/
      const double rho_bar = parameters.solid_rho * ind_exact +
                             s[0]->eulerian_density * (1 - ind_exact);

      fe_values.reinit(f_cell);
      scalar_fe_values.reinit(scalar_f_cell);

      local_rhs = 0;
      local_rhs_acceleration_part = 0;
      local_rhs_stress_part = 0;
      local_penalty_force = 0;

      // Fluid velocity at support points
      fe_values[velocities].get_function_values(sable_solver.present_solution,
                                                v);
      // Fluid velocity increment at support points
      fe_values[velocities].get_function_values(sable_solver.solution_increment,
                                                dv);
      // Fluid velocity gradient at support points
      fe_values[velocities].get_function_gradients(
        sable_solver.present_solution, grad_v);
      // Difference between Lagrangian solid and artifical material velocities
      fe_values[velocities].get_function_values(sable_solver.fsi_vel_diff_eul,
                                                fsi_vel_diff);

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
          // apply explicit Eulerian penalty
          fluid_acc_tensor += parameters.penalty_scale_factor[1] *
                              ((vs - v[q]) / time.get_delta_t());

          //(dv[q]) / time.get_delta_t() + grad_v[q] * v[q];
          // calculate FSI acceleration
          Tensor<1, dim> fsi_acc_tensor;
          fsi_acc_tensor = fluid_acc_tensor;
          fsi_acc_tensor -= solid_acc_tensor;
          // add penalty force based on the velocity difference between
          // Lagrangian solid and SABLE, calculated from previous time step
          Tensor<1, dim> fsi_penalty_tensor;
          /*fsi_acc_tensor += fsi_vel_diff[q] / time.get_delta_t();
          fsi_penalty_tensor = fsi_vel_diff[q] / time.get_delta_t();*/

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
              local_penalty_force(i) +=
                (rho_bar * fsi_penalty_tensor * phi_u[i]) * fe_values.JxW(q);
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
          sable_solver.fsi_penalty_force[local_dof_indices[i]] +=
            local_penalty_force(i);
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
          // Same as sable_solver.fe.system_to_base_index(i).first.second;
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
void OpenIFEM_Sable_FSI<dim>::find_fluid_bc_new()
{
  TimerOutput::Scope timer_section(timer, "Find fluid BC new");
  move_solid_mesh(true);

  const std::vector<Point<dim>> &unit_points =
    sable_solver.fe.get_unit_support_points();

  MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
  Quadrature<dim> dummy_q(unit_points);
  FEValues<dim> dummy_fe_values(mapping,
                                sable_solver.fe,
                                dummy_q,
                                update_quadrature_points | update_values |
                                  update_gradients);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<Tensor<2, dim>> grad_vf(unit_points.size());
  std::vector<Tensor<1, dim>> vf(unit_points.size());

  std::vector<int> vertex_touched(sable_solver.triangulation.n_vertices(), 0);

  std::vector<types::global_dof_index> scalar_dof_indices(
    sable_solver.scalar_fe.dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(
    sable_solver.fe.dofs_per_cell);

  auto scalar_f_cell = sable_solver.scalar_dof_handler.begin_active();
  auto f_cell = sable_solver.dof_handler.begin_active();
  for (; f_cell != sable_solver.dof_handler.end(),
         scalar_f_cell != sable_solver.scalar_dof_handler.end();
       ++f_cell, ++scalar_f_cell)
    {

      auto ptr = sable_solver.cell_property.get_data(f_cell);
      if (ptr[0]->indicator == 0)
        continue;

      scalar_f_cell->get_dof_indices(scalar_dof_indices);

      f_cell->get_dof_indices(dof_indices);
      dummy_fe_values.reinit(f_cell);

      // Eulerian velocity at support points
      dummy_fe_values[velocities].get_function_values(
        sable_solver.present_solution, vf);

      // Eulerian velocity gradient at support points
      dummy_fe_values[velocities].get_function_gradients(
        sable_solver.present_solution, grad_vf);

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          auto vertex = f_cell->vertex(v);
          if (parameters.indicator_field_condition == "PartiallyInsideSolid")
            {
              if (!point_in_solid(solid_solver.dof_handler, vertex))
                continue;
            }

          if (vertex_touched[f_cell->vertex_index(v)])
            continue;
          vertex_touched[f_cell->vertex_index(v)] = 1;

          Utils::GridInterpolator<dim, Vector<double>> interpolator(
            solid_solver.dof_handler, vertex);
          if (!interpolator.found_cell())
            {
              std::stringstream message;
              message << "Cannot find point in solid: " << vertex << std::endl;
              AssertThrow(interpolator.found_cell(), ExcMessage(message.str()));
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
            solid_solver.scalar_dof_handler, vertex, {}, scalar_s_cell);

          // Lagrangian solid acceleration at Eulerian nodes
          Vector<double> solid_acc(dim);
          Vector<double> solid_vel(dim);
          interpolator.point_value(solid_solver.current_acceleration,
                                   solid_acc);
          interpolator.point_value(solid_solver.current_velocity, solid_vel);

          Tensor<1, dim> vs;
          for (int j = 0; j < dim; ++j)
            {
              vs[j] = solid_vel[j];
            }

          for (unsigned int i = 0; i < unit_points.size(); i++)
            {
              auto dof_id = sable_solver.fe.system_to_component_index(i).first;
              auto vertex_id =
                sable_solver.fe.system_to_component_index(i).second;

              // select support points corresponding to the selected node and
              // skip pressure dofs
              if ((vertex_id == v) && (dof_id < dim))
                {
                  // Fluid total acceleration at support points
                  Tensor<1, dim> fluid_acc =
                    (vs - vf[i]) / time.get_delta_t() + grad_vf[i] * vf[i];
                  // apply explicit Eulerian penalty
                  fluid_acc += parameters.penalty_scale_factor[1] *
                               ((vs - vf[i]) / time.get_delta_t());
                  //(dv[i]) / time.get_delta_t() + grad_v[i] * v[i];
                  auto line = dof_indices[i];

                  sable_solver.fsi_acceleration(line) =
                    (fluid_acc[dof_id] - solid_acc[dof_id]);
                  sable_solver.fsi_velocity(line) = vs[dof_id];
                }
            }

          auto scalar_dof_id = scalar_f_cell->vertex_dof_index(v, 0);

          SymmetricTensor<2, dim> s_cell_stress;
          int stress_index = 0;
          for (unsigned int k = 0; k < dim; k++)
            {
              for (unsigned int m = k; m < dim; m++)
                {
                  // interpolate Lagrangian solid stress at Eulerian nodes
                  Vector<double> s_stress_component(1);
                  scalar_interpolator.point_value(solid_solver.stress[k][m],
                                                  s_stress_component);

                  // When node-based SABLE stress is used
                  if (parameters.fsi_force_calculation_option == "NodeBased")
                    {

                      sable_solver.fsi_stress[stress_index][scalar_dof_id] =
                        sable_solver.stress[k][m][scalar_dof_id] -
                        s_stress_component[0];
                    }
                  else
                    {
                      // When using cell-wise  SABLE stress is used, the SABLE
                      // stress is added at in assmeble_force() function
                      sable_solver.fsi_stress[stress_index][scalar_dof_id] =
                        -s_stress_component[0];
                    }
                }
              stress_index++;
            }
        }

      if (parameters.indicator_field_condition == "PartiallyInsideSolid")
        {

          // distribute fsi velocity to the nodes which are outside solid and
          // belongs to
          // cell which is partially inside the solid
          std::vector<int> vertex_visited(
            sable_solver.triangulation.n_vertices(), 0);
          for (auto f_cell = sable_solver.dof_handler.begin_active();
               f_cell != sable_solver.dof_handler.end();
               ++f_cell)
            {
              int cell_count = f_cell->index();
              if (cell_partially_inside_solid[cell_count])
                {
                  // get average solution from the nodes which are inside the
                  // solid
                  std::vector<double> solution_vec(dim, 0);
                  std::vector<int> nodes_inside =
                    cell_nodes_inside_solid[cell_count];
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

                  // distribute solution to the nodes which are outside the
                  // solid
                  std::vector<int> nodes_outside =
                    cell_nodes_outside_solid[cell_count];
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
                          // average the solution if the corresponding node is
                          // visited more than once
                          sable_solver.fsi_velocity[vertex_dof_index] /=
                            vertex_visited[vertex_index];
                        }
                    }
                }
            }
        }
    }
  move_solid_mesh(false);
}

template <int dim>
int OpenIFEM_Sable_FSI<dim>::compute_fluid_cell_index(
  Point<dim> &q_point, const Tensor<1, dim> &normal)
{
  // Note: Only works for unifrom, strucutred mesh
  auto f_cell = sable_solver.triangulation.begin_active();

  // compute the lower boundary of the Eulerian cell box

  double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));

  Point<dim> lower_boundary;

  for (unsigned int i = 0; i < dim; i++)
    lower_boundary(i) = f_cell->center()[i] - h / 2;

  // Currently assuming the target Eulerian cell has the same level as the first
  // Eulerian cell, won't work for AMR

  f_cell = sable_solver.triangulation.last_active();

  unsigned int N = 0;

  if (parameters.dimension == 2)
    {
      N = std::sqrt((f_cell->index() + 1));
    }
  else
    {
      N = std::cbrt((f_cell->index() + 1));
    }

  // compute the upper boundaries of the Eulerian cell box

  Point<dim> upper_boundary;

  for (unsigned int i = 0; i < dim; i++)
    upper_boundary(i) = f_cell->center()[i] + h / 2;

  bool point_inside = true;

  for (unsigned int i = 0; i < dim; i++)

    {

      if (q_point[i] < lower_boundary(i) && q_point[i] > upper_boundary(i))
        {
          point_inside = false;
          break;
        }
    }

  if (point_inside == true)

    {
      bool point_not_on_edge = true;

      for (unsigned int i = 0; i < dim; i++)

        {

          if (std::floor(q_point[i] / h) == (q_point[i] / h))
            {
              point_not_on_edge = false;
              break;
            }
        }

      // compute the theorical min and max cell ID for assertion
      int a = 0;
      int b = static_cast<int>(std::pow(N, dim) - 1);

      if (point_not_on_edge == true) // if the quad point is not on the edge of
                                     // the Eulerian cell
        {
          // compute the Eulerian cell index

          int n = 0;

          for (unsigned int i = 0; i < dim; i++)

            n += static_cast<int>(std::pow(N, i)) *
                 static_cast<int>(
                   std::floor((q_point[i] - lower_boundary(i)) / h));

          if (n < a || n > b)
            return -1;

          return n;
        }

      else // if the quad point is on the edge of the Eulerian
           // cell
        {
          // create a small distance in the outnormal direction
          const double tmp = h * 1e-6;

          int n = 0;

          // extend the current quad point positions along the
          // outward normal direction
          for (unsigned int i = 0; i < dim; i++)
            {
              q_point(i) = q_point(i) + tmp * normal[i];

              n += (q_point(i) < lower_boundary(i))
                     ? static_cast<int>(
                         std::ceil((q_point[i] - lower_boundary(i)) / h))
                     :

                     static_cast<int>(
                       std::floor((q_point[i] - lower_boundary(i)) / h));
            }

          if (n < a || n > b)
            return -1;

          return n;
        }
    }
  else // if the quad point is outside the fluid box
    return -1;
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
  auto f_cell = sable_solver.triangulation.begin_active();

  double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));

  // Currently assuming the target Eulerian cell has the same level as the first
  // Eulerian cell, won't work for AMR
  auto level = f_cell->level();

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
                  // old way//
                  // Get interpolated solution from the fluid
                  /*Vector<double> value(dim + 1);
                  Utils::GridInterpolator<dim, BlockVector<double>>
                    interpolator(sable_solver.dof_handler, q_point_extension);
                  interpolator.point_value(sable_solver.present_solution,
                                           value);
                  // Create the scalar interpolator for stresses based on the
                  // existing interpolator
                  auto f_cell = interpolator.get_cell();*/

                  // compute the Eulerian cell index
                  int n = compute_fluid_cell_index(q_point_extension, normal);

                  // if the solid quad point is within the fluid box
                  if (n != -1)
                    {
                      // construct the cell iterator that points to the
                      // desired Eulerian index
                      TriaActiveIterator<DoFCellAccessor<dim, dim, false>>
                        f_cell_temp(&sable_solver.triangulation,
                                    level,
                                    n,
                                    &sable_solver.dof_handler);
                      f_cell = f_cell_temp;
                    }

                  else // If the quadrature point is outside background mesh
                       // if (f_cell->index() == -1)
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
                  // old way //
                  // stress tensor from SABLE includes pressure //
                  /*SymmetricTensor<2, dim> stress =
                    -value[dim] * Physics::Elasticity::StandardTensors<dim>::I +
                    viscous_stress;*/
                  SymmetricTensor<2, dim> stress = viscous_stress;
                  ptr[f]->fsi_traction.push_back(stress * normal);
                }
            }
        }
    }
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::compute_added_mass()
{
  TimerOutput::Scope timer_section(timer, "Compute Added Mass");

  solid_solver.added_mass_effect.reinit(solid_solver.dof_handler.n_dofs());

  move_solid_mesh(true);
  std::vector<bool> vertex_touched(solid_solver.triangulation.n_vertices(),
                                   false);

  for (auto s_cell = solid_solver.dof_handler.begin_active();
       s_cell != solid_solver.dof_handler.end();
       ++s_cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          // Current face is at boundary.
          if (s_cell->face(f)->at_boundary())
            {
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face;
                   ++v)
                {
                  auto face = s_cell->face(f);
                  auto vertex = face->vertex(v);
                  if (!vertex_touched[face->vertex_index(v)])
                    {
                      vertex_touched[face->vertex_index(v)] = 1;
                      Vector<double> value(1);
                      // interpolate nodal mass
                      Utils::GridInterpolator<dim, Vector<double>>
                        scalar_interpolator(sable_solver.scalar_dof_handler,
                                            vertex);
                      scalar_interpolator.point_value(sable_solver.nodal_mass,
                                                      value);
                      // add nodal mass to added_mass_effect vector
                      for (unsigned int i = 0; i < dim; i++)
                        {
                          auto index = face->vertex_dof_index(v, i);
                          solid_solver.added_mass_effect[index] = value[0];
                        }
                    }
                }
            }
        }
    }
  solid_solver.constraints.condense(solid_solver.added_mass_effect);
  move_solid_mesh(false);
}

template <int dim>
void OpenIFEM_Sable_FSI<dim>::output_vel_diff(bool first_step)
{

  std::ofstream file_diff;
  if (first_step)
    {
      file_diff.open("velocity_diff.txt");
      file_diff << "Time"
                << "\t"
                << "vel_diff_Eul"
                << "\t"
                << "vel_diff_Lag"
                << "\t"
                << "vel_diff_Eul_scaled"
                << "\t"
                << "vel_diff_Lag_scaled"
                << "\t"
                << "KE_diff_Eul"
                << "\t"
                << "KE_diff_Lag"
                << "\n";
    }
  else
    {
      file_diff.open("velocity_diff.txt", std::ios_base::app);
    }
  // calculate velocity difference between the two domains at Lagrangian mesh
  move_solid_mesh(true);
  Vector<double> vel_diff_lag(solid_solver.dof_handler.n_dofs());
  std::vector<bool> vertex_touched(solid_solver.triangulation.n_vertices(),
                                   false);

  // double maxEulVel = -1.0e+35;
  double maxLagVel = -1.0e+35;
  // interpolate Eulerian velocity to Lagrangian mesh
  for (auto s_cell = solid_solver.dof_handler.begin_active();
       s_cell != solid_solver.dof_handler.end();
       ++s_cell)
    {

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if (!vertex_touched[s_cell->vertex_index(v)])
            {
              vertex_touched[s_cell->vertex_index(v)] = true;
              Vector<double> value(dim + 1);
              Utils::GridInterpolator<dim, BlockVector<double>> interpolator(
                sable_solver.dof_handler, s_cell->vertex(v));
              interpolator.point_value(sable_solver.present_solution, value);
              auto f_cell = interpolator.get_cell();
              // save Lagrangian velocity at the given vertex
              Vector<double> lagVel(dim);
              if (f_cell->index() != -1)
                {
                  // get material vf of background Eulerian cell
                  auto ptr = sable_solver.cell_wise_stress.get_data(f_cell);
                  double vf = ptr[0]->material_vf;
                  auto ptr_i = sable_solver.cell_property.get_data(f_cell);
                  double indicator = ptr_i[0]->indicator;
                  for (unsigned int i = 0; i < dim; i++)
                    {
                      auto index = s_cell->vertex_dof_index(v, i);
                      vel_diff_lag(index) = value[i];
                      // subtract Lagrangian solid velocity
                      vel_diff_lag(index) -=
                        solid_solver.current_velocity(index);
                      // scale velocity difference by background Eulerian volume
                      // fraction and indicator
                      vel_diff_lag(index) *= vf * indicator;

                      lagVel[i] = solid_solver.current_velocity(index);
                    }
                  /*double vnorm = value.l2_norm();
                  if (vnorm > maxEulVel)
                    maxEulVel = vnorm;*/
                  double vnorm = lagVel.l2_norm();
                  if (vnorm > maxLagVel)
                    maxLagVel = vnorm;
                }
            }
        }
    }
  move_solid_mesh(false);
  solid_solver.fsi_vel_diff_lag = vel_diff_lag;
  // calculate velocity difference between the two domains at Eulerian mesh
  Vector<double> vel_diff_eul;
  vel_diff_eul = sable_solver.fsi_vel_diff_eul.block(0);
  // output l2 norms of the two vectors
  // maxEulVel = 1.0 / std::max(maxEulVel, 1.0);
  maxLagVel = 1.0 / std::max(maxLagVel, 1.0);
  double vel_diff_norm_eul = vel_diff_eul.l2_norm();
  double vel_diff_norm_lag = vel_diff_lag.l2_norm();
  double vel_norm_lag = solid_solver.current_velocity.l2_norm();

  file_diff << time.current() << "\t" << vel_diff_norm_eul << "\t"
            << vel_diff_norm_lag << "\t"
            << vel_diff_norm_eul * maxLagVel /
                 sable_solver.triangulation.n_vertices()
            << "\t"
            << vel_diff_norm_lag * maxLagVel /
                 sable_solver.triangulation.n_vertices()
            << "\t" << vel_diff_norm_eul / vel_norm_lag << "\t"
            << vel_diff_norm_lag / vel_norm_lag << "\n";
  file_diff.close();
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
      setup_cell_hints();
      find_solid_bc();
      if (parameters.use_added_mass == "yes")
        {
          compute_added_mass();
        }
      solid_solver.run_one_step(first_step);
      // indicator field
      update_solid_box();

      if (parameters.fsi_force_criteria == "Nodes")
        {
          update_indicator();
          find_fluid_bc();
          // find_fluid_bc_new();
        }
      else
        {
          update_indicator_qpoints();
          find_fluid_bc_qpoints();
        }
      // send_indicator_field
      sable_solver.send_fsi_force(sable_solver.sable_no_nodes);
      sable_solver.send_indicator(sable_solver.sable_no_ele,
                                  sable_solver.sable_no_nodes);
      sable_solver.run_one_step();
      output_vel_diff(first_step);
      first_step = false;
    }
}

template class OpenIFEM_Sable_FSI<2>;
template class OpenIFEM_Sable_FSI<3>;

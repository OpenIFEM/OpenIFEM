#include "mpi_openifem_sable_fsi.h"
#include <complex>
#include <iostream>

namespace MPI
{

  template <int dim>
  OpenIFEM_Sable_FSI<dim>::~OpenIFEM_Sable_FSI()
  {
    timer.print_summary();
  }

  template <int dim>
  OpenIFEM_Sable_FSI<dim>::OpenIFEM_Sable_FSI(
    Fluid::MPI::SableWrap<dim> &f,
    Solid::MPI::SharedSolidSolver<dim> &s,
    const Parameters::AllParameters &p,
    bool use_dirichlet_bc)
    : FSI<dim>(f, s, p, use_dirichlet_bc), sable_solver(f)
  {
    assert(use_dirichlet_bc == false);
  }

  template <int dim>
  void OpenIFEM_Sable_FSI<dim>::run()
  {
    solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
    sable_solver.setup_dofs();
    sable_solver.make_constraints();
    sable_solver.initialize_system();
    solid_solver.setup_dofs();
    solid_solver.initialize_system();

    pcout << "Number of fluid active cells and dofs: ["
          << sable_solver.triangulation.n_active_cells() << ", "
          << sable_solver.dof_handler.n_dofs() << "]" << std::endl
          << "Number of solid active cells and dofs: ["
          << solid_solver.triangulation.n_active_cells() << ", "
          << solid_solver.dof_handler.n_dofs() << "]" << std::endl;

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

        solid_solver.run_one_step(first_step);
        // indicator field
        update_solid_box();

        if (parameters.fsi_force_criteria == "Nodes")
          {
          }
        else
          {
            update_indicator_qpoints();
          }
        // send_indicator_field
        sable_solver.send_fsi_force(sable_solver.sable_no_nodes);
        sable_solver.send_indicator(sable_solver.sable_no_ele,
                                    sable_solver.sable_no_nodes);
        sable_solver.run_one_step();
        // output_vel_diff(first_step);
        first_step = false;
      }
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

        Point<dim> maxp = cell->vertex(0);
        Point<dim> minp = cell->vertex(0);

        for (unsigned int v = 1; v < cell->n_vertices(); ++v)
          for (unsigned int d = 0; d < dim; ++d)
            {
              maxp[d] = std::max(maxp[d], cell->vertex(v)[d]);
              minp[d] = std::min(minp[d], cell->vertex(v)[d]);
            }

        // rule out points outside the
        // bounding box of this cell
        bool inside_box = true;
        for (unsigned int d = 0; d < dim; ++d)
          {
            if ((point[d] < minp[d]) || (point[d] > maxp[d]))
              {
                inside_box = false;
                break;
              }
          }

        if (!inside_box)
          continue;

        if (point_in_cell(cell, point))
          return {true, cell};
      }
    return {false, {}};
  }

  template <int dim>
  bool OpenIFEM_Sable_FSI<dim>::point_in_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Point<dim> &p)
  {

    if (dim == 2)
      {
        return (cell->point_inside(p));
      }
    else
      {
        // we need to check more carefully: transform to the
        // unit cube and check there. unfortunately, this isn't
        // completely trivial since the transform_real_to_unit_cell
        // function may throw an exception that indicates that the
        // point given could not be inverted. we take this as a sign
        // that the point actually lies outside, as also documented
        // for that function
        double tolerence = 1e-10;
        MappingQ1<dim> mapping;
        try
          {
            auto p_unit = mapping.transform_real_to_unit_cell(cell, p);
            return GeometryInfo<dim>::is_inside_unit_cell(p_unit, tolerence);
          }
        catch (const Mapping<3, 3>::ExcTransformationFailed &)
          {
            return false;
          }
      }
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

    for (auto f_cell = sable_solver.dof_handler.begin_active();
         f_cell != sable_solver.dof_handler.end();
         ++f_cell)
      {

        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        fe_values.reinit(f_cell);

        // check which cell nodes are inside cells to calculate velocity bc
        std::vector<int> inside_nodes;
        std::vector<int> outside_nodes;
        unsigned int inside_count = 0;
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            auto is_inside_solid =
              point_in_solid_new(solid_solver.dof_handler, f_cell->vertex(v));
            if (is_inside_solid.first)
              {
                inside_nodes.push_back(v);
                ++inside_count;
              }
            else
              outside_nodes.push_back(v);
          }

        auto p = sable_solver.cell_property.get_data(f_cell);
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

        auto q_points = fe_values.get_quadrature_points();
        unsigned int inside_qpoint = 0;

        for (unsigned int q = 0; q < q_points.size(); q++)
          {
            if (point_in_solid_new(solid_solver.dof_handler, q_points[q]).first)
              {
                ++inside_qpoint;
              }
          }

        AssertThrow(
          parameters.indicator_field_condition == "CompletelyInsideSolid",
          ExcMessage(
            "PartiallyInsideSolid option is not implemented in the module"));

        p[0]->indicator = (inside_qpoint == q_points.size() ? 1 : 0);

        // update exact indicator field
        // initialize it to zero
        p[0]->exact_indicator = 0;
        // get upper and lower corner for the Eulerian cell
        Point<dim> l_eu = f_cell->vertex(0);
        Point<dim> u_eu;
        if (dim == 2)
          u_eu = f_cell->vertex(3);
        else
          u_eu = f_cell->vertex(7);
        // get eulerian cell size
        double h = abs(f_cell->vertex(0)(0) - f_cell->vertex(1)(0));
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

                if (dim == 2)
                  {
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
                else
                  {
                    for (int k = 0; k < n + 1; k++)
                      {
                        sample[2] = l_eu[2] + dh * k;
                        bool inside_box = true;
                        for (unsigned int d = 0; d < 1; d++)
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
                if (point_in_cell(s_cell, sample))
                  sample_inside += 1;
              }

            p[0]->exact_indicator +=
              (double(sample_inside) / double(sample_count));
          }
        // if the exact indicator is greater than one then round it off to 1
        if (p[0]->exact_indicator > 1.0)
          p[0]->exact_indicator = 1.0;
      }

    move_solid_mesh(false);
  }

  template class OpenIFEM_Sable_FSI<2>;
  template class OpenIFEM_Sable_FSI<3>;

} // namespace MPI

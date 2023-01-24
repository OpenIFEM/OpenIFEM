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

    cell_nodes_inside_solid.clear();
    cell_nodes_outside_solid.clear();
    vertex_indicator_data.clear();

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
                vertex_indicator_data.insert(
                  {f_cell->vertex_index(v), is_inside_solid.second});
              }
            else
              outside_nodes.push_back(v);
          }

        cell_nodes_inside_solid.insert({f_cell->index(), inside_nodes});
        cell_nodes_outside_solid.insert({f_cell->index(), outside_nodes});

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

  template <int dim>
  void OpenIFEM_Sable_FSI<dim>::find_fluid_bc_qpoints()
  {
    TimerOutput::Scope timer_section(
      timer, "Find fluid BC based on quadrature points");
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
    // sable_solver.fsi_force_acceleration_part = 0;
    // sable_solver.fsi_force_stress_part = 0;
    // sable_solver.fsi_penalty_force = 0;
    sable_solver.fsi_velocity = 0;

    Vector<double> localized_solid_acceleration(
      solid_solver.current_acceleration);
    Vector<double> localized_solid_velocity(solid_solver.current_velocity);

    PETScWrappers::MPI::BlockVector tmp_fsi_force;
    tmp_fsi_force.reinit(fluid_solver.owned_partitioning,
                         fluid_solver.mpi_communicator);

    std::vector<std::vector<Vector<double>>> localized_stress(
      dim, std::vector<Vector<double>>(dim));
    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          {
            localized_stress[i][j] = solid_solver.stress[i][j];
          }
      }

    const unsigned int dofs_per_cell = sable_solver.fe.dofs_per_cell;
    const unsigned int u_dofs = sable_solver.fe.base_element(0).dofs_per_cell;
    const unsigned int p_dofs = sable_solver.fe.base_element(1).dofs_per_cell;
    const unsigned int n_q_points = sable_solver.volume_quad_formula.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
                ExcMessage("Wrong partitioning of dofs!"));

    Vector<double> local_rhs(dofs_per_cell);
    // Vector<double> local_rhs_acceleration_part(dofs_per_cell);
    // Vector<double> local_rhs_stress_part(dofs_per_cell);
    // Vector<double> local_penalty_force(dofs_per_cell);

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

        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        auto ptr = sable_solver.cell_property.get_data(f_cell);
        const double ind = ptr[0]->indicator;
        const double ind_exact = ptr[0]->exact_indicator;
        auto s = sable_solver.sable_cell_data.get_data(f_cell);

        if (ind == 0)
          continue;
        /*const double rho_bar =
          parameters.solid_rho * ind + s[0]->eulerian_density * (1 - ind);*/
        const double rho_bar = parameters.solid_rho * ind_exact +
                               s[0]->eulerian_density * (1 - ind_exact);

        fe_values.reinit(f_cell);
        scalar_fe_values.reinit(scalar_f_cell);

        local_rhs = 0;
        // local_rhs_acceleration_part = 0;
        // local_rhs_stress_part = 0;
        // local_penalty_force = 0;

        // Fluid velocity at support points
        fe_values[velocities].get_function_values(sable_solver.present_solution,
                                                  v);
        // Fluid velocity increment at support points
        fe_values[velocities].get_function_values(
          sable_solver.solution_increment, dv);
        // Fluid velocity gradient at support points
        fe_values[velocities].get_function_gradients(
          sable_solver.present_solution, grad_v);

        auto q_points = fe_values.get_quadrature_points();
        for (unsigned int q = 0; q < n_q_points; q++)
          {

            if (parameters.indicator_field_condition == "PartiallyInsideSolid")
              {
                if (!point_in_solid_new(solid_solver.dof_handler, q_points[q])
                       .first)
                  continue;
              }

            Utils::GridInterpolator<dim, Vector<double>> interpolator(
              solid_solver.dof_handler, q_points[q]);
            if (!interpolator.found_cell())
              {
                std::stringstream message;
                message << "Cannot find point in solid: " << q_points[q]
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
            // add penalty force based on the velocity difference
            Tensor<1, dim> fsi_penalty_tensor;
            fsi_penalty_tensor = parameters.penalty_scale_factor[1] *
                                 ((vs - v[q]) / time.get_delta_t());
            fsi_acc_tensor += fsi_penalty_tensor;

            SymmetricTensor<2, dim> f_cell_stress;
            int count = 0;
            for (unsigned int k = 0; k < dim; k++)
              {
                for (unsigned int m = 0; m < k + 1; m++)
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
                for (unsigned int m = 0; m < k + 1; m++)
                  {

                    Vector<double> s_stress_component(1);
                    scalar_interpolator.point_value(localized_stress[k][m],
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
                /*local_rhs_acceleration_part(i) +=
                  (rho_bar * fsi_acc_tensor * phi_u[i]) * fe_values.JxW(q);
                local_rhs_stress_part(i) +=
                  (scalar_product(grad_phi_u[i], fsi_stress_tensor)) *
                  fe_values.JxW(q);
                local_penalty_force(i) +=
                  (rho_bar * fsi_penalty_tensor * phi_u[i]) *
                fe_values.JxW(q);*/
              }
          }

        f_cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            // sable_solver.system_rhs[local_dof_indices[i]] += local_rhs(i);
            tmp_fsi_force[local_dof_indices[i]] += local_rhs(i);
            /*sable_solver.fsi_force_acceleration_part[local_dof_indices[i]] +=
              local_rhs_acceleration_part(i);
            sable_solver.fsi_force_stress_part[local_dof_indices[i]] +=
              local_rhs_stress_part(i);
            sable_solver.fsi_penalty_force[local_dof_indices[i]] +=
              local_penalty_force(i);*/
          }
      }

    tmp_fsi_force.compress(VectorOperation::add);
    sable_solver.fsi_force = tmp_fsi_force;

    // interpolate velocity to the nodes inside Lagrangian solid
    std::vector<int> vertex_touched(sable_solver.triangulation.n_vertices(), 0);
    PETScWrappers::MPI::Vector tmp_fsi_velocity;
    tmp_fsi_velocity.reinit(fluid_solver.owned_partitioning[0],
                            fluid_solver.mpi_communicator);

    for (auto f_cell = sable_solver.dof_handler.begin_active();
         f_cell != sable_solver.dof_handler.end();
         ++f_cell)
      {

        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        auto ptr = sable_solver.cell_property.get_data(f_cell);
        if (ptr[0]->indicator == 0)
          continue;

        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            auto vertex = f_cell->vertex(v);
            int key = f_cell->vertex_index(v);
            // check if the vertex is inside Lagrangian solid mesh
            if (vertex_indicator_data.find(key) == vertex_indicator_data.end())
              continue;

            if (vertex_touched[f_cell->vertex_index(v)])
              continue;
            vertex_touched[f_cell->vertex_index(v)] = 1;

            // get Lagrangian cell iterator which holds the given vertex
            auto s_cell = vertex_indicator_data[key];
            // construct the interpolator
            Utils::GridInterpolator<dim, Vector<double>> interpolator(
              solid_solver.dof_handler, vertex, {}, s_cell);
            if (!interpolator.found_cell())
              {
                std::stringstream message;
                message << "Cannot find point in solid: " << vertex
                        << std::endl;
                AssertThrow(interpolator.found_cell(),
                            ExcMessage(message.str()));
              }

            // Lagrangian solid velocity at Eulerian nodes
            Vector<double> solid_vel(dim);
            interpolator.point_value(localized_solid_velocity, solid_vel);

            int n_dofs_per_vertex = sable_solver.fe.n_dofs_per_vertex();
            // skip pressure dof
            n_dofs_per_vertex -= 1;
            for (int i = 0; i < n_dofs_per_vertex; i++)
              {
                int dof_index = f_cell->vertex_dof_index(v, i);
                tmp_fsi_velocity[dof_index] = solid_vel[i];
              }
          }
      }

    tmp_fsi_velocity.compress(VectorOperation::insert);
    sable_solver.fsi_velocity = tmp_fsi_velocity;

    // distribute solution to the nodes which are outside solid and belongs to
    // cell which is partially inside the solid
    PETScWrappers::MPI::Vector dofs_visited;
    dofs_visited.reinit(fluid_solver.owned_partitioning[0],
                        fluid_solver.mpi_communicator);
    for (auto f_cell = sable_solver.dof_handler.begin_active();
         f_cell != sable_solver.dof_handler.end();
         ++f_cell)
      {

        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        auto ptr = sable_solver.cell_property.get_data(f_cell);
        int cell_count = f_cell->index();
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
            std::vector<int> nodes_outside =
              cell_nodes_outside_solid[cell_count];
            for (unsigned int i = 0; i < nodes_outside.size(); i++)
              {
                for (unsigned int j = 0; j < dim; j++)
                  {
                    int vertex_dof_index =
                      f_cell->vertex_dof_index(nodes_outside[i], j);
                    dofs_visited[vertex_dof_index] += 1;

                    tmp_fsi_velocity[vertex_dof_index] += solution_vec[j];
                  }
              }
          }
      }

    tmp_fsi_velocity.compress(VectorOperation::add);
    dofs_visited.compress(VectorOperation::add);

    sable_solver.fsi_velocity = tmp_fsi_velocity;
    Vector<double> localized_dofs_visited(dofs_visited);

    // if a particular vertex is visited more than once, average the fsi
    // velocity
    for (unsigned int i = 0; i < localized_dofs_visited.size(); i++)
      {
        if (localized_dofs_visited[i] != 0)
          sable_solver.fsi_velocity[i] /= localized_dofs_visited[i];
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

    // Currently assuming the target Eulerian cell has the same level as the
    // first Eulerian cell, won't work for AMR

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

        if (point_not_on_edge == true) // if the quad point is not on the edge
                                       // of the Eulerian cell
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

    // Currently assuming the target Eulerian cell has the same level as the
    // first Eulerian cell, won't work for AMR
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

                    // If the quadrature point is outside background mesh
                    // if (f_cell->index() == -1)
                    if (n == -1)
                      {
                        Tensor<1, dim> zero_tensor;
                        ptr[f]->fsi_traction.push_back(zero_tensor);
                        continue;
                      }
                    else
                      {
                        TriaActiveIterator<DoFCellAccessor<dim, dim, false>>
                          f_cell_temp(&sable_solver.triangulation,
                                      level,
                                      n,
                                      &sable_solver.dof_handler);
                        f_cell = f_cell_temp;
                      }

                    if (!f_cell->is_locally_owned())
                      {
                        Tensor<1, dim> zero_tensor;
                        ptr[f]->fsi_traction.push_back(zero_tensor);
                        continue;
                      }

                    // get cell-wise stress from SABLE
                    auto ptr_f = sable_solver.sable_cell_data.get_data(f_cell);
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
                        for (unsigned int j = 0; j < i + 1; j++)
                          {
                            // Interpolate stress from nodal stress field
                            AssertThrow(
                              parameters.traction_calculation_option ==
                                "CellBased",
                              ExcMessage("NodeBased option is not implemented "
                                         "in this solver"));

                            // Get cell-wise stress
                            viscous_stress[i][j] =
                              ptr_f[0]->cell_stress_no_bgmat[count];

                            count++;
                          }
                      }
                    // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                    // old way //
                    // stress tensor from SABLE includes pressure //
                    /*SymmetricTensor<2, dim> stress =
                      -value[dim] * Physics::Elasticity::StandardTensors<dim>::I
                      + viscous_stress;*/
                    SymmetricTensor<2, dim> stress = viscous_stress;
                    ptr[f]->fsi_traction.push_back(stress * normal);
                  }
              }
          }
      }

    // add up all local traction vectors
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
                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    ptr[f]->fsi_traction[q] = Utilities::MPI::sum(
                      ptr[f]->fsi_traction[q], solid_solver.mpi_communicator);
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
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_face;
                     ++v)
                  {
                    auto face = s_cell->face(f);
                    auto vertex = face->vertex(v);
                    if (!vertex_touched[face->vertex_index(v)])
                      {
                        vertex_touched[face->vertex_index(v)] = 1;
                        Vector<double> value(1);
                        // interpolate nodal mass
                        Utils::GridInterpolator<dim, PETScWrappers::MPI::Vector>
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
    Utilities::MPI::sum(solid_solver.added_mass_effect,
                        solid_solver.mpi_communicator,
                        solid_solver.added_mass_effect);
    move_solid_mesh(false);
  }

  template class OpenIFEM_Sable_FSI<2>;
  template class OpenIFEM_Sable_FSI<3>;

} // namespace MPI

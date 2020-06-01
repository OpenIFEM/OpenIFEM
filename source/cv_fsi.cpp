#include "cv_fsi.h"

namespace
{
  template <int dim>
  std::vector<Point<dim>>
  compute_cut_points(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     double x)
  {
    std::vector<Point<dim>> cut_points;
    // Compute the intersections between the inlet/outlet surface and the cells
    std::set<unsigned int> left_vertices, right_vertices;
    auto is_left_to = [](double cell_x, double cv_x) { return cell_x <= cv_x; };
    // find the lines being cut
    std::vector<unsigned int> cut_lines;
    for (unsigned l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
      {
        if (is_left_to(cell->line(l)->vertex(0)[0], x) !=
            is_left_to(cell->line(l)->vertex(1)[0], x))
          {
            cut_lines.push_back(l);
          }
      }
    for (auto l : cut_lines)
      {
        auto p1 = cell->line(l)->vertex(0);
        auto p2 = cell->line(l)->vertex(1);
        Point<dim> cut_point;
        cut_point[0] = x;
        for (unsigned d = 1; d < dim; ++d)
          {
            if (p1[d] != p2[d])
              {
                cut_point[d] = p1[d] + (x - p1[d]) / (p2[d] - p1[d]);
              }
            else
              {
                cut_point[d] = p1[d];
              }
          }
        cut_points.push_back(cut_point);
      }
    // Note the output vector must have the ordering "bottom, top" in case of
    // 3D!
    std::sort(cut_points.begin(),
              cut_points.end(),
              [](Point<dim> a, Point<dim> b) { return a[1] < b[1]; });
    if (dim == 3)
      {
        std::sort(cut_points.begin(),
                  cut_points.end(),
                  [](Point<dim> a, Point<dim> b) { return a[2] < b[2]; });
      }
    return cut_points;
  }

  // Compute the volume of the cut cell using Gauss theorem
  template <int dim>
  double compute_volume_fraction(
    const typename DoFHandler<dim>::active_cell_iterator cell,
    const std::vector<Point<dim>> &cut_points,
    std::string inlet_outlet)
  {
    double volume = 0.0;
    double boundary_x = cut_points[0][0];
    auto is_inside = inlet_outlet == "inlet"
                       ? [](Point<dim> p, double x) { return p[0] > x; }
                       : [](Point<dim> p, double x) { return p[0] <= x; };

    if (dim == 2)
      {
        auto lower = [](Point<dim> a, Point<dim> b) { return a[1] < b[1]; };
        std::set<Point<dim>, decltype(lower)> cell_points(lower);
        for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            Point<dim> vertex = cell->vertex(v);
            if (is_inside(vertex, boundary_x))
              {
                cell_points.insert(vertex);
              }
          }
        // Construct the faces and face centers for the cut shape
        std::list<std::pair<Tensor<1, dim>, Tensor<1, dim>>> edges_and_centers;
        auto add_edge = [&edges_and_centers](Point<dim> p1,
                                             Point<dim> p2) mutable {
          edges_and_centers.push_back(
            std::pair<Point<dim>, Point<dim>>(p1 - p2, 0.5 * (p1 + p2)));
        };

        add_edge(*(cut_points.begin()), *(cell_points.begin()));
        add_edge(*(cut_points.rbegin()), *(cell_points.rbegin()));
        add_edge(*(cut_points.rbegin()), *(cut_points.begin()));
        // If the cut cell is quadrilateral
        if (cell_points.size() == 2)
          {
            add_edge(*(cell_points.rbegin()), *(cell_points.begin()));
          }
        // Remove trivial edges (nearly zero)
        std::remove_if(edges_and_centers.begin(),
                       edges_and_centers.end(),
                       [](std::pair<Tensor<1, dim>, Tensor<1, dim>> e) {
                         return e.first.norm() < 1e-10;
                       });

        // If there are only 2 edges then they must coincide with the cell edges
        if (edges_and_centers.size() < 3)
          {
            return 0.0;
          }

        // Compute the center of the cut cell
        Tensor<1, dim> center({0, 0});
        auto add_point = [&center](Point<dim> p) mutable { center += p; };
        std::for_each(cut_points.begin(), cut_points.end(), add_point);
        std::for_each(cell_points.begin(), cell_points.end(), add_point);
        center /= (cut_points.size() + cell_points.size());

        for (auto item : edges_and_centers)
          {
            const auto &e = item.first;
            const auto &edge_center = item.second;
            auto edge_normal = cross_product_2d(e) / e.norm();
            volume +=
              0.5 *
              std::abs(scalar_product(edge_center - center, edge_normal)) *
              e.norm();
          }
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    double volume_fraction = volume / cell->measure();
    return volume_fraction;
  }
} // namespace

namespace MPI
{
  template <int dim>
  ControlVolumeFSI<dim>::ControlVolumeFSI(Fluid::MPI::FluidSolver<dim> &f,
                                          Solid::MPI::SharedSolidSolver<dim> &s,
                                          const Parameters::AllParameters &p,
                                          bool use_dirichlet_bc)
    : FSI<dim>(f, s, p, use_dirichlet_bc)
  {
  }

  template <int dim>
  ControlVolumeFSI<dim>::~ControlVolumeFSI()
  {
    cv_values.output.close();
  }

  template <int dim>
  void ControlVolumeFSI<dim>::run_with_cv_analysis()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
    // Try load from previous computation.
    bool success_load =
      solid_solver.load_checkpoint() && fluid_solver.load_checkpoint();
    AssertThrow(
      solid_solver.time.current() == fluid_solver.time.current(),
      ExcMessage("Solid and fluid restart files have different time steps. "
                 "Check and remove inconsistent restart files!"));
    if (!success_load)
      {
        solid_solver.setup_dofs();
        solid_solver.initialize_system();
        fluid_solver.triangulation.refine_global(
          parameters.global_refinements[0]);
        fluid_solver.setup_dofs();
        fluid_solver.make_constraints();
        fluid_solver.initialize_system();
        fluid_previous_solution.reinit(fluid_solver.owned_partitioning,
                                       fluid_solver.relevant_partitioning,
                                       mpi_communicator);
      }
    else
      {
        while (time.get_timestep() < solid_solver.time.get_timestep())
          {
            time.increment();
          }
      }

    collect_solid_boundaries();
    setup_cell_hints();
    update_vertices_mask();
    collect_inlet_outlet_cells();
    collect_control_volume_cells();
    cv_values.initialize_output(this->mpi_communicator);

    pcout << "Number of fluid active cells and dofs: ["
          << fluid_solver.triangulation.n_active_cells() << ", "
          << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
          << "Number of solid active cells and dofs: ["
          << solid_solver.triangulation.n_active_cells() << ", "
          << solid_solver.dof_handler.n_dofs() << "]" << std::endl;
    bool first_step = !success_load;
    if (parameters.refinement_interval < parameters.end_time)
      {
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        setup_cell_hints();
      }
    while (time.end() - time.current() > 1e-12)
      {
        find_solid_bc();
        if (success_load)
          {
            solid_solver.assemble_system(true);
          }
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

        control_volume_analysis();
        fluid_previous_solution = fluid_solver.present_solution;

        if (time.time_to_refine())
          {
            refine_mesh(parameters.global_refinements[0],
                        parameters.global_refinements[0] + 3);
            setup_cell_hints();
          }
        if (time.time_to_save())
          {
            solid_solver.save_checkpoint(time.get_timestep());
            fluid_solver.save_checkpoint(time.get_timestep());
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::set_control_volume_boundary(
    const std::vector<double> &boundaries)
  {
    // Check the validity of the size
    AssertThrow(boundaries.size() == 2 * dim,
                ExcMessage("Wrong control volume boundary size!"));
    // Check the validity of the values
    for (unsigned i = 0; i < dim; ++i)
      {
        AssertThrow(boundaries[2 * i] < boundaries[2 * i + 1],
                    ExcMessage("Wrong control volume boundary values!"));
      }
    // Assign the boundaries
    this->control_volume_boundaries.assign(boundaries.begin(),
                                           boundaries.end());
  }

  template <int dim>
  void ControlVolumeFSI<dim>::collect_control_volume_cells()
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }
        const Point<dim> &center = f_cell->center();
        // Check if the fluid cell is inside the control volume
        bool in_cv = true;
        for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            const Point<dim> &vertex = f_cell->vertex(v);
            if (vertex[0] <= control_volume_boundaries[0] ||
                vertex[0] > control_volume_boundaries[1])
              {
                in_cv = false;
              }
          }
        if (in_cv)
          {
            cv_f_cells[center[0]].insert(
              std::pair<double, typename DoFHandler<dim>::active_cell_iterator>(
                center[1], f_cell));
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::collect_inlet_outlet_cells()
  {
    // Initialize the surface cutters cell storage
    for (auto cell = fluid_solver.triangulation.begin_active();
         cell != fluid_solver.triangulation.end();
         ++cell)
      {
        if (cell->is_locally_owned())
          {
            surface_cutters.initialize(cell, 1);
          }
      }
    // DoFs per block for the cutters;
    std::vector<unsigned int> cutter_dofs_per_block;
    cutter_dofs_per_block.resize(2);
    cutter_dofs_per_block[0] = dim * GeometryInfo<dim - 1>::vertices_per_cell;
    cutter_dofs_per_block[1] = GeometryInfo<dim - 1>::vertices_per_cell;
    // Mapping to transform the cut points to isoparametric space
    MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
    MappingQGeneric<dim - 1, dim> cutter_mapping(1);
    // Collect the cv inlet and outlet fluid cells
    double inlet_x = this->control_volume_boundaries[0];
    double outlet_x = this->control_volume_boundaries[1];
    auto is_left_to = [](double cell_x, double cv_x) { return cell_x <= cv_x; };
    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }
        // Initialize the cutter on this cell
        auto cutter = surface_cutters.get_data(f_cell);
        bool has_left_vertex_for_inlet = false;
        bool has_right_vertex_for_inlet = false;
        bool has_left_vertex_for_outlet = false;
        bool has_right_vertex_for_outlet = false;
        for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            double cell_x = f_cell->vertex(v)[0];
            if (is_left_to(cell_x, inlet_x))
              {
                has_left_vertex_for_inlet = true;
              }
            else
              {
                has_right_vertex_for_inlet = true;
              }
            if (is_left_to(cell_x, outlet_x))
              {
                has_left_vertex_for_outlet = true;
              }
            else
              {
                has_right_vertex_for_outlet = true;
              }
          }
        std::vector<Point<dim>> cut_points;
        if (has_left_vertex_for_inlet && has_right_vertex_for_inlet)
          {
            // Compute the end points for the cutter
            cut_points = ::compute_cut_points<dim>(f_cell, inlet_x);
            // Compute the volume fraction of the cut for volume integral
            // computation
            cutter[0]->volume_fraction =
              ::compute_volume_fraction(f_cell, cut_points, "inlet");
            this->inlet_cells.insert(f_cell);
          }
        else if (has_left_vertex_for_outlet && has_right_vertex_for_outlet)
          {
            // Compute the end points for the cutter
            cut_points = ::compute_cut_points<dim>(f_cell, outlet_x);
            // Compute the volume fraction of the cut for volume integral
            // computation
            cutter[0]->volume_fraction =
              ::compute_volume_fraction(f_cell, cut_points, "outlet");
            this->outlet_cells.insert(f_cell);
          }
        if (!cut_points.empty())
          {
            CellData<dim - 1> cut_cell;
            for (unsigned i = 0; i < GeometryInfo<dim - 1>::vertices_per_cell;
                 ++i)
              {
                cut_cell.vertices[i] = i;
              }

            // FIXME: In 3D the cut plane might be a triangle which is not
            // supported by deal.II. Therefore, this part only works for cases
            // where all the cuts are quadrilateral in 3D.
            // FIXME: There could be corner cases where the cut only has 1 point
            // in 2D or 2 points in 3D, where there shouldn't be any cut cell,
            // and the volume fraction should be 0 or 1.
            cutter[0]->tria.create_triangulation(
              cut_points, {cut_cell}, SubCellData());
            cutter[0]->dof_handler.initialize(cutter[0]->tria, cutter[0]->fe);

            // Get quadrature points for the cutter
            const std::vector<Point<dim - 1>> &unit_cutter_quadrature_points =
              cutter[0]->quad_formula.get_points();
            std::vector<Point<dim>> cutter_quadrature_points(
              unit_cutter_quadrature_points.size());
            for (unsigned i = 0; i < cutter_quadrature_points.size(); ++i)
              {
                cutter_quadrature_points[i] =
                  cutter_mapping.transform_unit_to_real_cell(
                    cutter[0]->tria.begin_active(),
                    unit_cutter_quadrature_points[i]);
              }

            // Compute the interpolate points in the isoparametric space
            std::vector<Point<dim>> unit_cut_points(
              cutter_quadrature_points.size());
            for (unsigned i = 0; i < unit_cut_points.size(); ++i)
              {
                unit_cut_points[i] = mapping.transform_real_to_unit_cell(
                  f_cell, cutter_quadrature_points[i]);
              }
            cutter[0]->interpolate_q.initialize(
              unit_cut_points, cutter[0]->quad_formula.get_weights());
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::control_volume_analysis()
  {
    cv_values.reset();
    compute_efflux();
    compute_volume_integral();
    compute_interface_integral();
    get_separation_point();
    // Output results to file
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        cv_values.output << std::scientific << time.current() << ","
                         << cv_values.inlet_volume_flow << ","
                         << cv_values.inlet_pressure << ","
                         << cv_values.outlet_volume_flow << ","
                         << cv_values.outlet_pressure << ","
                         << cv_values.energy.rate_kinetic_energy << ","
                         << cv_values.momentum.VF_drag << std::endl;
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::get_separation_point()
  {
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_efflux()
  {
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> vel(GeometryInfo<dim - 1>::vertices_per_cell);
    std::vector<double> pre(GeometryInfo<dim - 1>::vertices_per_cell);

    MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
    MappingQGeneric<dim - 1, dim> cutter_mapping(1);

    // Quantities to be integrated
    auto int_x_velocity = [&vel](int q) { return vel[q][0]; };
    auto int_pressure = [&pre](int q) { return pre[q]; };
    auto int_pressure_work = [&vel, &pre](int q) { return pre[q] * vel[q][0]; };

    auto compute_efflux_internal =
      [&](typename DoFHandler<dim>::active_cell_iterator f_cell) mutable {
        double volume_flow = 0.0;
        double integrated_pressure = 0.0;
        double rate_pressure_work = 0.0;
        // We don't check localness here because they are local already
        auto cutter = surface_cutters.get_data(f_cell);
        FEValues<dim> dummy_fe_values(mapping,
                                      fluid_solver.fe,
                                      cutter[0]->interpolate_q,
                                      update_quadrature_points | update_values |
                                        update_gradients);
        FEValues<dim - 1, dim> cutter_fe_values(cutter_mapping,
                                                cutter[0]->fe,
                                                cutter[0]->quad_formula,
                                                update_JxW_values);
        dummy_fe_values.reinit(f_cell);
        cutter_fe_values.reinit(cutter[0]->dof_handler.begin_active());

        // Get the solution values on the cutter points
        dummy_fe_values[velocities].get_function_values(
          fluid_solver.present_solution, vel);
        dummy_fe_values[pressure].get_function_values(
          fluid_solver.present_solution, pre);
        // Integrate the fluxes on the cutter
        auto integrate = [&cutter, &vel, &pre, &cutter_fe_values](
                           const std::function<double(int)> &quant) {
          double results = 0;
          for (unsigned q = 0; q < cutter[0]->interpolate_q.size(); ++q)
            {
              results += quant(q) * cutter_fe_values.JxW(q);
            }
          return results;
        };
        volume_flow += integrate(int_x_velocity);
        integrated_pressure += integrate(int_pressure);
        rate_pressure_work += integrate(int_pressure_work);
        return std::make_tuple(
          volume_flow, integrated_pressure, rate_pressure_work);
      };
    for (auto f_cell : this->inlet_cells)
      {
        auto increases = compute_efflux_internal(f_cell);
        cv_values.inlet_volume_flow += std::get<0>(increases);
        cv_values.inlet_pressure += std::get<1>(increases);
        cv_values.energy.inlet_pressure_work += std::get<2>(increases);
      }
    for (auto f_cell : this->outlet_cells)
      {
        auto increases = compute_efflux_internal(f_cell);
        cv_values.outlet_volume_flow += std::get<0>(increases);
        cv_values.outlet_pressure += std::get<1>(increases);
        cv_values.energy.outlet_pressure_work += std::get<2>(increases);
      }
    // Sum the results over all MPI ranks
    cv_values.inlet_volume_flow =
      Utilities::MPI::sum(cv_values.inlet_volume_flow, this->mpi_communicator);
    cv_values.outlet_volume_flow =
      Utilities::MPI::sum(cv_values.outlet_volume_flow, this->mpi_communicator);
    cv_values.inlet_pressure =
      Utilities::MPI::sum(cv_values.inlet_pressure, this->mpi_communicator);
    cv_values.outlet_pressure =
      Utilities::MPI::sum(cv_values.outlet_pressure, this->mpi_communicator);
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_volume_integral()
  {
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    std::vector<Tensor<1, dim>> present_vel(fe_values.n_quadrature_points);
    std::vector<double> present_pre(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> previous_vel(fe_values.n_quadrature_points);
    std::vector<double> previous_pre(fe_values.n_quadrature_points);

    // Quantities to be integrated
    auto int_rate_kinetic_energy = [&present_vel,
                                    &previous_vel,
                                    rho = parameters.fluid_rho,
                                    dt = time.get_delta_t()](int q) {
      return 0.5 * rho *
             (present_vel[q].norm_square() - previous_vel[q].norm_square()) /
             dt;
    };

    // The integrate function
    auto integrate = [&fe_values](const std::function<double(int)> &quant) {
      double results = 0;
      for (unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          results += quant(q) * fe_values.JxW(q);
        }
      return results;
    };

    // Internal volume integral function
    auto compute_volume_integral_internal =
      [&](typename DoFHandler<dim>::active_cell_iterator f_cell,
          double volume_fraction = 1) mutable {
        fe_values.reinit(f_cell);

        fe_values[velocities].get_function_values(fluid_solver.present_solution,
                                                  present_vel);
        fe_values[velocities].get_function_values(fluid_previous_solution,
                                                  previous_vel);
        fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                present_pre);
        fe_values[pressure].get_function_values(fluid_previous_solution,
                                                previous_pre);

        // Integrate the quantities on the cells
        cv_values.energy.rate_kinetic_energy +=
          integrate(int_rate_kinetic_energy) * volume_fraction;
      };

    for (auto sections : cv_f_cells)
      {
        for (auto cv_item : sections.second)
          {
            auto f_cell = cv_item.second;
            // Skip artificial fluid
            auto fsi_data = fluid_solver.cell_property.get_data(f_cell);
            if (fsi_data[0]->indicator != 0)
              {
                continue;
              }
            compute_volume_integral_internal(f_cell, 1);
          }
      }
    for (auto f_cell : inlet_cells)
      {
        // Skip artificial elements
        auto fsi_data = fluid_solver.cell_property.get_data(f_cell);
        if (fsi_data[0]->indicator != 0)
          {
            continue;
          }
        auto cutter = surface_cutters.get_data(f_cell);
        compute_volume_integral_internal(f_cell, cutter[0]->volume_fraction);
      }
    for (auto f_cell : outlet_cells)
      {
        // Skip artificial elements
        auto fsi_data = fluid_solver.cell_property.get_data(f_cell);
        if (fsi_data[0]->indicator != 0)
          {
            continue;
          }
        auto cutter = surface_cutters.get_data(f_cell);
        compute_volume_integral_internal(f_cell, cutter[0]->volume_fraction);
      }
    // Sum the results over all MPI ranks
    cv_values.energy.rate_kinetic_energy = Utilities::MPI::sum(
      cv_values.energy.rate_kinetic_energy, this->mpi_communicator);
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_interface_integral()
  {
    cv_values.momentum.VF_drag = 0;

    FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                     solid_solver.face_quad_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = solid_solver.fe.dofs_per_cell;
    const unsigned int n_f_q_points = solid_solver.face_quad_formula.size();

    Vector<double> localized_displacement(solid_solver.current_displacement);

    std::vector<std::vector<Tensor<1, dim>>> fsi_stress_rows_values(dim);
    for (unsigned int d = 0; d < dim; ++d)
      {
        fsi_stress_rows_values[d].resize(n_f_q_points);
      }
    // A "viewer" to describe the nodal dofs as a vector.
    FEValuesExtractors::Vector displacements(0);

    for (auto cell = solid_solver.dof_handler.begin_active();
         cell != solid_solver.dof_handler.end();
         ++cell)
      {
        if (cell->subdomain_id() != solid_solver.this_mpi_process)
          {
            continue;
          }
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary())
              {
                if (!cell->face(face)->at_boundary())
                  {
                    continue;
                  }

                // Get FSI stress values on face quadrature points
                std::vector<Tensor<2, dim>> fsi_stress(n_f_q_points);
                std::vector<Point<dim>> vertex_displacement(
                  GeometryInfo<dim>::vertices_per_face);
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_face;
                     ++v)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        vertex_displacement[v][d] = localized_displacement(
                          cell->face(face)->vertex_dof_index(v, d));
                      }
                    cell->face(face)->vertex(v) += vertex_displacement[v];
                  }
                fe_face_values.reinit(cell, face);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    fe_face_values[displacements].get_function_values(
                      solid_solver.fsi_stress_rows[d],
                      fsi_stress_rows_values[d]);
                  }
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_face;
                     ++v)
                  {
                    cell->face(face)->vertex(v) -= vertex_displacement[v];
                  }
                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                      {
                        for (unsigned int d2 = 0; d2 < dim; ++d2)
                          {
                            fsi_stress[q][d1][d2] =
                              fsi_stress_rows_values[d1][q][d2];
                          }
                      }
                  } // End looping face quadrature points

                Tensor<1, dim> drag;
                Tensor<1, dim> friction;

                Tensor<2, dim> normal_stress;
                Tensor<2, dim> tangential_stress;

                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    normal_stress = fsi_stress[q];
                    tangential_stress = fsi_stress[q];
                    for (unsigned int i = 0; i < dim; ++i)
                      {
                        for (unsigned int j = 0; j < dim; ++j)
                          {
                            if (i == j)
                              tangential_stress[i][j] = 0;
                            else
                              normal_stress[i][j] = 0;
                          }
                      }

                    drag = normal_stress * fe_face_values.normal_vector(q);
                    friction =
                      tangential_stress * fe_face_values.normal_vector(q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const unsigned int component_j =
                          solid_solver.fe.system_to_component_index(j).first;
                        cv_values.momentum.VF_drag +=
                          fe_face_values.shape_value(j, q) * drag[component_j] *
                          fe_face_values.JxW(q);
                      }
                  }
              }
          }
      }
    // Sum the results over all MPI ranks
    cv_values.momentum.VF_drag =
      Utilities::MPI::sum(cv_values.momentum.VF_drag, this->mpi_communicator);
  }

  template <int dim>
  void
  ControlVolumeFSI<dim>::CVValues::initialize_output(MPI_Comm &mpi_communicator)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        output.open("./control_volume_analysis.csv",
                    std::ios::in | std::ios::out | std::ios::app);
        output.precision(6);
        output << "Time,"
               << "Inlet volume flow,"
               << "Inlet pressure,"
               << "Outlet volume flow,"
               << "Outlet pressure,"
               << "Rate kinetic energy,"
               << "VF drag" << std::endl;
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::CVValues::reset()
  {
    // General terms
    inlet_volume_flow = 0;
    outlet_volume_flow = 0;
    inlet_pressure = 0;
    outlet_pressure = 0;
    // Momentum equation terms
    momentum.inlet_efflux = 0;
    momentum.outlet_efflux = 0;
    momentum.VF_drag = 0;
    // Energy equation terms
    energy.inlet_pressure_work = 0;
    energy.outlet_pressure_work = 0;
    energy.rate_kinetic_energy = 0;
    energy.rate_dissipation = 0;
  }

  template <int dim>
  ControlVolumeFSI<dim>::SurfaceCutter::SurfaceCutter()
    : fe(FE_Q<dim - 1, dim>(1), dim, FE_Q<dim - 1, dim>(1), 1),
      quad_formula(2),
      volume_fraction(1)
  {
  }

  template class ControlVolumeFSI<2>;
  template class ControlVolumeFSI<3>;
} // namespace MPI

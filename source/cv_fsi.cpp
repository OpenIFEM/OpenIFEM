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
            if (std::abs(p1[0] - p2[0]) > std::abs(p1[0]) * 1e-10)
              {
                cut_point[d] =
                  p1[d] + (p2[d] - p1[d]) * (x - p1[0]) / (p2[0] - p1[0]);
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
    : FSI<dim>(f, s, p, use_dirichlet_bc), output_solid_boundary(false)
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
        fluid_previous_solution.reinit(fluid_solver.owned_partitioning,
                                       fluid_solver.relevant_partitioning,
                                       mpi_communicator);
        fluid_previous_solution = fluid_solver.present_solution;
      }

    collect_solid_boundaries();
    setup_cell_hints();
    update_vertices_mask();
    collect_inlet_outlet_cells();
    collect_control_volume_cells();
    if (this->output_solid_boundary)
      {
        collect_solid_boundary_vertices();
      }
    cv_values.initialize_output(this->time, this->mpi_communicator);

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
          if (penetration_criterion)
            {
              apply_contact_model(first_step);
            }
          else
            {
              solid_solver.run_one_step(first_step);
            }
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
  void ControlVolumeFSI<dim>::set_output_solid_boundary(const bool output)
  {
    this->output_solid_boundary = output;
  }

  template <int dim>
  void ControlVolumeFSI<dim>::collect_control_volume_cells()
  {
    AssertThrow(dim == 2, ExcNotImplemented());
    auto f_cell = fluid_solver.dof_handler.begin_active();
    auto f_scalar_cell = fluid_solver.scalar_dof_handler.begin_active();
    for (; f_cell != fluid_solver.dof_handler.end(); ++f_cell, ++f_scalar_cell)
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
            // Collect streamline path cells for Bernoulli analysis
            // For now (enforeced symmetry), use upper boundary of CV
            if (f_cell->at_boundary() &&
                std::fabs(f_cell->center()[1] - control_volume_boundaries[3]) <
                  f_cell->diameter())
              {
                streamline_path_cells.emplace(f_cell->center()[0],
                                              std::pair(f_cell, f_scalar_cell));
              }
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

    auto f_cell = fluid_solver.dof_handler.begin_active();
    auto f_scalar_cell = fluid_solver.scalar_dof_handler.begin_active();
    for (; f_cell != fluid_solver.dof_handler.end(); ++f_cell, ++f_scalar_cell)
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
            // Identify the start point for contraction region in Bernoulli
            // analysis
            if (f_cell->at_boundary() &&
                std::fabs(f_cell->center()[1] - control_volume_boundaries[3]) <
                  f_cell->diameter())
              {
                for (unsigned int face_n = 0;
                     face_n < GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                  {
                    if (f_cell->at_boundary(face_n))
                      {
                        if (dim == 2)
                          {
                            auto v0 = f_cell->face(face_n)->vertex(0);
                            auto v1 = f_cell->face(face_n)->vertex(1);
                            double left = std::min({v0[0], v1[0]});
                            double right = std::max({v0[0], v1[0]});
                            double fraction =
                              (right - control_volume_boundaries[0]) /
                              (right - left);
                            bernoulli_start_end.first =
                              std::make_tuple(f_cell, f_scalar_cell, fraction);
                          }
                      }
                  }
              }
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
            // Identify the end point for jet region in Bernoulli analysis
            if (f_cell->at_boundary() &&
                std::fabs(f_cell->center()[1] - control_volume_boundaries[3]) <
                  f_cell->diameter())
              {
                for (unsigned int face_n = 0;
                     face_n < GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                  {
                    if (f_cell->at_boundary(face_n))
                      {
                        if (dim == 2)
                          {
                            auto v0 = f_cell->face(face_n)->vertex(0);
                            auto v1 = f_cell->face(face_n)->vertex(1);
                            double left = std::min({v0[0], v1[0]});
                            double right = std::max({v0[0], v1[0]});
                            double fraction =
                              (control_volume_boundaries[1] - left) /
                              (right - left);
                            bernoulli_start_end.second =
                              std::make_tuple(f_cell, f_scalar_cell, fraction);
                          }
                      }
                  }
              }
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
  void ControlVolumeFSI<dim>::collect_solid_boundary_vertices()
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
      {
        return;
      }
    unsigned fixed_bc_flag = (1 << dim) - 1;

    for (auto &face : solid_solver.triangulation.active_face_iterators())
      {
        if (face->at_boundary())
          {
            // Check if the boundary is fixed
            auto bc = parameters.solid_dirichlet_bcs.find(face->boundary_id());
            if (bc != parameters.solid_dirichlet_bcs.end() &&
                bc->second == fixed_bc_flag)
              {
                // Skip those fixed vertices
                continue;
              }

            for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
              {
                solid_boundary_vertices.insert(face->vertex_iterator(v));
              }
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::control_volume_analysis()
  {
    cv_values.reset();
    compute_flux();
    compute_volume_integral();
    compute_interface_integral();
    compute_bernoulli_terms();
    get_separation_point();
    cv_values.reduce(this->mpi_communicator, time.get_delta_t());
    // Output results to file
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        cv_values.output << std::scientific << time.current() << ","
                         << cv_values.inlet_volume_flow << ","
                         << cv_values.outlet_volume_flow << ","
                         << cv_values.inlet_pressure_force << ","
                         << cv_values.outlet_pressure_force << ","
                         << cv_values.VF_volume << ","
                         << cv_values.bernoulli.rate_convection_contraction
                         << "," << cv_values.bernoulli.rate_convection_jet
                         << ","
                         << cv_values.bernoulli.rate_pressure_grad_contraction
                         << "," << cv_values.bernoulli.rate_pressure_grad_jet
                         << "," << cv_values.bernoulli.acceleration_contraction
                         << "," << cv_values.bernoulli.acceleration_jet << ","
                         << cv_values.bernoulli.rate_density_contraction << ","
                         << cv_values.bernoulli.rate_density_jet << ","
                         << cv_values.bernoulli.rate_friction_contraction << ","
                         << cv_values.bernoulli.rate_friction_jet << ","
                         << cv_values.momentum.inlet_flux << ","
                         << cv_values.momentum.outlet_flux << ","
                         << cv_values.momentum.rate_momentum << ","
                         << cv_values.momentum.VF_drag << ","
                         << cv_values.momentum.VF_friction << ","
                         << cv_values.energy.inlet_pressure_work << ","
                         << cv_values.energy.outlet_pressure_work << ","
                         << cv_values.energy.inlet_flux << ","
                         << cv_values.energy.outlet_flux << ","
                         << cv_values.energy.convective_KE << ","
                         << cv_values.energy.penetrating_KE << ","
                         << cv_values.energy.rate_kinetic_energy << ","
                         << cv_values.energy.rate_kinetic_energy_direct << ","
                         << cv_values.energy.pressure_convection << ","
                         << cv_values.energy.rate_dissipation << ","
                         << cv_values.energy.rate_compression_work << ","
                         << cv_values.energy.rate_friction_work << ","
                         << cv_values.energy.rate_vf_work << ","
                         << cv_values.energy.rate_vf_work_from_solid << ","
                         << cv_values.bernoulli.contraction_end_x << ","
                         << cv_values.bernoulli.jet_start_x << std::endl;
      }
    if (output_solid_boundary)
      {
        output_solid_boundary_vertices();
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::get_separation_point()
  {
    // **IMPORTANT** this method only works for VF simulation with enforeced
    // symmetry
    // Get solid surface velocity
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_flux()
  {
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    std::vector<Tensor<1, dim>> vel(GeometryInfo<dim - 1>::vertices_per_cell);
    std::vector<Tensor<2, dim>> vel_grad(
      GeometryInfo<dim - 1>::vertices_per_cell);
    std::vector<double> pre(GeometryInfo<dim - 1>::vertices_per_cell);

    MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
    MappingQGeneric<dim - 1, dim> cutter_mapping(1);

    // Quantities to be integrated
    auto int_x_velocity = [&vel](int q) { return vel[q][0]; };
    auto int_pressure = [&pre](int q) { return pre[q]; };
    auto int_momentum = [&vel, this](int q) {
      return parameters.fluid_rho * vel[q][0] * vel[q][0];
    };
    auto int_KE = [&vel, this](int q) {
      return 0.5 * parameters.fluid_rho * vel[q][0] * vel[q].norm_square();
    };
    auto int_pressure_work = [&vel, &pre](int q) { return pre[q] * vel[q][0]; };
    auto int_friction_work =
      [&vel, &vel_grad, mu = parameters.viscosity](int q) {
        double retval = 0.0;
        for (unsigned d = 0; d < dim; ++d)
          {
            retval += mu * vel_grad[q][d][0] * vel[q][d];
          }
        return retval;
      };

    auto compute_efflux_internal =
      [&](typename DoFHandler<dim>::active_cell_iterator f_cell) mutable {
        double volume_flow = 0.0;
        double pressure_force = 0.0;
        double momentum_flux = 0.0;
        double KE_flux = 0.0;
        double rate_pressure_work = 0.0;
        double rate_friction_work = 0.0;
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
        dummy_fe_values[velocities].get_function_gradients(
          fluid_solver.present_solution, vel_grad);
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
        pressure_force += integrate(int_pressure);
        momentum_flux += integrate(int_momentum);
        KE_flux += integrate(int_KE);
        rate_pressure_work += integrate(int_pressure_work);
        rate_friction_work += integrate(int_friction_work);
        return std::make_tuple(volume_flow,
                               pressure_force,
                               momentum_flux,
                               KE_flux,
                               rate_pressure_work,
                               rate_friction_work);
      };
    for (auto f_cell : this->inlet_cells)
      {
        auto increases = compute_efflux_internal(f_cell);
        cv_values.inlet_volume_flow += std::get<0>(increases);
        cv_values.inlet_pressure_force += std::get<1>(increases);
        cv_values.momentum.inlet_flux += std::get<2>(increases);
        cv_values.energy.inlet_flux += std::get<3>(increases);
        cv_values.energy.inlet_pressure_work += std::get<4>(increases);
        cv_values.energy.rate_friction_work -= std::get<5>(increases);
      }
    for (auto f_cell : this->outlet_cells)
      {
        auto increases = compute_efflux_internal(f_cell);
        cv_values.outlet_volume_flow += std::get<0>(increases);
        cv_values.outlet_pressure_force += std::get<1>(increases);
        cv_values.momentum.outlet_flux += std::get<2>(increases);
        cv_values.energy.outlet_flux += std::get<3>(increases);
        cv_values.energy.outlet_pressure_work += std::get<4>(increases);
        cv_values.energy.rate_friction_work += std::get<5>(increases);
      }
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
    std::vector<Tensor<2, dim>> vel_grad(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> pre_grad(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> previous_vel(fe_values.n_quadrature_points);

    // Quantities to be integrated
    auto int_rate_momentum = [&present_vel,
                              &previous_vel,
                              rho = parameters.fluid_rho,
                              dt = time.get_delta_t()](int q) {
      return rho * (present_vel[q][0] - previous_vel[q][0]) / dt;
    };
    auto int_previous_kinetic_energy = [rho = parameters.fluid_rho,
                                        &previous_vel](int q) {
      return 0.5 * rho * previous_vel[q].norm_square();
    };
    auto int_present_kinetic_energy = [rho = parameters.fluid_rho,
                                       &present_vel](int q) {
      return 0.5 * rho * present_vel[q].norm_square();
    };
    auto int_rate_kinetic_energy = [&rho = parameters.fluid_rho,
                                    dt = time.get_delta_t(),
                                    &previous_vel,
                                    &present_vel](int q) {
      return rho * scalar_product((present_vel[q] - previous_vel[q]) / dt,
                                  present_vel[q]);
    };
    auto int_convective_kinetive_energy =
      [&rho = parameters.fluid_rho, &present_vel, &vel_grad](int q) {
        return rho * vel_grad[q] * present_vel[q] * present_vel[q];
      };
    auto int_pressure_convection = [&present_vel, &pre_grad](int q) {
      double retval = 0.0;
      for (unsigned i = 0; i < dim; ++i)
        {
          retval += pre_grad[q][i] * present_vel[q][i];
        }
      return retval;
    };
    auto int_rate_dissipation = [&vel_grad, mu = parameters.viscosity](int q) {
      return mu * scalar_product(vel_grad[q], vel_grad[q]);
    };
    auto int_rate_compression = [&present_pre, &vel_grad](int q) {
      double retval = 0.0;
      for (unsigned i = 0; i < dim; ++i)
        {
          retval += present_pre[q] * vel_grad[q][i][i];
        }
      return retval;
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
        fe_values[velocities].get_function_gradients(
          fluid_solver.present_solution, vel_grad);
        fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                present_pre);
        fe_values[pressure].get_function_gradients(
          fluid_solver.present_solution, pre_grad);

        // Integrate the quantities on the cells
        cv_values.momentum.rate_momentum +=
          integrate(int_rate_momentum) * volume_fraction;
        cv_values.energy.previous_KE +=
          integrate(int_previous_kinetic_energy) * volume_fraction;
        cv_values.energy.present_KE +=
          integrate(int_present_kinetic_energy) * volume_fraction;
        cv_values.energy.rate_kinetic_energy_direct +=
          integrate(int_rate_kinetic_energy) * volume_fraction;
        cv_values.energy.convective_KE +=
          integrate(int_convective_kinetive_energy) * volume_fraction;
        cv_values.energy.pressure_convection +=
          integrate(int_pressure_convection) * volume_fraction;
        cv_values.energy.rate_dissipation +=
          integrate(int_rate_dissipation) * volume_fraction;
        cv_values.energy.rate_compression_work +=
          integrate(int_rate_compression) * volume_fraction;
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
    // Carry out the VF volume integral
    move_solid_mesh(true);
    for (auto s_cell : this->solid_solver.dof_handler.active_cell_iterators())
      {
        if (s_cell->subdomain_id() == solid_solver.this_mpi_process)
          {
            cv_values.VF_volume += s_cell->measure();
          }
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_interface_integral()
  {
    FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                     solid_solver.face_quad_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);
    FEFaceValues<dim> scalar_fe_face_values(
      solid_solver.scalar_fe, solid_solver.face_quad_formula, update_values);

    const unsigned int n_f_q_points = solid_solver.face_quad_formula.size();

    Vector<double> localized_displacement(solid_solver.current_displacement);
    Vector<double> localized_velocity(solid_solver.current_velocity);

    std::vector<std::vector<Tensor<1, dim>>> fsi_stress_rows_values(dim);
    std::vector<Tensor<1, dim>> fluid_velocity_values(n_f_q_points);
    std::vector<Tensor<1, dim>> solid_velocity_values(n_f_q_points);
    std::vector<Tensor<1, dim>> relative_velocity_values(n_f_q_points);
    std::vector<double> fluid_pressure_values(n_f_q_points);
    for (unsigned int d = 0; d < dim; ++d)
      {
        fsi_stress_rows_values[d].resize(n_f_q_points);
      }
    // A "viewer" to describe the nodal dofs as a vector.
    FEValuesExtractors::Vector displacements(0);

    for (auto cell = solid_solver.dof_handler.begin_active(),
              scalar_cell = solid_solver.scalar_dof_handler.begin_active();
         cell != solid_solver.dof_handler.end();
         ++cell, ++scalar_cell)
      {
        if (cell->subdomain_id() != solid_solver.this_mpi_process)
          {
            continue;
          }
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary() &&
                parameters.solid_dirichlet_bcs.find(
                  cell->face(face)->boundary_id()) ==
                  parameters.solid_dirichlet_bcs.end())
              {
                if (!cell->face(face)->at_boundary())
                  {
                    continue;
                  }

                // Get FSI stress values on face quadrature points
                std::vector<SymmetricTensor<2, dim>> fsi_stress(n_f_q_points);
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
                scalar_fe_face_values.reinit(scalar_cell, face);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    fe_face_values[displacements].get_function_values(
                      solid_solver.fsi_stress_rows[d],
                      fsi_stress_rows_values[d]);
                  }
                fe_face_values[displacements].get_function_values(
                  solid_solver.fluid_velocity, fluid_velocity_values);
                fe_face_values[displacements].get_function_values(
                  localized_velocity, solid_velocity_values);
                scalar_fe_face_values.get_function_values(
                  solid_solver.fluid_pressure, fluid_pressure_values);

                for (unsigned q = 0; q < n_f_q_points; ++q)
                  {
                    relative_velocity_values[q] =
                      fluid_velocity_values[q] - solid_velocity_values[q];
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
                    fsi_stress[q] +=
                      fluid_pressure_values[q] *
                      Physics::Elasticity::StandardTensors<dim>::I;
                  } // End looping face quadrature points

                // Pressure drag force is the normal component of the traction
                // Friction work is the tangential component of the traction
                // *times* surface fluid velocity
                double drag_force, drag_work = 0.0, drag_work_from_solid = 0.0,
                                   friction_force = 0.0, friction_work = 0.0,
                                   penetrating_KE = 0.0;

                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    // Drag: \int_S_{VF}{p n_1}dS
                    drag_force = fluid_pressure_values[q] *
                                 fe_face_values.normal_vector(q)[0] *
                                 fe_face_values.JxW(q);
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        // Friction (in x): \int_S_{VF}{\tau_{1j}n_j}dS
                        friction_force += fsi_stress[q][0][j] *
                                          fe_face_values.normal_vector(q)[j] *
                                          fe_face_values.JxW(q);
                        // Drag work: \int_S{p u_i n_i}dS
                        drag_work += fluid_pressure_values[q] *
                                     fluid_velocity_values[q][j] *
                                     fe_face_values.normal_vector(q)[j] *
                                     fe_face_values.JxW(q);
                        drag_work_from_solid +=
                          fluid_pressure_values[q] *
                          solid_velocity_values[q][j] *
                          fe_face_values.normal_vector(q)[j] *
                          fe_face_values.JxW(q);
                        // Penetrating KE flux
                        penetrating_KE +=
                          fluid_velocity_values[q].norm_square() *
                          relative_velocity_values[q][j] *
                          fe_face_values.normal_vector(q)[j] *
                          fe_face_values.JxW(q);
                        // Friction work: \int_S_{VF}{\tau_{ij} u_i n_j}dS
                        for (unsigned int i = 0; i < dim; ++i)
                          {
                            friction_work +=
                              fsi_stress[q][i][j] *
                              fluid_velocity_values[q][i] *
                              fe_face_values.normal_vector(q)[j] *
                              fe_face_values.JxW(q);
                          }
                      }
                    cv_values.momentum.VF_drag += drag_force;
                    cv_values.momentum.VF_friction += friction_force;
                    cv_values.energy.rate_vf_work += drag_work;
                    cv_values.energy.rate_vf_work_from_solid +=
                      drag_work_from_solid;
                    cv_values.energy.rate_friction_work += friction_work;
                  }
              }
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::compute_bernoulli_terms()
  {
    /*
     ONLY WORKS FOR HALF SPACE NOW!
    */
    move_solid_mesh(true);
    double centerline_y = control_volume_boundaries[3];
    // Identify the separation point for contraction and jet regions
    double gap_tolerance = 0.0045; // Use 2 layers in glottis for tolerance
    Point<dim> highest_p;
    double highest_y = 0.0;
    // Get the highest y coordinate
    for (auto &s_vert : solid_solver.triangulation.get_vertices())
      {
        if (std::fabs(s_vert[1] - centerline_y) <
            std::fabs(highest_y - centerline_y))
          {
            highest_y = s_vert[1];
            highest_p = s_vert;
          }
      }
    // Collect the highest points and sort them from left to right
    std::vector<Point<dim>> highest_points;
    for (auto &s_vert : solid_solver.triangulation.get_vertices())
      {
        if (std::fabs(s_vert[1] - highest_y) < gap_tolerance)
          {
            highest_points.emplace_back(s_vert);
          }
      }
    std::sort(highest_points.begin(),
              highest_points.end(),
              [](Point<dim> &a, Point<dim> &b) { return a[0] < b[0]; });
    Point<dim> contraction_end_point, jet_start_point;
    // Check if the contraction end point overlaps with jet start point
    if (std::fabs(highest_y - centerline_y) < gap_tolerance)
      {
        // Left most for contraction end point and right most for jet start
        // point
        contraction_end_point = *(highest_points.begin());
        jet_start_point = *(highest_points.rbegin());
      }
    else // They overlap, use the highest point
      {
        contraction_end_point = highest_p;
        jet_start_point = highest_p;
      }
    // Save the x coordinates for output
    cv_values.bernoulli.contraction_end_x = contraction_end_point[0];
    cv_values.bernoulli.jet_start_x = jet_start_point[0];

    // Compute the streamline path integral
    // Need to copy a stress vector for vector query
    // The order for stress grad is: Sxx, Sxy, (Sxz), Syy, (Syz, Szz)
    std::vector<PETScWrappers::MPI::Vector> relevant_partition_stress(
      (dim * dim + dim) / 2,
      PETScWrappers::MPI::Vector(fluid_solver.locally_owned_scalar_dofs,
                                 fluid_solver.locally_relevant_scalar_dofs,
                                 mpi_communicator));
    for (unsigned component = 0; component < relevant_partition_stress.size();
         ++component)
      {
        unsigned int first_index = component / dim;
        relevant_partition_stress[component] =
          fluid_solver.stress[first_index][component - first_index];
      }

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_gradients | update_JxW_values);
    FEValues<dim> scalar_fe_values(fluid_solver.scalar_fe,
                                   fluid_solver.volume_quad_formula,
                                   update_values | update_quadrature_points |
                                     update_gradients | update_JxW_values);
    std::vector<Tensor<1, dim>> present_vel(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> previous_vel(fe_values.n_quadrature_points);
    std::vector<Tensor<2, dim>> vel_grad(fe_values.n_quadrature_points);
    std::vector<double> present_pre(fe_values.n_quadrature_points);
    std::vector<Tensor<1, dim>> pre_grad(fe_values.n_quadrature_points);
    // The order for stress grad is: Sxx, Sxy, (Sxz), Syy, (Syz, Szz)
    std::vector<std::vector<Tensor<1, dim>>> stress_grad(
      (dim * dim + dim) / 2,
      std::vector<Tensor<1, dim>>(scalar_fe_values.n_quadrature_points));

    // Helper functions to compute integrals
    auto get_area_fraction = [](cell_iterator cell) {
      double cell_volume = cell->measure();
      double cell_diamter = 0.0;
      // Presume the cell is rectangular/hex
      for (unsigned face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->at_boundary(face))
            {
              cell_diamter = cell->face(face)->measure();
            }
        }
      return cell_diamter / cell_volume;
    };

    auto int_acceleration_head =
      [dt = time.get_delta_t(), &present_vel, &previous_vel](int q) {
        return (present_vel[q][0] - previous_vel[q][0]) / dt;
      };
    auto int_pressure_head = [&pre_grad, rho = parameters.fluid_rho](int q) {
      return pre_grad[q][0] / rho;
    };
    auto int_convection_head = [&present_vel, &vel_grad](int q) {
      return present_vel[q] * vel_grad[q][0];
    };
    auto int_density_head = [&present_pre,
                             &pre_grad,
                             rho = parameters.fluid_rho](int q) {
      double atm = 1013250;
      return present_pre[q] / rho / (atm + 2 * present_pre[q]) * pre_grad[q][0];
    };
    auto int_friction_head = [&stress_grad,
                              &pre_grad,
                              mu = parameters.viscosity,
                              rho = parameters.fluid_rho](int q) {
      double retval = 0.0;
      // For 2d this is (Sxx_grad[q][0] + Sxy_grad[q][1] - Syy_grad[q][0]) / rho
      // For 3d this is (Sxx_grad[q][0] + Sxy_grad[q][1] + Sxz_grad[q][2] -
      // Syy_grad[q][0] - Szz_grad[q][0]) / rho
      // for Sxx_grad[q][0] Sxy_grad[q][1] (or/and Sxz_grad[q][2])
      for (unsigned d = 0; d < dim; ++d)
        {
          retval += stress_grad[d][q][d] / rho;
        }
      // for Syy_grad[q][0]
      retval -= stress_grad[dim][q][0] / rho;
      // Since Sxx includes pressure gradient we need to subtract it
      if (dim == 3)
        {
          // for Szz_grad[q][0] (if exists).
          retval -= stress_grad[5][q][0] / rho;
          // Only happens in 3d because in 2d Sxx_grad[q][0] and Syy_grad[q][0]
          // cancel out
          retval -= pre_grad[q][0] / rho;
        }
      return retval;
    };
    auto int_friction_work =
      [&present_vel, &vel_grad, mu = parameters.viscosity](int q) {
        double retval = 0.0;
        for (unsigned d = 0; d < dim; ++d)
          {
            retval += mu * vel_grad[q][d][1] * present_vel[q][d];
          }
        return retval;
      };

    auto integrate = [&fe_values](const std::function<double(int)> &quant) {
      double results = 0;
      for (unsigned q = 0; q < fe_values.n_quadrature_points; ++q)
        {
          results += quant(q) * fe_values.JxW(q);
        }
      return results;
    };

    for (auto &cell_pairs : streamline_path_cells)
      {
        auto &cell = cell_pairs.second.first;
        auto &scalar_cell = cell_pairs.second.second;
        /* Check if the cell is in contraction or jet region (or
        neither). Use bit flags:
        For separated contraction end & jet start points
            contraction end    jet start
                  |              |
        |------|------|------|------|------|
          0101   0111   0110   1110   1010
            5      7      6     14     10

        For overlapped contraction end & jet start points
            contraction end
                  |
        |------|------|------|
          0101   1111   1010
            5     15     10
        */
        int region = 0;
        int in_contraction = 1;
        int not_in_contraction = 2;
        int not_in_jet = 4;
        int in_jet = 8;

        for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            Point<dim> vertex = cell->vertex(v);
            region =
              region |
              (vertex[0] <= contraction_end_point[0] ? in_contraction : 0);
            region =
              region |
              (vertex[0] > contraction_end_point[0] ? not_in_contraction : 0);
            region =
              region | (vertex[0] <= jet_start_point[0] ? not_in_jet : 0);
            region = region | (vertex[0] > jet_start_point[0] ? in_jet : 0);
          }
        fe_values.reinit(cell);
        scalar_fe_values.reinit(scalar_cell);

        fe_values[velocities].get_function_values(fluid_solver.present_solution,
                                                  present_vel);
        fe_values[velocities].get_function_values(fluid_previous_solution,
                                                  previous_vel);
        fe_values[velocities].get_function_gradients(fluid_previous_solution,
                                                     vel_grad);
        fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                present_pre);
        fe_values[pressure].get_function_gradients(
          fluid_solver.present_solution, pre_grad);
        for (unsigned component = 0; component < stress_grad.size();
             ++component)
          {
            scalar_fe_values.get_function_gradients(
              relevant_partition_stress[component], stress_grad[component]);
          }

        double convection_head = 0.0;
        double pressure_head = 0.0;
        double acceleration = 0.0;
        double rate_density = 0.0;
        double rate_friction_head = 0.0;
        double rate_friction_work = 0.0;
        // Don't account for those partially in contraction region or jet region
        bool contraction_end =
          (region & in_contraction) && (region & not_in_contraction);
        bool jet_start = (region & in_jet) && (region & not_in_jet);
        if (contraction_end || jet_start)
          {
            continue;
          }

        // Integral for contracion region
        double area_fraction = get_area_fraction(cell);
        if (region != (not_in_contraction | not_in_jet))
          {
            // Compute integral for Bernoulli equation
            convection_head = integrate(int_convection_head);
            pressure_head = integrate(int_pressure_head);
            acceleration = integrate(int_acceleration_head);
            rate_density = integrate(int_density_head);
            rate_friction_head = integrate(int_friction_head);
            rate_friction_work = integrate(int_friction_work);
            // Compute integral for friction work in energy equation
            cv_values.energy.rate_friction_work +=
              rate_friction_work * area_fraction;
          }
        if (region & in_contraction)
          {
            cv_values.bernoulli.rate_convection_contraction +=
              convection_head * area_fraction;
            cv_values.bernoulli.rate_pressure_grad_contraction +=
              pressure_head * area_fraction;
            cv_values.bernoulli.acceleration_contraction +=
              acceleration * area_fraction;
            cv_values.bernoulli.rate_density_contraction +=
              rate_density * area_fraction;
            cv_values.bernoulli.rate_friction_contraction +=
              rate_friction_head * area_fraction;
          }
        // Integral for jet region
        if (region & in_jet)
          {
            cv_values.bernoulli.rate_convection_jet +=
              convection_head * area_fraction;
            cv_values.bernoulli.rate_pressure_grad_jet +=
              pressure_head * area_fraction;
            cv_values.bernoulli.acceleration_jet +=
              acceleration * area_fraction;
            cv_values.bernoulli.rate_density_jet +=
              rate_density * area_fraction;
            cv_values.bernoulli.rate_friction_jet +=
              rate_friction_head * area_fraction;
          }
      }
    // Start point for contraction region and end point for jet region
    auto get_start_end_quantities =
      [&](double &convection_head_int,
          double &pressure_head_int,
          double &acc_int,
          double &density_int,
          double &friction_head_int,
          double &friction_work_int,
          std::tuple<cell_iterator, cell_iterator, double> &cell_tuple) {
        auto &[cell, scalar_cell, fraction] = cell_tuple;

        if (cell.state() == IteratorState::IteratorStates::valid &&
            cell->is_locally_owned())
          {
            // Compute integral

            fe_values.reinit(cell);
            scalar_fe_values.reinit(scalar_cell);

            fe_values[velocities].get_function_values(
              fluid_solver.present_solution, present_vel);
            fe_values[velocities].get_function_values(fluid_previous_solution,
                                                      previous_vel);
            fe_values[velocities].get_function_gradients(
              fluid_previous_solution, vel_grad);
            fe_values[pressure].get_function_values(
              fluid_solver.present_solution, present_pre);
            fe_values[pressure].get_function_gradients(
              fluid_solver.present_solution, pre_grad);
            for (unsigned component = 0; component < stress_grad.size();
                 ++component)
              {
                scalar_fe_values.get_function_gradients(
                  relevant_partition_stress[component], stress_grad[component]);
              }

            double area_fraction = fraction * get_area_fraction(cell);
            convection_head_int +=
              integrate(int_convection_head) * area_fraction;
            pressure_head_int += integrate(int_pressure_head) * area_fraction;
            acc_int += integrate(int_acceleration_head) * area_fraction;
            density_int += integrate(int_density_head) * area_fraction;
            friction_head_int += integrate(int_friction_head) * area_fraction;
            friction_work_int += integrate(int_friction_work) * area_fraction;
          }
      };
    get_start_end_quantities(cv_values.bernoulli.rate_convection_contraction,
                             cv_values.bernoulli.rate_pressure_grad_contraction,
                             cv_values.bernoulli.acceleration_contraction,
                             cv_values.bernoulli.rate_density_contraction,
                             cv_values.bernoulli.rate_friction_contraction,
                             cv_values.energy.rate_friction_work,
                             bernoulli_start_end.first);
    get_start_end_quantities(cv_values.bernoulli.rate_convection_jet,
                             cv_values.bernoulli.rate_pressure_grad_jet,
                             cv_values.bernoulli.acceleration_jet,
                             cv_values.bernoulli.rate_density_jet,
                             cv_values.bernoulli.rate_friction_jet,
                             cv_values.energy.rate_friction_work,
                             bernoulli_start_end.second);

    move_solid_mesh(false);
  }

  template <int dim>
  void ControlVolumeFSI<dim>::output_solid_boundary_vertices()
  {
    move_solid_mesh(true);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        fs::create_directory("solid_trace");
        std::string filename("./solid_trace/BoundaryTrace-" +
                             Utilities::int_to_string(time.get_timestep(), 6));
        std::ofstream output_file(filename, std::ios::out);

        for (const auto &v : solid_boundary_vertices)
          {
            output_file << v->index() << " " << v->vertex(0) << std::endl;
          }

        output_file.close();
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void
  ControlVolumeFSI<dim>::CVValues::initialize_output(const Utils::Time &time,
                                                     MPI_Comm &mpi_communicator)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        // Need to make sure there is a file instance
        std::fstream tmp_file("./control_volume_analysis.csv",
                              std::ios::out | std::ios::app);
        tmp_file.close();
        output.open("./control_volume_analysis.csv",
                    std::ios::in | std::ios::out);
        output.precision(6);
        // If we start from beginning
        if (time.get_timestep() == 0)
          {
            output << "Time,"
                   << "Inlet volume flow,"
                   << "Outlet volume flow,"
                   << "Inlet pressure force,"
                   << "Outlet pressure force,"
                   << "VF volume,"
                   << "Rate convection contraction,"
                   << "Rate convection jet,"
                   << "Rate pressure contraction,"
                   << "Rate pressure jet,"
                   << "Acceleration contraction,"
                   << "Acceleration jet,"
                   << "Rate density contraction,"
                   << "Rate density jet,"
                   << "Rate friction contraction,"
                   << "Rate friction jet,"
                   << "Inlet momentum flux,"
                   << "outlet momentum flux,"
                   << "Momentum change rate,"
                   << "VF drag,"
                   << "Friction force,"
                   << "Inlet pressure work,"
                   << "Outlet pressure work,"
                   << "Inlet KE flux,"
                   << "Outlet KE flux,"
                   << "Convective KE,"
                   << "Penetrating KE,"
                   << "Rate KE,"
                   << "Rate KE direct,"
                   << "Pressure convection,"
                   << "Rate dissipation,"
                   << "Rate compression work,"
                   << "Rate friction work,"
                   << "Rate VF work,"
                   << "Rate VF work from solid,"
                   << "Contraction end xcoord,"
                   << "Jet start xcoord" << std::endl;
          }
        else // Start from a checkpoint
          {
            output.seekg(std::ios::beg);
            // Ignore the first line (labels)
            output.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            double read_time = 0.0;
            while (time.current() - read_time > 1e-6 * time.get_delta_t())
              {
                output >> read_time;
                output.ignore(std::numeric_limits<std::streamsize>::max(),
                              '\n');
              }
            output.seekp(output.tellg());
          }
      }
  }

  template <int dim>
  void ControlVolumeFSI<dim>::CVValues::reduce(MPI_Comm &mpi_communicator,
                                               double delta_t)
  {
    auto reduce_internal = [&](double &quantity) {
      quantity = Utilities::MPI::sum(quantity, mpi_communicator);
    };
    // Sum the results over all MPI ranks
    // Bernoulli terms
    reduce_internal(bernoulli.rate_convection_contraction);
    reduce_internal(bernoulli.rate_convection_jet);
    reduce_internal(bernoulli.rate_pressure_grad_contraction);
    reduce_internal(bernoulli.rate_pressure_grad_jet);
    reduce_internal(bernoulli.acceleration_contraction);
    reduce_internal(bernoulli.acceleration_jet);
    reduce_internal(bernoulli.rate_density_contraction);
    reduce_internal(bernoulli.rate_density_jet);
    reduce_internal(bernoulli.rate_friction_contraction);
    reduce_internal(bernoulli.rate_friction_jet);
    // fluxes
    reduce_internal(inlet_volume_flow);
    reduce_internal(outlet_volume_flow);
    reduce_internal(inlet_pressure_force);
    reduce_internal(outlet_pressure_force);
    reduce_internal(momentum.inlet_flux);
    reduce_internal(momentum.outlet_flux);
    reduce_internal(energy.inlet_pressure_work);
    reduce_internal(energy.outlet_pressure_work);
    reduce_internal(energy.inlet_flux);
    reduce_internal(energy.outlet_flux);
    // volume integrals
    reduce_internal(VF_volume);
    reduce_internal(momentum.rate_momentum);
    reduce_internal(energy.previous_KE);
    reduce_internal(energy.present_KE);
    reduce_internal(energy.convective_KE);
    reduce_internal(energy.rate_kinetic_energy_direct);
    reduce_internal(energy.pressure_convection);
    reduce_internal(energy.rate_dissipation);
    reduce_internal(energy.rate_compression_work);
    // KE rate from volume integral
    energy.rate_kinetic_energy =
      (energy.present_KE - energy.previous_KE) / delta_t;
    // surface integrals;
    reduce_internal(momentum.VF_drag);
    reduce_internal(momentum.VF_friction);
    reduce_internal(energy.penetrating_KE);
    reduce_internal(energy.rate_friction_work);
    reduce_internal(energy.rate_vf_work);
    reduce_internal(energy.rate_vf_work_from_solid);
  }

  template <int dim>
  void ControlVolumeFSI<dim>::CVValues::reset()
  {
    // General terms
    inlet_volume_flow = 0;
    outlet_volume_flow = 0;
    inlet_pressure_force = 0;
    outlet_pressure_force = 0;
    VF_volume = 0;
    // Bernoulli terms
    bernoulli.rate_convection_contraction = 0;
    bernoulli.rate_convection_jet = 0;
    bernoulli.rate_pressure_grad_contraction = 0;
    bernoulli.rate_pressure_grad_jet = 0;
    bernoulli.acceleration_contraction = 0;
    bernoulli.acceleration_jet = 0;
    bernoulli.rate_density_contraction = 0;
    bernoulli.rate_density_jet = 0;
    bernoulli.rate_friction_contraction = 0;
    bernoulli.rate_friction_jet = 0;
    bernoulli.contraction_end_x = 0;
    bernoulli.jet_start_x = 0;
    // Momentum equation terms
    momentum.inlet_flux = 0;
    momentum.outlet_flux = 0;
    momentum.rate_momentum = 0;
    momentum.VF_drag = 0;
    momentum.VF_friction = 0;
    // Energy equation terms
    energy.inlet_pressure_work = 0;
    energy.outlet_pressure_work = 0;
    energy.inlet_flux = 0;
    energy.outlet_flux = 0;
    energy.convective_KE = 0;
    energy.penetrating_KE = 0;
    energy.previous_KE = 0;
    energy.present_KE = 0;
    energy.pressure_convection = 0;
    energy.rate_kinetic_energy = 0;
    energy.rate_kinetic_energy_direct = 0;
    energy.rate_dissipation = 0;
    energy.rate_compression_work = 0;
    energy.rate_friction_work = 0;
    energy.rate_vf_work = 0;
    energy.rate_vf_work_from_solid = 0;
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

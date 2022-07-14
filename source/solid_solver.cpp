#include "solid_solver.h"

namespace Solid
{
  using namespace dealii;

  template <int dim, int spacedim>
  SolidSolver<dim, spacedim>::SolidSolver(
    Triangulation<dim, spacedim> &tria,
    const Parameters::AllParameters &parameters)
    : triangulation(tria),
      parameters(parameters),
      dof_handler(triangulation),
      scalar_dof_handler(triangulation),
      fe(FE_Q<dim, spacedim>(parameters.solid_degree), spacedim),
      scalar_fe(parameters.solid_degree),
      volume_quad_formula(parameters.solid_degree + 1),
      face_quad_formula(parameters.solid_degree + 1),
      is_lag_penalty_explicit(true),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval,
           parameters.save_interval),
      timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
  {
  }

  template <int dim, int spacedim>
  SolidSolver<dim, spacedim>::~SolidSolver()
  {
    scalar_dof_handler.clear();
    dof_handler.clear();
    timer.print_summary();
  }

  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::setup_dofs()
  {
    TimerOutput::Scope timer_section(timer, "Setup system");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);
    scalar_dof_handler.distribute_dofs(scalar_fe);

    // The Dirichlet boundary conditions are stored in the
    // AffineConstraints<double> object. It does not need to modify the sparse
    // matrix after assembly, because it is applied in the assembly process,
    // therefore is better compared with apply_boundary_values approach.
    // Note that ZeroFunction is used here for convenience. In more
    // complicated applications, write a BoundaryValue class to replace it.

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // Homogeneous BC only!
    for (auto itr = parameters.solid_dirichlet_bcs.begin();
         itr != parameters.solid_dirichlet_bcs.end();
         ++itr)
      {
        unsigned int id = itr->first;
        unsigned int flag = itr->second;
        std::vector<bool> mask(spacedim, false);
        // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
        if (flag == 1 || flag == 3 || flag == 5 || flag == 7)
          {
            mask[0] = true;
          }
        if (flag == 2 || flag == 3 || flag == 6 || flag == 7)
          {
            mask[1] = true;
          }
        if (flag == 4 || flag == 5 || flag == 6 || flag == 7)
          {
            mask[2] = true;
          }
        VectorTools::interpolate_boundary_values(
          dof_handler,
          id,
          Functions::ZeroFunction<spacedim>(spacedim),
          constraints,
          ComponentMask(mask));
      }

    // compute bc map input from user-specified points and directions

    std::vector<Point<dim>> points = point_boundary_values.first;
    std::vector<unsigned int> directions = point_boundary_values.second;

    if (!points.empty() && !directions.empty())
      {

        AssertThrow(points.size() == directions.size(),
                    ExcMessage("Number of points and direcions must match!"));

        for (unsigned int i = 0; i < point_boundary_values.first.size(); i++)
          {

            bool find_point = false;

            std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

            for (auto cell = dof_handler.begin_active();
                 cell != dof_handler.end();
                 ++cell)
              {

                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     ++v)
                  {

                    if (!vertex_touched[cell->vertex_index(v)])
                      {
                        vertex_touched[cell->vertex_index(v)] = true;
                        if (abs(cell->vertex(v)(0) - points[i](0)) < 1e-8 &&
                            abs(cell->vertex(v)(1) - points[i](1)) < 1e-8)
                          {
                            find_point = true;
                            unsigned int d = directions[i];
                            assert(d < dim);
                            unsigned int dof_index =
                              cell->vertex_dof_index(v, d);
                            constraints.add_line(dof_index);
                          }
                      }
                  }
              }
            AssertThrow(find_point == true,
                        ExcMessage("Did not find the specified point!"));
          }
      }
    constraints.close();
  }

  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::initialize_system()
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    pattern.copy_from(dsp);

    system_matrix.reinit(pattern);
    system_matrix_updated.reinit(pattern);
    mass_matrix.reinit(pattern);
    mass_matrix_updated.reinit(pattern);
    stiffness_matrix.reinit(pattern);
    damping_matrix.reinit(pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());
    nodal_mass.reinit(dof_handler.n_dofs());
    nodal_forces_traction.reinit(dof_handler.n_dofs());
    nodal_forces_penalty.reinit(dof_handler.n_dofs());
    added_mass_effect.reinit(dof_handler.n_dofs());
    fsi_vel_diff_lag.reinit(dof_handler.n_dofs());

    // Add initial velocity
    if (time.current() == 0.0)
      {
        const std::vector<Point<dim>> &unit_points =
          fe.get_unit_support_points();
        std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
        std::vector<unsigned int> dof_touched(dof_handler.n_dofs(), 0);
        for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
             ++cell)
          {
            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < unit_points.size(); ++i)
              {
                if (dof_touched[dof_indices[i]] == 0)
                  {
                    dof_touched[dof_indices[i]] = 1;
                    auto component_index =
                      fe.system_to_component_index(i).first;
                    auto line = dof_indices[i];
                    previous_velocity[line] =
                      parameters.initial_velocity[component_index];
                  }
              }
          }
        constraints.distribute(previous_velocity);
        current_velocity = previous_velocity;
      }

    strain = std::vector<std::vector<Vector<double>>>(
      spacedim,
      std::vector<Vector<double>>(spacedim,
                                  Vector<double>(scalar_dof_handler.n_dofs())));
    stress = std::vector<std::vector<Vector<double>>>(
      spacedim,
      std::vector<Vector<double>>(spacedim,
                                  Vector<double>(scalar_dof_handler.n_dofs())));

    // Set up cell property, which contains the FSI traction required in FSI
    // simulation
    cell_property.initialize(triangulation.begin_active(),
                             triangulation.end(),
                             GeometryInfo<dim>::faces_per_cell);
  }

  // store user input points and directions
  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::constrain_points(
    const std::vector<Point<dim>> &points,
    const std::vector<unsigned int> &directions)
  {
    point_boundary_values.first = points;
    point_boundary_values.second = directions;
  }

  // Solve linear system \f$Ax = b\f$ using CG solver.
  template <int dim, int spacedim>
  std::pair<unsigned int, double> SolidSolver<dim, spacedim>::solve(
    const SparseMatrix<double> &A, Vector<double> &x, const Vector<double> &b)
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    SolverControl solver_control(A.m() * 100, 1e-12 * b.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);

    cg.solve(A, x, b, preconditioner);
    constraints.distribute(x);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::calculate_KE()
  {
    double ke = 0;
    std::vector<double> solid_momentum(dim, 0);
    std::ofstream file_ke;
    std::ofstream file_momx;
    std::ofstream file_momy;
    std::ofstream file_momz;

    if (time.current() == 0.0)
      {
        file_ke.open("solid_KE.txt");
        file_ke << "Time"
                << "\t"
                << "Solid KE"
                << "\n";

        file_momx.open("solid_mom_x.txt");
        file_momx << "Time"
                  << "\t"
                  << "Solid Mom X"
                  << "\n";

        file_momy.open("solid_mom_y.txt");
        file_momy << "Time"
                  << "\t"
                  << "Solid Mom Y"
                  << "\n";

        if (dim == 3)
          {
            file_momz.open("solid_mom_z.txt");
            file_momz << "Time"
                      << "\t"
                      << "Solid Mom Z"
                      << "\n";
          }
      }
    else
      {
        file_ke.open("solid_KE.txt", std::ios_base::app);
        file_momx.open("solid_mom_x.txt", std::ios_base::app);
        file_momy.open("solid_mom_y.txt", std::ios_base::app);
        if (dim == 3)
          file_momz.open("solid_mom_z.txt", std::ios_base::app);
      }

    FEValues<dim, spacedim> fe_values(
      fe, volume_quad_formula, update_values | update_quadrature_points);
    std::vector<unsigned int> dof_touched(dof_handler.n_dofs(), 0);
    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            auto index = dof_indices[i];
            if (!dof_touched[index])
              {
                dof_touched[index] = 1;
                ke += 0.5 * current_velocity[index] * current_velocity[index] *
                      nodal_mass[index];
                auto component = fe.system_to_component_index(i).first;
                solid_momentum[component] +=
                  current_velocity[index] * nodal_mass[index];
              }
          }
      }
    file_ke << time.current() << "\t" << ke << "\n";
    file_ke.close();

    file_momx << time.current() << "\t" << solid_momentum[0] << "\n";
    file_momx.close();

    file_momy << time.current() << "\t" << solid_momentum[1] << "\n";
    file_momy.close();

    if (dim == 3)
      {
        file_momz << time.current() << "\t" << solid_momentum[2] << "\n";
        file_momz.close();
      }
  }

  template <int dim, int spacedim>
  void
  SolidSolver<dim, spacedim>::output_results(const unsigned int output_index)
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    std::vector<std::string> solution_names(spacedim, "displacements");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        spacedim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim, DoFHandler<dim, spacedim>> data_out;
    data_out.attach_dof_handler(dof_handler);

    // displacements
    data_out.add_data_vector(dof_handler,
                             current_displacement,
                             solution_names,
                             data_component_interpretation);
    // velocity
    solution_names = std::vector<std::string>(spacedim, "velocities");
    data_out.add_data_vector(dof_handler,
                             current_velocity,
                             solution_names,
                             data_component_interpretation);
    // acceleration
    solution_names = std::vector<std::string>(spacedim, "acceleration");
    data_out.add_data_vector(dof_handler,
                             current_acceleration,
                             solution_names,
                             data_component_interpretation);
    // nodal forces due to surface tractrion
    solution_names =
      std::vector<std::string>(spacedim, "nodal_forces_traction");
    data_out.add_data_vector(dof_handler,
                             nodal_forces_traction,
                             solution_names,
                             data_component_interpretation);

    // nodal forces due to penalty
    solution_names = std::vector<std::string>(spacedim, "nodal_forces_penalty");
    data_out.add_data_vector(dof_handler,
                             nodal_forces_penalty,
                             solution_names,
                             data_component_interpretation);

    // velocity difference between Eulerian and Lagrangian mesh calculated at
    // Lagrangian mesh
    solution_names =
      std::vector<std::string>(spacedim, "fsi_velocity_difference");
    data_out.add_data_vector(dof_handler,
                             fsi_vel_diff_lag,
                             solution_names,
                             data_component_interpretation);

    // nodal mass with added mass effect
    Vector<double> nodal_mass_output;
    nodal_mass_output.reinit(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
      nodal_mass_output[i] = nodal_mass[i] + added_mass_effect[i];
    solution_names = std::vector<std::string>(spacedim, "nodal_mass");
    data_out.add_data_vector(dof_handler,
                             nodal_mass_output,
                             solution_names,
                             data_component_interpretation);
    // material ID
    Vector<float> mat(triangulation.n_active_cells());
    int i = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        mat[i++] = cell->material_id();
      }
    data_out.add_data_vector(mat, "material_id");

    // strain and stress
    data_out.add_data_vector(scalar_dof_handler, strain[0][0], "Exx");
    data_out.add_data_vector(scalar_dof_handler, strain[0][1], "Exy");
    data_out.add_data_vector(scalar_dof_handler, strain[1][1], "Eyy");
    data_out.add_data_vector(scalar_dof_handler, stress[0][0], "Sxx");
    data_out.add_data_vector(scalar_dof_handler, stress[0][1], "Sxy");
    data_out.add_data_vector(scalar_dof_handler, stress[1][1], "Syy");
    if (spacedim == 3)
      {
        data_out.add_data_vector(scalar_dof_handler, strain[0][2], "Exz");
        data_out.add_data_vector(scalar_dof_handler, strain[1][2], "Eyz");
        data_out.add_data_vector(scalar_dof_handler, strain[2][2], "Ezz");
        data_out.add_data_vector(scalar_dof_handler, stress[0][2], "Sxz");
        data_out.add_data_vector(scalar_dof_handler, stress[1][2], "Syz");
        data_out.add_data_vector(scalar_dof_handler, stress[2][2], "Szz");
      }

    data_out.build_patches();

    std::string basename = "solid";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  template <int dim, int spacedim>
  void
  SolidSolver<dim, spacedim>::refine_mesh(const unsigned int min_grid_level,
                                          const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    using type =
      std::map<types::boundary_id, const Function<spacedim, double> *>;
    KellyErrorEstimator<dim, spacedim>::estimate(dof_handler,
                                                 face_quad_formula,
                                                 type(),
                                                 current_displacement,
                                                 estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, estimated_error_per_cell, 0.6, 0.4);
    if (triangulation.n_levels() > max_grid_level)
      {
        for (auto cell = triangulation.begin_active(max_grid_level);
             cell != triangulation.end();
             ++cell)
          {
            cell->clear_refine_flag();
          }
      }

    for (auto cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level);
         ++cell)
      {
        cell->clear_coarsen_flag();
      }

    std::vector<
      SolutionTransfer<dim, Vector<double>, DoFHandler<dim, spacedim>>>
      solution_trans(
        3,
        SolutionTransfer<dim, Vector<double>, DoFHandler<dim, spacedim>>(
          dof_handler));
    std::vector<Vector<double>> buffer{
      previous_displacement, previous_velocity, previous_acceleration};

    triangulation.prepare_coarsening_and_refinement();

    for (unsigned int i = 0; i < 3; ++i)
      {
        solution_trans[i].prepare_for_coarsening_and_refinement(buffer[i]);
      }

    triangulation.execute_coarsening_and_refinement();

    setup_dofs();
    initialize_system();

    solution_trans[0].interpolate(buffer[0], previous_displacement);
    solution_trans[1].interpolate(buffer[1], previous_velocity);
    solution_trans[2].interpolate(buffer[2], previous_acceleration);

    constraints.distribute(previous_displacement);
    constraints.distribute(previous_velocity);
    constraints.distribute(previous_acceleration);
  }

  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::run()
  {
    triangulation.refine_global(parameters.global_refinements[1]);
    setup_dofs();
    initialize_system();

    // Time loop
    run_one_step(true);
    while (time.end() - time.current() > 1e-12)
      {
        run_one_step(false);
      }
  }

  template <int dim, int spacedim>
  Vector<double> SolidSolver<dim, spacedim>::get_current_solution() const
  {
    return current_displacement;
  }

  template class SolidSolver<2>;
  template class SolidSolver<3>;
  template class SolidSolver<2, 3>;
} // namespace Solid

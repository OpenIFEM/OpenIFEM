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

    constraints.close();
  }

  template <int dim, int spacedim>
  void SolidSolver<dim, spacedim>::initialize_system()
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    pattern.copy_from(dsp);

    system_matrix.reinit(pattern);
    mass_matrix.reinit(pattern);
    stiffness_matrix.reinit(pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());

    strain = std::vector<std::vector<Vector<double>>>(
      spacedim,
      std::vector<Vector<double>>(spacedim,
                                  Vector<double>(scalar_dof_handler.n_dofs())));
    stress = std::vector<std::vector<Vector<double>>>(
      spacedim,
      std::vector<Vector<double>>(spacedim,
                                  Vector<double>(scalar_dof_handler.n_dofs())));

    cellwise_stress = std::vector<Vector<double>>(
      6, Vector<double>(triangulation.n_active_cells()));

    for (unsigned int i = 0; i < cellwise_stress.size(); ++i)
      {
        cellwise_stress[i].reinit(triangulation.n_active_cells());
      }

    // Set up cell property, which contains the FSI traction required in FSI
    // simulation
    cell_property.initialize(triangulation.begin_active(),
                             triangulation.end(),
                             GeometryInfo<dim>::faces_per_cell);
  }

  // Solve linear system \f$Ax = b\f$ using CG solver.
  template <int dim, int spacedim>
  std::pair<unsigned int, double> SolidSolver<dim, spacedim>::solve(
    const SparseMatrix<double> &A, Vector<double> &x, const Vector<double> &b)
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    SolverControl solver_control(A.m(), 1e-6 * b.l2_norm());
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);

    cg.solve(A, x, b, preconditioner);
    constraints.distribute(x);

    return {solver_control.last_step(), solver_control.last_value()};
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
    data_out.add_data_vector(cellwise_stress[0], "cell_sxx");
    data_out.add_data_vector(cellwise_stress[1], "cell_sxy");
    data_out.add_data_vector(cellwise_stress[2], "cell_syy");
    if (spacedim == 3)
      {
        data_out.add_data_vector(scalar_dof_handler, strain[0][2], "Exz");
        data_out.add_data_vector(scalar_dof_handler, strain[1][2], "Eyz");
        data_out.add_data_vector(scalar_dof_handler, strain[2][2], "Ezz");
        data_out.add_data_vector(scalar_dof_handler, stress[0][2], "Sxz");
        data_out.add_data_vector(scalar_dof_handler, stress[1][2], "Syz");
        data_out.add_data_vector(scalar_dof_handler, stress[2][2], "Szz");
        data_out.add_data_vector(cellwise_stress[3], "cell_sxz");
        data_out.add_data_vector(cellwise_stress[4], "cell_syz");
        data_out.add_data_vector(cellwise_stress[5], "cell_szz");
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

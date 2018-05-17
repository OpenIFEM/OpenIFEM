#include "fluidSolver.h"

namespace Fluid
{
  template <int dim>
  BlockVector<double> FluidSolver<dim>::get_current_solution() const
  {
    return present_solution;
  }

  template <int dim>
  FluidSolver<dim>::FluidSolver(Triangulation<dim> &tria,
                                const Parameters::AllParameters &parameters,
                                std::shared_ptr<Function<dim>> bc)
    : triangulation(tria),
      fe(FE_Q<dim>(parameters.fluid_velocity_degree),
         dim,
         FE_Q<dim>(parameters.fluid_pressure_degree),
         1),
      dof_handler(triangulation),
      volume_quad_formula(parameters.fluid_velocity_degree + 1),
      face_quad_formula(parameters.fluid_velocity_degree + 1),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval),
      timer(std::cout, TimerOutput::never, TimerOutput::wall_times),
      parameters(parameters),
      boundary_values(bc)
  {
  }

  template <int dim>
  void FluidSolver<dim>::setup_dofs()
  {
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);

    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block.resize(2);
    DoFTools::count_dofs_per_block(
      dof_handler, dofs_per_block, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    std::cout << "   Number of active fluid cells: "
              << triangulation.n_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dof_u << '+' << dof_p << ')' << std::endl;
  }

  template <int dim>
  void FluidSolver<dim>::make_constraints()
  {
    nonzero_constraints.clear();
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    for (auto itr = parameters.fluid_dirichlet_bcs.begin();
         itr != parameters.fluid_dirichlet_bcs.end();
         ++itr)
      {
        // First get the id, flag and value from the input file
        unsigned int id = itr->first;
        unsigned int flag = itr->second.first;
        std::vector<double> value = itr->second.second;

        // To make VectorTools::interpolate_boundary_values happy,
        // a vector of bool and a vector of double which are of size
        // dim + 1 are required.
        std::vector<bool> mask(dim + 1, false);
        std::vector<double> augmented_value(dim + 1, 0.0);
        // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
        switch (flag)
          {
          case 1:
            mask[0] = true;
            augmented_value[0] = value[0];
            break;
          case 2:
            mask[1] = true;
            augmented_value[1] = value[0];
            break;
          case 3:
            mask[0] = true;
            mask[1] = true;
            augmented_value[0] = value[0];
            augmented_value[1] = value[1];
            break;
          case 4:
            mask[2] = true;
            augmented_value[2] = value[0];
            break;
          case 5:
            mask[0] = true;
            mask[2] = true;
            augmented_value[0] = value[0];
            augmented_value[2] = value[1];
            break;
          case 6:
            mask[1] = true;
            mask[2] = true;
            augmented_value[1] = value[0];
            augmented_value[2] = value[1];
            break;
          case 7:
            mask[0] = true;
            mask[1] = true;
            mask[2] = true;
            augmented_value[0] = value[0];
            augmented_value[1] = value[1];
            augmented_value[2] = value[2];
            break;
          default:
            AssertThrow(false, ExcMessage("Unrecogonized component flag!"));
            break;
          }
        if (parameters.use_hard_coded_values == 1)
          {
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     id,
                                                     *boundary_values,
                                                     nonzero_constraints,
                                                     ComponentMask(mask));
          }
        else
          {
            VectorTools::interpolate_boundary_values(
              dof_handler,
              id,
              Functions::ConstantFunction<dim>(augmented_value),
              nonzero_constraints,
              ComponentMask(mask));
          }
        VectorTools::interpolate_boundary_values(
          dof_handler,
          id,
          Functions::ZeroFunction<dim>(dim + 1),
          zero_constraints,
          ComponentMask(mask));
      }
    nonzero_constraints.close();
    zero_constraints.close();
  }

  template <int dim>
  void FluidSolver<dim>::setup_cell_property()
  {
    const unsigned int n_q_points = volume_quad_formula.size();
    cell_property.initialize(
      triangulation.begin_active(), triangulation.end(), n_q_points);
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<CellProperty>> p =
          cell_property.get_data(cell);
        Assert(p.size() == n_q_points,
               ExcMessage("Wrong number of cell property!"));
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            p[q]->indicator = 0;
            p[q]->fsi_acceleration = 0;
            p[q]->fsi_stress = 0;
          }
      }
  }

  template <int dim>
  void FluidSolver<dim>::initialize_system()
  {
    system_matrix.clear();
    mass_matrix.clear();
    mass_schur.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);

    present_solution.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);

    // Compute the sparsity pattern for mass schur in advance.
    // It should be the same as \f$BB^T\f$.
    DynamicSparsityPattern schur_pattern(dofs_per_block[1], dofs_per_block[1]);
    schur_pattern.compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                        sparsity_pattern.block(0, 1));
    mass_schur_pattern.copy_from(schur_pattern);
    mass_schur.reinit(mass_schur_pattern);

    // Cell property
    setup_cell_property();
  }

  template <int dim>
  void FluidSolver<dim>::refine_mesh(const unsigned int min_grid_level,
                                     const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(parameters.fluid_velocity_degree),
      typename FunctionMap<dim>::type(),
      present_solution,
      estimated_error_per_cell,
      fe.component_mask(velocity));
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

    BlockVector<double> buffer(present_solution);
    SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);

    triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(buffer);

    triangulation.execute_coarsening_and_refinement();

    setup_dofs();
    make_constraints();
    initialize_system();

    solution_transfer.interpolate(buffer, present_solution);
    nonzero_constraints.distribute(present_solution);
  }

  template <int dim>
  void FluidSolver<dim>::output_results(const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    std::cout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Indicator
    Vector<float> ind(triangulation.n_active_cells());
    int i = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        auto p = cell_property.get_data(cell);
        bool artificial = false;
        for (auto ptr : p)
          {
            if (ptr->indicator == 1)
              {
                artificial = true;
                break;
              }
          }
        ind[i++] = artificial;
      }
    data_out.add_data_vector(ind, "Indicator");

    data_out.build_patches(parameters.fluid_velocity_degree);

    std::string basename = "fluid";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  template class FluidSolver<2>;
  template class FluidSolver<3>;
} // namespace Fluid

#include "mpi_shared_solid_solver.h"

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    template <int dim>
    SharedSolidSolver<dim>::SharedSolidSolver(
      Triangulation<dim> &tria, const Parameters::AllParameters &parameters)
      : triangulation(tria),
        parameters(parameters),
        dof_handler(triangulation),
        dg_dof_handler(triangulation),
        fe(FE_Q<dim>(parameters.solid_degree), dim),
        dg_fe(FE_DGQ<dim>(parameters.solid_degree)),
        volume_quad_formula(parameters.solid_degree + 1),
        face_quad_formula(parameters.solid_degree + 1),
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, (this_mpi_process == 0)),
        time(parameters.end_time,
             parameters.time_step,
             parameters.output_interval,
             parameters.refinement_interval),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
    {
    }

    template <int dim>
    SharedSolidSolver<dim>::~SharedSolidSolver()
    {
      dg_dof_handler.clear();
      dof_handler.clear();
      timer.print_summary();
    }

    template <int dim>
    void SharedSolidSolver<dim>::setup_dofs()
    {
      TimerOutput::Scope timer_section(timer, "Setup system");

      // Because in mpi solid solver we take serial triangulation,
      // here we partition it.
      GridTools::partition_triangulation(n_mpi_processes, triangulation);

      dof_handler.distribute_dofs(fe);
      DoFRenumbering::subdomain_wise(dof_handler);
      dg_dof_handler.distribute_dofs(dg_fe);

      // Extract the locally owned and relevant dofs
      const std::vector<IndexSet> locally_owned_dofs_per_proc =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
      locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

      // The Dirichlet boundary conditions are stored in the ConstraintMatrix
      // object. It does not need to modify the sparse matrix after assembly,
      // because it is applied in the assembly process,
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
          std::vector<bool> mask(dim, false);
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
            Functions::ZeroFunction<dim>(dim),
            constraints,
            ComponentMask(mask));
        }

      constraints.close();

      pcout << "  Number of active solid cells: "
            << triangulation.n_active_cells() << std::endl
            << "  Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
    }

    template <int dim>
    void SharedSolidSolver<dim>::initialize_system()
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

      system_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      mass_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      stiffness_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      system_rhs.reinit(locally_owned_dofs, mpi_communicator);

      current_acceleration.reinit(locally_owned_dofs, mpi_communicator);

      current_velocity.reinit(locally_owned_dofs, mpi_communicator);

      current_displacement.reinit(locally_owned_dofs, mpi_communicator);

      previous_acceleration.reinit(locally_owned_dofs, mpi_communicator);

      previous_velocity.reinit(locally_owned_dofs, mpi_communicator);

      previous_displacement.reinit(locally_owned_dofs, mpi_communicator);

      // Set up cell property, which contains the FSI traction required in FSI
      // simulation
      const unsigned int n_data =
        face_quad_formula.size() * GeometryInfo<dim>::faces_per_cell;
      cell_property.initialize(
        triangulation.begin_active(), triangulation.end(), n_data);
    }

    // Solve linear system \f$Ax = b\f$ using CG solver.
    template <int dim>
    std::pair<unsigned int, double>
    SharedSolidSolver<dim>::solve(const PETScWrappers::MPI::SparseMatrix &A,
                                  PETScWrappers::MPI::Vector &x,
                                  const PETScWrappers::MPI::Vector &b)
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");

      SolverControl solver_control(dof_handler.n_dofs() * 2,
                                   1e-8 * b.l2_norm());

      PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

      PETScWrappers::PreconditionNone preconditioner(A);

      cg.solve(A, x, b, preconditioner);

      Vector<double> localized_x(x);
      constraints.distribute(localized_x);
      x = localized_x;

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void SharedSolidSolver<dim>::output_results(
      const unsigned int output_index) const
    {
      TimerOutput::Scope timer_section(timer, "Output results");
      pcout << "Writing solid results..." << std::endl;

      // Since only process 0 writes the output, we want all the others
      // to sned their data to process 0, which is automatically done
      // in this copy constructor.
      Vector<double> solution(current_displacement);
      if (this_mpi_process == 0)
        {
          std::vector<std::string> solution_names(dim, "displacements");
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);

          data_out.add_data_vector(solution,
                                   solution_names,
                                   DataOut<dim>::type_dof_data,
                                   data_component_interpretation);

          std::vector<unsigned int> subdomain_int(
            triangulation.n_active_cells());
          GridTools::get_subdomain_association(triangulation, subdomain_int);
          Vector<float> subdomain(subdomain_int.begin(), subdomain_int.end());
          data_out.add_data_vector(subdomain, "subdomain");

          data_out.build_patches();

          std::string basename =
            "solid-" + Utilities::int_to_string(output_index, 6) + "-";

          std::string filename = basename + ".vtu";

          std::ofstream output(filename);
          data_out.write_vtu(output);

          static std::vector<std::pair<double, std::string>> times_and_names;
          times_and_names.push_back({time.current(), filename});
          std::ofstream pvd_output("solid.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template <int dim>
    void SharedSolidSolver<dim>::refine_mesh(const unsigned int min_grid_level,
                                             const unsigned int max_grid_level)
    {
      TimerOutput::Scope timer_section(timer, "Refine mesh");
      pcout << "Refining mesh..." << std::endl;

      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

      // In order to estimate error, the vector must has the entire solution.
      Vector<double> solution(current_displacement);

      KellyErrorEstimator<dim>::estimate(dof_handler,
                                         face_quad_formula,
                                         typename FunctionMap<dim>::type(),
                                         solution,
                                         estimated_error_per_cell);

      // Set the refine and coarsen flag
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

      // Prepare to transfer previous solutions
      std::vector<parallel::distributed::
                    SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
        trans(
          3,
          parallel::distributed::SolutionTransfer<dim,
                                                  PETScWrappers::MPI::Vector>(
            dof_handler));
      std::vector<PETScWrappers::MPI::Vector> buffers(
        3,
        PETScWrappers::MPI::Vector(
          locally_owned_dofs, locally_relevant_dofs, mpi_communicator));
      buffers[0] = previous_displacement;
      buffers[1] = previous_velocity;
      buffers[2] = previous_acceleration;

      triangulation.prepare_coarsening_and_refinement();

      for (unsigned int i = 0; i < 3; ++i)
        {
          trans[i].prepare_for_coarsening_and_refinement(buffers[i]);
        }

      // Refine the mesh
      triangulation.execute_coarsening_and_refinement();

      // Reinitialize the system
      setup_dofs();
      initialize_system();

      // Transfer the previous solutions and handle the constraints
      trans[0].interpolate(previous_displacement);
      trans[1].interpolate(previous_velocity);
      trans[2].interpolate(previous_acceleration);

      constraints.distribute(previous_displacement);
      constraints.distribute(previous_velocity);
      constraints.distribute(previous_acceleration);
    }

    template <int dim>
    void SharedSolidSolver<dim>::run()
    {
      triangulation.refine_global(parameters.global_refinement);
      setup_dofs();
      initialize_system();

      // Time loop
      run_one_step(true);
      while (time.end() - time.current() > 1e-12)
        {
          run_one_step(false);
        }
    }

    template <int dim>
    PETScWrappers::MPI::Vector
    SharedSolidSolver<dim>::get_current_solution() const
    {
      return current_displacement;
    }

    template class SharedSolidSolver<2>;
    template class SharedSolidSolver<3>;
  } // namespace MPI
} // namespace Solid

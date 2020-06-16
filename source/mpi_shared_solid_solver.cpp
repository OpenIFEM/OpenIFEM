#include "mpi_shared_solid_solver.h"

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    template <int dim, int spacedim>
    SharedSolidSolver<dim, spacedim>::SharedSolidSolver(
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
        mpi_communicator(MPI_COMM_WORLD),
        n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
        this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
        pcout(std::cout, (this_mpi_process == 0)),
        time(parameters.end_time,
             parameters.time_step,
             parameters.output_interval,
             parameters.refinement_interval,
             parameters.save_interval),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
    {
    }

    template <int dim, int spacedim>
    SharedSolidSolver<dim, spacedim>::~SharedSolidSolver()
    {
      scalar_dof_handler.clear();
      dof_handler.clear();
      timer.print_summary();
    }

    template <int dim, int spacedim>
    void SharedSolidSolver<dim, spacedim>::setup_dofs()
    {
      TimerOutput::Scope timer_section(timer, "Setup system");

      // Because in mpi solid solver we take serial triangulation,
      // here we partition it.
      GridTools::partition_triangulation(n_mpi_processes, triangulation);

      dof_handler.distribute_dofs(fe);
      DoFRenumbering::subdomain_wise(dof_handler);
      scalar_dof_handler.distribute_dofs(scalar_fe);
      DoFRenumbering::subdomain_wise(scalar_dof_handler);

      // Extract the locally owned and relevant dofs
      const std::vector<IndexSet> locally_owned_dofs_per_proc =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
      locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];

      const std::vector<IndexSet> locally_owned_scalar_dofs_per_proc =
        DoFTools::locally_owned_dofs_per_subdomain(scalar_dof_handler);
      locally_owned_scalar_dofs =
        locally_owned_scalar_dofs_per_proc[this_mpi_process];

      // The Dirichlet boundary conditions are stored in the AffineConstraints
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

      pcout << "  Number of active solid cells: "
            << triangulation.n_active_cells() << std::endl
            << "  Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
    }

    template <int dim, int spacedim>
    void SharedSolidSolver<dim, spacedim>::initialize_system()
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

      system_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      mass_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      stiffness_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      damping_matrix.reinit(
        locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

      system_rhs.reinit(locally_owned_dofs, mpi_communicator);

      current_acceleration.reinit(locally_owned_dofs, mpi_communicator);

      current_velocity.reinit(locally_owned_dofs, mpi_communicator);

      current_displacement.reinit(locally_owned_dofs, mpi_communicator);

      previous_acceleration.reinit(locally_owned_dofs, mpi_communicator);

      previous_velocity.reinit(locally_owned_dofs, mpi_communicator);

      previous_displacement.reinit(locally_owned_dofs, mpi_communicator);

      fsi_stress_rows.resize(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          fsi_stress_rows[d].reinit(dof_handler.n_dofs());
        }
      fluid_velocity.reinit(dof_handler.n_dofs());

      strain = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        spacedim,
        std::vector<PETScWrappers::MPI::Vector>(
          spacedim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));
      stress = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        spacedim,
        std::vector<PETScWrappers::MPI::Vector>(
          spacedim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));
    }

    // Solve linear system \f$Ax = b\f$ using CG solver.
    template <int dim, int spacedim>
    std::pair<unsigned int, double> SharedSolidSolver<dim, spacedim>::solve(
      const PETScWrappers::MPI::SparseMatrix &A,
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

    template <int dim, int spacedim>
    void SharedSolidSolver<dim, spacedim>::output_results(
      const unsigned int output_index)
    {
      TimerOutput::Scope timer_section(timer, "Output results");
      pcout << "Writing solid results..." << std::endl;

      // Since only process 0 writes the output, we want all the others
      // to sned their data to process 0, which is automatically done
      // in this copy constructor.
      Vector<double> displacement(current_displacement);
      Vector<double> velocity(current_velocity);

      std::vector<std::vector<Vector<double>>> localized_strain(
        spacedim, std::vector<Vector<double>>(spacedim));
      std::vector<std::vector<Vector<double>>> localized_stress(
        spacedim, std::vector<Vector<double>>(spacedim));
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            {
              localized_strain[i][j] = strain[i][j];
              localized_stress[i][j] = stress[i][j];
            }
        }
      if (this_mpi_process == 0)
        {
          std::vector<std::string> solution_names(spacedim, "displacements");
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
              spacedim,
              DataComponentInterpretation::component_is_part_of_vector);
          DataOut<dim, DoFHandler<dim, spacedim>> data_out;
          data_out.attach_dof_handler(dof_handler);

          // displacements
          data_out.add_data_vector(
            displacement,
            solution_names,
            DataOut<dim, DoFHandler<dim, spacedim>>::type_dof_data,
            data_component_interpretation);

          // velocity
          solution_names = std::vector<std::string>(spacedim, "velocities");
          data_out.add_data_vector(
            velocity,
            solution_names,
            DataOut<dim, DoFHandler<dim, spacedim>>::type_dof_data,
            data_component_interpretation);

          std::vector<unsigned int> subdomain_int(
            triangulation.n_active_cells());
          GridTools::get_subdomain_association(triangulation, subdomain_int);
          Vector<float> subdomain(subdomain_int.begin(), subdomain_int.end());
          data_out.add_data_vector(subdomain, "subdomain");

          // material ID
          Vector<float> mat(triangulation.n_active_cells());
          int i = 0;
          for (auto cell = triangulation.begin_active();
               cell != triangulation.end();
               ++cell)
            {
              mat[i++] = cell->material_id();
            }
          data_out.add_data_vector(mat, "material_id");

          data_out.add_data_vector(
            scalar_dof_handler, localized_strain[0][0], "Exx");
          data_out.add_data_vector(
            scalar_dof_handler, localized_strain[0][1], "Exy");
          data_out.add_data_vector(
            scalar_dof_handler, localized_strain[1][1], "Eyy");
          data_out.add_data_vector(
            scalar_dof_handler, localized_stress[0][0], "Sxx");
          data_out.add_data_vector(
            scalar_dof_handler, localized_stress[0][1], "Sxy");
          data_out.add_data_vector(
            scalar_dof_handler, localized_stress[1][1], "Syy");
          if (spacedim == 3)
            {
              data_out.add_data_vector(
                scalar_dof_handler, localized_strain[0][2], "Exz");
              data_out.add_data_vector(
                scalar_dof_handler, localized_strain[1][2], "Eyz");
              data_out.add_data_vector(
                scalar_dof_handler, localized_strain[2][2], "Ezz");
              data_out.add_data_vector(
                scalar_dof_handler, localized_stress[0][2], "Sxz");
              data_out.add_data_vector(
                scalar_dof_handler, localized_stress[1][2], "Syz");
              data_out.add_data_vector(
                scalar_dof_handler, localized_stress[2][2], "Szz");
            }

          data_out.build_patches();

          std::string basename =
            "solid-" + Utilities::int_to_string(output_index, 6);

          std::string filename = basename + ".vtu";

          std::ofstream output(filename);
          data_out.write_vtu(output);

          times_and_names.push_back({time.current(), filename});
          std::ofstream pvd_output("solid.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template <int dim, int spacedim>
    void SharedSolidSolver<dim, spacedim>::refine_mesh(
      const unsigned int min_grid_level, const unsigned int max_grid_level)
    {
      TimerOutput::Scope timer_section(timer, "Refine mesh");
      pcout << "Refining mesh..." << std::endl;

      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

      // In order to estimate error, the vector must has the entire solution.
      Vector<double> solution(current_displacement);

      using type =
        std::map<types::boundary_id, const Function<spacedim, double> *>;
      KellyErrorEstimator<dim, spacedim>::estimate(dof_handler,
                                                   face_quad_formula,
                                                   type(),
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
      std::vector<
        parallel::distributed::SolutionTransfer<dim,
                                                PETScWrappers::MPI::Vector,
                                                DoFHandler<dim, spacedim>>>
        trans(
          3,
          parallel::distributed::SolutionTransfer<dim,
                                                  PETScWrappers::MPI::Vector,
                                                  DoFHandler<dim, spacedim>>(
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

    template <int dim, int spacedim>
    void SharedSolidSolver<dim, spacedim>::run()
    {
      triangulation.refine_global(parameters.global_refinements[1]);
      bool success_load = load_checkpoint();
      if (!success_load)
        {
          setup_dofs();
          initialize_system();
        }

      // Time loop
      if (!success_load)
        run_one_step(true);
      else
        // If we load from previous task, we need to assemble the mass matrix
        assemble_system(true);
      while (time.end() - time.current() > 1e-12)
        {
          run_one_step(false);
        }
    }

    template <int dim, int spacedim>
    PETScWrappers::MPI::Vector
    SharedSolidSolver<dim, spacedim>::get_current_solution() const
    {
      return current_displacement;
    }

    template <int dim, int spacedim>
    void
    SharedSolidSolver<dim, spacedim>::save_checkpoint(const int output_index)
    {
      // Save the solution
      Vector<double> localized_disp(current_displacement);
      Vector<double> localized_vel(current_velocity);
      Vector<double> localized_acc(current_acceleration);

      if (this_mpi_process == 0)
        {
          // Specify the current working path
          fs::path local_path = fs::current_path();
          // A set to store all the filenames for checkpoints
          std::set<fs::path> checkpoints;
          // Find the checkpoints and remove excess ones
          // Only keep the latest one
          for (const auto &p : fs::directory_iterator(local_path))
            {
              if (p.path().extension() == ".solid_checkpoint_displacement")
                {
                  checkpoints.insert(p.path());
                }
            }
          while (checkpoints.size() > 1)
            {
              pcout << "Removing " << *checkpoints.begin() << std::endl;
              fs::path to_be_removed(*checkpoints.begin());
              fs::remove(to_be_removed);
              to_be_removed.replace_extension(".solid_checkpoint_velocity");
              fs::remove(to_be_removed);
              to_be_removed.replace_extension(".solid_checkpoint_acceleration");
              fs::remove(to_be_removed);
              checkpoints.erase(checkpoints.begin());
            }
          // Name the checkpoint file
          fs::path checkpoint_file(local_path);
          checkpoint_file.append(Utilities::int_to_string(output_index, 6));
          checkpoint_file.replace_extension(".solid_checkpoint_displacement");
          pcout << "Prepare to save to " << checkpoint_file << std::endl;
          std::ofstream disp(checkpoint_file);
          checkpoint_file.replace_extension(".solid_checkpoint_velocity");
          pcout << "Prepare to save to " << checkpoint_file << std::endl;
          std::ofstream vel(checkpoint_file);
          checkpoint_file.replace_extension(".solid_checkpoint_acceleration");
          std::ofstream acc(checkpoint_file);
          pcout << "Prepare to save to " << checkpoint_file << std::endl;
          localized_disp.block_write(disp);
          localized_vel.block_write(vel);
          localized_acc.block_write(acc);
        }

      pcout << "Checkpoint file successfully saved at time step "
            << output_index << "!" << std::endl;
    }

    template <int dim, int spacedim>
    bool SharedSolidSolver<dim, spacedim>::load_checkpoint()
    {
      // Specify the current working path
      fs::path local_path = fs::current_path();
      fs::path checkpoint_file(local_path);
      // Find the latest checkpoint
      for (const auto &p : fs::directory_iterator(local_path))
        {
          if (p.path().extension() == ".solid_checkpoint_displacement" &&
              (std::string(p.path().stem()) >
                 std::string(checkpoint_file.stem()) ||
               checkpoint_file == local_path))
            {
              checkpoint_file = p.path();
            }
        }
      // if no restart file is found, return false
      if (checkpoint_file == local_path)
        {
          pcout
            << "Did not find solid checkpoint files. Start from the beginning !"
            << std::endl;
          return false;
        }
      // set time step load the checkpoint file
      setup_dofs();
      initialize_system();
      Vector<double> localized_disp(current_displacement);
      Vector<double> localized_vel(current_velocity);
      Vector<double> localized_acc(current_acceleration);
      std::ifstream disp(checkpoint_file);
      checkpoint_file.replace_extension(".solid_checkpoint_velocity");
      std::ifstream vel(checkpoint_file);
      checkpoint_file.replace_extension(".solid_checkpoint_acceleration");
      std::ifstream acc(checkpoint_file);
      localized_disp.block_read(disp);
      localized_vel.block_read(vel);
      localized_acc.block_read(acc);

      current_displacement = localized_disp;
      current_velocity = localized_vel;
      current_acceleration = localized_acc;
      previous_displacement = current_displacement;
      previous_velocity = current_velocity;
      previous_acceleration = current_acceleration;
      // Update the time and names to set the current time and write
      // correct .pvd file.

      for (int i = 0; i <= Utilities::string_to_int(checkpoint_file.stem());
           ++i)
        {
          if ((time.current() == 0 || time.time_to_output()) &&
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
              std::string basename =
                "solid-" + Utilities::int_to_string(time.get_timestep(), 6);
              std::string filename = basename + ".vtu";

              times_and_names.push_back({time.current(), filename});
            }
          if (i == Utilities::string_to_int(checkpoint_file.stem()))
            break;
          time.increment();
        }

      pcout << "Checkpoint file successfully loaded from time step "
            << time.get_timestep() << "!" << std::endl;
      return true;
    }

    template class SharedSolidSolver<2>;
    template class SharedSolidSolver<3>;
    template class SharedSolidSolver<2, 3>;
  } // namespace MPI
} // namespace Solid

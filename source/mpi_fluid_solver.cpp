#include "mpi_fluid_solver.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    FluidSolver<dim>::~FluidSolver()
    {
      timer.print_summary();
      timer2.print_summary();
    }

    template <int dim>
    PETScWrappers::MPI::BlockVector
    FluidSolver<dim>::get_current_solution() const
    {
      return present_solution;
    }

    template <int dim>
    FluidSolver<dim>::FluidSolver(
      parallel::distributed::Triangulation<dim> &tria,
      const Parameters::AllParameters &parameters)
      : triangulation(tria),
        fe(FE_Q<dim>(parameters.fluid_velocity_degree),
           dim,
           FE_Q<dim>(parameters.fluid_pressure_degree),
           1),
        scalar_fe(parameters.fluid_velocity_degree),
        dof_handler(triangulation),
        scalar_dof_handler(triangulation),
        volume_quad_formula(parameters.fluid_velocity_degree + 1),
        face_quad_formula(parameters.fluid_velocity_degree + 1),
        parameters(parameters),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout,
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
        time(parameters.end_time,
             parameters.time_step,
             parameters.output_interval,
             parameters.refinement_interval,
             parameters.save_interval),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
        timer2(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
    {
    }

    template <int dim>
    void FluidSolver<dim>::add_hard_coded_boundary_condition(
      const int id,
      const std::function<double(
        const Point<dim> &, const unsigned int, const double)> &value_function)
    {
      AssertThrow(
        parameters.fluid_dirichlet_bcs.find(id) !=
          parameters.fluid_dirichlet_bcs.end(),
        ExcMessage("Hard coded BC ID not included in parameters file!"));
      auto success =
        this->hard_coded_boundary_values.insert({id, Field(value_function)});
      AssertThrow(success.second,
                  ExcMessage("Duplicated hard coded boundary conditions!"));
    }

    template <int dim>
    void FluidSolver<dim>::set_body_force(
      const std::function<double(const Point<dim> &, const unsigned int)> &bf)
    {
      body_force.reset(new Field([bf](const Point<dim> &p,
                                      const unsigned int component,
                                      const double time) {
        (void)time;
        return bf(p, component);
      }));
    }

    template <int dim>
    void FluidSolver<dim>::set_sigma_pml_field(
      const std::function<double(const Point<dim> &, const unsigned int)> &pml)
    {
      sigma_pml_field.reset(new Field([pml](const Point<dim> &p,
                                            const unsigned int component,
                                            const double time) {
        (void)time;
        return pml(p, component);
      }));
    }

    template <int dim>
    void FluidSolver<dim>::set_initial_condition(
      const std::function<double(const Point<dim> &, const unsigned int)>
        &condition)
    {
      initial_condition_field.reset(
        new std::function<double(const Point<dim> &, const unsigned int)>(
          condition));
    }

    template <int dim>
    void FluidSolver<dim>::setup_dofs()
    {
      // The first step is to associate DoFs with a given mesh.
      dof_handler.distribute_dofs(fe);
      scalar_dof_handler.distribute_dofs(scalar_fe);

      // We renumber the components to have all velocity DoFs come before
      // the pressure DoFs to be able to split the solution vector in two blocks
      // which are separately accessed in the block preconditioner.
      DoFRenumbering::Cuthill_McKee(dof_handler);
      std::vector<unsigned int> block_component(dim + 1, 0);
      block_component[dim] = 1;
      DoFRenumbering::component_wise(dof_handler, block_component);
      DoFRenumbering::component_wise(scalar_dof_handler);

      dofs_per_block.resize(2);
      DoFTools::count_dofs_per_block(
        dof_handler, dofs_per_block, block_component);
      unsigned int dof_u = dofs_per_block[0];
      unsigned int dof_p = dofs_per_block[1];

      // This part is new compared to serial code: we need to split up the
      // IndexSet
      // based on how we want to create the block matrices and vectors
      owned_partitioning.resize(2);
      owned_partitioning[0] =
        dof_handler.locally_owned_dofs().get_view(0, dof_u);
      owned_partitioning[1] =
        dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      relevant_partitioning.resize(2);
      relevant_partitioning[0] = locally_relevant_dofs.get_view(0, dof_u);
      relevant_partitioning[1] =
        locally_relevant_dofs.get_view(dof_u, dof_u + dof_p);

      locally_owned_scalar_dofs = scalar_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(scalar_dof_handler,
                                              locally_relevant_scalar_dofs);

      pcout << "   Number of active fluid cells: "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << dof_u << '+' << dof_p << ')' << std::endl;
    }

    template <int dim>
    void FluidSolver<dim>::make_constraints()
    {
      // In Newton's scheme, we first apply the boundary condition on the
      // solution obtained from the initial step. To make sure the boundary
      // conditions remain satisfied during Newton's iteration, zero boundary
      // conditions are used for the update \f$\delta u^k\f$. Therefore we set
      // up two different constraint objects. Dirichlet boundary conditions are
      // applied to both boundaries 0 and 1.

      // For inhomogeneous BC, only constant input values can be read from
      // the input file. If time or space dependent Dirichlet BCs are
      // desired, they must be implemented in BoundaryValues.
      {
        nonzero_constraints.clear();
        zero_constraints.clear();
        nonzero_constraints.reinit(locally_relevant_dofs);
        zero_constraints.reinit(locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                nonzero_constraints);
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
            auto hbc = hard_coded_boundary_values.find(id);
            if (hbc != hard_coded_boundary_values.end())
              {
                VectorTools::interpolate_boundary_values(
                  MappingQGeneric<dim>(parameters.fluid_velocity_degree),
                  dof_handler,
                  id,
                  hbc->second,
                  nonzero_constraints,
                  ComponentMask(mask));
              }
            else
              {
                VectorTools::interpolate_boundary_values(
                  MappingQGeneric<dim>(parameters.fluid_velocity_degree),
                  dof_handler,
                  id,
                  Functions::ConstantFunction<dim>(augmented_value),
                  nonzero_constraints,
                  ComponentMask(mask));
              }
            VectorTools::interpolate_boundary_values(
              MappingQGeneric<dim>(parameters.fluid_velocity_degree),
              dof_handler,
              id,
              Functions::ZeroFunction<dim>(dim + 1),
              zero_constraints,
              ComponentMask(mask));
          }
      }
      nonzero_constraints.close();
      zero_constraints.close();
    }

    template <int dim>
    void FluidSolver<dim>::setup_cell_property()
    {
      pcout << "   Setting up cell property..." << std::endl;
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell_property.initialize(cell, 1);
              const std::vector<std::shared_ptr<CellProperty>> p =
                cell_property.get_data(cell);
              p[0]->indicator = 0;
              p[0]->fsi_acceleration = 0;
              p[0]->fsi_stress = 0;
              p[0]->material_id = 1;
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
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
      mass_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

      // Compute the sparsity pattern for mass schur in advance.
      // The only nonzero block is (1, 1), which is the same as \f$BB^T\f$.
      BlockDynamicSparsityPattern schur_dsp(dofs_per_block, dofs_per_block);
      schur_dsp.block(1, 1).compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                                  sparsity_pattern.block(0, 1));
      mass_schur.reinit(owned_partitioning, schur_dsp, mpi_communicator);

      // present_solution is ghosted because it is used in the
      // output and mesh refinement functions.
      present_solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      solution_increment.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      fsi_acceleration.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      // system_rhs is non-ghosted because it is only used in the linear
      // solver and residual evaluation.
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      // Cell property
      setup_cell_property();

      // Apply initial condition
      if (initial_condition_field)
        {
          apply_initial_condition();
        }

      stress = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        dim,
        std::vector<PETScWrappers::MPI::Vector>(
          dim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));
    }

    template <int dim>
    void FluidSolver<dim>::apply_initial_condition()
    {
      AssertThrow(initial_condition_field != nullptr,
                  ExcMessage("No initial condition specified!"));
      AffineConstraints<double> initial_condition;
      initial_condition.reinit(locally_relevant_dofs);

      const std::vector<Point<dim>> &unit_points = fe.get_unit_support_points();
      Quadrature<dim> dummy_q(unit_points.size());
      MappingQGeneric<dim> mapping(1);
      FEValues<dim> dummy_fe_values(
        mapping, fe, dummy_q, update_quadrature_points);

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_artificial())
            continue;
          dummy_fe_values.reinit(cell);
          auto support_points = dummy_fe_values.get_quadrature_points();
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < unit_points.size(); ++i)
            {
              // Identify the component from the group
              auto base_index = fe.system_to_base_index(i);
              unsigned int component =
                base_index.first.first == 0 ? base_index.first.second : dim;
              Assert(component <= dim,
                     ExcMessage("Component should not excess dim!"));
              // Compute the initial condition value from the component and
              // coordinate
              double initial_condition_value = std::invoke(
                *initial_condition_field, support_points[i], component);
              // Assign it to the corresponding dof index
              auto line = dof_indices[i];
              initial_condition.add_line(line);
              initial_condition.set_inhomogeneity(line,
                                                  initial_condition_value);
            }
        }
      initial_condition.close();
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);
      initial_condition.distribute(tmp);
      present_solution = tmp;
    }

    template <int dim>
    void FluidSolver<dim>::refine_mesh(const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
    {
      TimerOutput::Scope timer_section(timer, "Refine mesh");

      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Vector velocity(0);
      using type = std::map<types::boundary_id, const Function<dim, double> *>;
      KellyErrorEstimator<dim>::estimate(dof_handler,
                                         face_quad_formula,
                                         type(),
                                         present_solution,
                                         estimated_error_per_cell,
                                         fe.component_mask(velocity));
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
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

      // Prepare to transfer
      parallel::distributed::SolutionTransfer<dim,
                                              PETScWrappers::MPI::BlockVector>
        trans(dof_handler);

      triangulation.prepare_coarsening_and_refinement();

      trans.prepare_for_coarsening_and_refinement(present_solution);

      // Refine the mesh
      triangulation.execute_coarsening_and_refinement();

      // Reinitialize the system
      setup_dofs();
      make_constraints();
      initialize_system();

      // Transfer solution
      // Need a non-ghosted vector for interpolation
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);
      tmp = 0;
      trans.interpolate(tmp);
      nonzero_constraints.distribute(tmp); // Is this line necessary?
      present_solution = tmp;
    }

    template <int dim>
    void FluidSolver<dim>::output_results(const unsigned int output_index) const
    {
      TimerOutput::Scope timer_section(timer, "Output results");

      pcout << "Writing results..." << std::endl;
      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.push_back("pressure");

      std::vector<std::string> fsi_force_names(dim, "fsi_force");
      fsi_force_names.push_back("dummy_fsi_force");

      std::vector<std::vector<PETScWrappers::MPI::Vector>> tmp_stress =
        std::vector<std::vector<PETScWrappers::MPI::Vector>>(
          dim,
          std::vector<PETScWrappers::MPI::Vector>(
            dim,
            PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                       locally_relevant_scalar_dofs,
                                       mpi_communicator)));
      tmp_stress = stress;

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      // vector to be output must be ghosted
      data_out.add_data_vector(present_solution,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);
      data_out.add_data_vector(fsi_acceleration,
                               fsi_force_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      // Partition
      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");

      // Indicator
      Vector<float> ind(triangulation.n_active_cells());
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);
              ind[cell->active_cell_index()] = p[0]->indicator;
            }
        }
      data_out.add_data_vector(ind, "Indicator");

      // stress
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[0][0], "Sxx");
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[0][1], "Sxy");
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[1][1], "Syy");
      if (dim == 3)
        {
          data_out.add_data_vector(scalar_dof_handler, tmp_stress[0][2], "Sxz");
          data_out.add_data_vector(scalar_dof_handler, tmp_stress[1][2], "Syz");
          data_out.add_data_vector(scalar_dof_handler, tmp_stress[2][2], "Szz");
        }

      data_out.build_patches(parameters.fluid_pressure_degree);

      std::string basename =
        "fluid" + Utilities::int_to_string(output_index, 6) + "-";

      std::string filename =
        basename +
        Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
        ".vtu";

      std::ofstream output(filename);
      data_out.write_vtu(output);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          for (unsigned int i = 0;
               i < Utilities::MPI::n_mpi_processes(mpi_communicator);
               ++i)
            {
              times_and_names.push_back(
                {time.current(),
                 basename + Utilities::int_to_string(i, 4) + ".vtu"});
            }
          std::ofstream pvd_output("fluid.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template <int dim>
    void FluidSolver<dim>::save_checkpoint(const int output_index)
    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          // Specify the current working path
          fs::path local_path = fs::current_path();
          // A set to store all the filenames for checkpoints
          std::set<fs::path> checkpoints;
          // Find the checkpoints and remove excess ones
          // Only keep the latest one
          for (const auto &p : fs::directory_iterator(local_path))
            {
              if (p.path().extension() == ".fluid_checkpoint")
                {
                  checkpoints.insert(p.path());
                }
            }
          while (checkpoints.size() > 1)
            {
              pcout << "Removing " << *checkpoints.begin() << std::endl;
              fs::path to_be_removed(*checkpoints.begin());
              fs::remove(to_be_removed);
              to_be_removed.replace_extension(".fluid_checkpoint.info");
              fs::remove(to_be_removed);
              checkpoints.erase(checkpoints.begin());
            }
        }
      // Name the checkpoint file
      std::string checkpoint_file = Utilities::int_to_string(output_index, 6);
      checkpoint_file.append(".fluid_checkpoint");
      // Save the solution
      parallel::distributed::SolutionTransfer<dim,
                                              PETScWrappers::MPI::BlockVector>
        sol_trans(dof_handler);
      sol_trans.prepare_for_serialization(present_solution);
      triangulation.save(checkpoint_file.c_str());
      pcout << "Checkpoint file successfully saved at time step "
            << output_index << "!" << std::endl;
    }

    template <int dim>
    bool FluidSolver<dim>::load_checkpoint()
    {
      // Specify the current working path
      fs::path local_path = fs::current_path();
      fs::path checkpoint_file(local_path);
      // Find the latest checkpoint
      for (const auto &p : fs::directory_iterator(local_path))
        {
          if (p.path().extension() == ".fluid_checkpoint" &&
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
            << "Did not find fluid checkpoint files. Start from the beginning !"
            << std::endl;
          return false;
        }
      // set time step load the checkpoint file
      pcout << "Loading checkpoint file " << checkpoint_file.filename().c_str()
            << "!" << std::endl;
      triangulation.load(checkpoint_file.filename().c_str());
      setup_dofs();
      make_constraints();
      initialize_system();
      parallel::distributed::SolutionTransfer<dim,
                                              PETScWrappers::MPI::BlockVector>
        sol_trans(dof_handler);
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);
      sol_trans.deserialize(tmp);
      present_solution = tmp;
      // Update the time and names to set the current time and write
      // correct .pvd file.

      for (int i = 0; i <= Utilities::string_to_int(checkpoint_file.stem());
           ++i)
        {
          if ((time.current() == 0 || time.time_to_output()) &&
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
              std::string basename =
                "fluid" + Utilities::int_to_string(time.get_timestep(), 6) +
                "-";
              for (unsigned int j = 0;
                   j < Utilities::MPI::n_mpi_processes(mpi_communicator);
                   ++j)
                {
                  times_and_names.push_back(
                    {time.current(),
                     basename + Utilities::int_to_string(j, 4) + ".vtu"});
                }
            }
          if (i == Utilities::string_to_int(checkpoint_file.stem()))
            break;
          time.increment();
          // Update the time for hard coded boundary conditions
          if (!this->hard_coded_boundary_values.empty())
            {
              for (auto &bc : hard_coded_boundary_values)
                {
                  bc.second.advance_time(time.get_delta_t());
                }
            }
        }

      pcout << "Checkpoint file successfully loaded from time step "
            << time.get_timestep() << "!" << std::endl;
      return true;
    }

    template <int dim>
    void FluidSolver<dim>::update_stress()
    {
      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            {
              stress[i][j] = 0;
            }
        }
      PETScWrappers::MPI::Vector surrounding_cells(locally_owned_scalar_dofs,
                                                   mpi_communicator);
      surrounding_cells = 0.0;
      // The stress tensors are stored as 2D vectors of shape dim*dim
      // at cell and quadrature point level.
      std::vector<std::vector<Vector<double>>> cell_stress(
        dim,
        std::vector<Vector<double>>(dim,
                                    Vector<double>(scalar_fe.dofs_per_cell)));
      std::vector<std::vector<Vector<double>>> quad_stress(
        dim,
        std::vector<Vector<double>>(
          dim, Vector<double>(volume_quad_formula.size())));

      // The projection matrix from quadrature points to the dofs.
      FullMatrix<double> qpt_to_dof(scalar_fe.dofs_per_cell,
                                    volume_quad_formula.size());
      FETools::compute_projection_from_quadrature_points_matrix(
        scalar_fe, volume_quad_formula, volume_quad_formula, qpt_to_dof);

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      const unsigned int n_q_points = volume_quad_formula.size();
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);
      std::vector<SymmetricTensor<2, dim>> sym_grad_v(n_q_points);
      std::vector<double> p(n_q_points);

      auto cell = dof_handler.begin_active();
      auto scalar_cell = scalar_dof_handler.begin_active();
      Vector<double> local_sorrounding_cells(scalar_fe.dofs_per_cell);
      local_sorrounding_cells = 1.0;
      for (; cell != dof_handler.end(); ++cell, ++scalar_cell)
        {
          if (!cell->is_locally_owned())
            continue;
          fe_values.reinit(cell);

          // Fluid symmetric velocity gradient
          fe_values[velocities].get_function_symmetric_gradients(
            present_solution, sym_grad_v);
          // Fluid pressure
          fe_values[pressure].get_function_values(present_solution, p);

          // Loop over all quadrature points to set FSI forces.
          for (unsigned int q = 0; q < volume_quad_formula.size(); ++q)
            {
              SymmetricTensor<2, dim> sigma =
                -p[q] * Physics::Elasticity::StandardTensors<dim>::I +
                2 * parameters.viscosity * sym_grad_v[q];
              for (unsigned int i = 0; i < dim; ++i)
                {
                  for (unsigned int j = 0; j < dim; ++j)
                    {
                      quad_stress[i][j][q] = sigma[i][j];
                    }
                }
            }

          for (unsigned int i = 0; i < dim; ++i)
            {
              for (unsigned int j = 0; j < dim; ++j)
                {
                  qpt_to_dof.vmult(cell_stress[i][j], quad_stress[i][j]);
                  scalar_cell->distribute_local_to_global(cell_stress[i][j],
                                                          stress[i][j]);
                }
            }
          scalar_cell->distribute_local_to_global(local_sorrounding_cells,
                                                  surrounding_cells);
        }
      surrounding_cells.compress(VectorOperation::add);

      for (unsigned int i = 0; i < dim; ++i)
        {
          for (unsigned int j = 0; j < dim; ++j)
            {
              stress[i][j].compress(VectorOperation::add);
              const unsigned int local_begin =
                surrounding_cells.local_range().first;
              const unsigned int local_end =
                surrounding_cells.local_range().second;
              for (unsigned int k = local_begin; k < local_end; ++k)
                {
                  stress[i][j][k] /= surrounding_cells[k];
                }
              stress[i][j].compress(VectorOperation::insert);
            }
        }
    }

    template <int dim>
    FluidSolver<dim>::Field::Field(const Field &source)
      : Function<dim>(dim + 1), value_function(source.value_function)
    {
    }

    template <int dim>
    FluidSolver<dim>::Field::Field(
      const std::function<double(
        const Point<dim> &, const unsigned int, const double)> &value_function)
      : Function<dim>(dim + 1), value_function(value_function)
    {
    }

    template <int dim>
    double FluidSolver<dim>::Field::value(const Point<dim> &p,
                                          const unsigned int component) const
    {
      return value_function(p, component, this->get_time());
    }

    template <int dim>
    void FluidSolver<dim>::Field::vector_value(const Point<dim> &p,
                                               Vector<double> &values) const
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        {
          values(c) = value_function(p, c, this->get_time());
        }
    }

    template <int dim>
    void FluidSolver<dim>::Field::double_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<double> &values,
      const unsigned int component)
    {
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          values[i] = this->value(points[i], component);
        }
    }

    template <int dim>
    void FluidSolver<dim>::Field::tensor_value_list(
      const std::vector<Point<dim>> &points,
      std::vector<Tensor<1, dim>> &values)
    {
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          for (unsigned int c = 0; c < dim; ++c)
            {
              values[i][c] = this->value(points[i], c);
            }
        }
    }

    template class FluidSolver<2>;
    template class FluidSolver<3>;
  } // namespace MPI
} // namespace Fluid

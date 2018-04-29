#include "mpi_linearelasticity.h"

namespace Solid
{
  using namespace dealii;

  template <int dim>
  ParallelLinearElasticity<dim>::ParallelLinearElasticity(
    parallel::distributed::Triangulation<dim> &tria,
    const Parameters::AllParameters &parameters)
    : material(parameters.E, parameters.nu, parameters.solid_rho),
      gamma(0.5 + parameters.damping),
      beta(gamma / 2),
      degree(parameters.solid_degree),
      tolerance(1e-12),
      triangulation(tria),
      fe(FE_Q<dim>(degree), dim),
      dof_handler(triangulation),
      volume_quad_formula(degree + 1),
      face_quad_formula(degree + 1),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval),
      parameters(parameters),
      mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      timer(
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
  {
  }

  template <int dim>
  void ParallelLinearElasticity<dim>::setup_dofs()
  {
    TimerOutput::Scope timer_section(timer, "Setup system");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    // Extract the locally owned and relevant dofs
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // The Dirichlet boundary conditions are stored in the ConstraintMatrix
    // object. It does not need to modify the sparse matrix after assembly,
    // because it is applied in the assembly process,
    // therefore is better compared with apply_boundary_values approach.
    // Note that ZeroFunction is used here for convenience. In more
    // complicated applications, write a BoundaryValue class to replace it.

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
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
          << triangulation.n_global_active_cells() << std::endl
          << "  Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
  }

  template <int dim>
  void ParallelLinearElasticity<dim>::initialize_system()
  {
    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

    SparsityTools::distribute_sparsity_pattern(
      dsp,
      dof_handler.n_locally_owned_dofs_per_processor(),
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(
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
  }

  template <int dim>
  void ParallelLinearElasticity<dim>::assemble_system(const bool is_initial)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    system_matrix = 0;
    stiffness_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values);

    const SymmetricTensor<4, dim> elasticity = material.get_elasticity();
    const double rho = material.get_density();
    const double dt = time.get_delta_t();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_f_q_points = face_quad_formula.size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_stiffness(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // The symmetric gradients of the displacement shape functions at a certain
    // point.
    // There are dofs_per_cell shape functions so the size is dofs_per_cell.
    std::vector<SymmetricTensor<2, dim>> symmetric_grad_phi(dofs_per_cell);
    // The shape functions at a certain point.
    std::vector<Tensor<1, dim>> phi(dofs_per_cell);
    // A "viewer" to describe the nodal dofs as a vector.
    FEValuesExtractors::Vector displacements(0);

    // Loop over cells
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        // Only operates on the locally owned cells
        if (cell->is_locally_owned())
          {
            local_matrix = 0;
            local_stiffness = 0;
            local_rhs = 0;

            fe_values.reinit(cell);

            // Loop over quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // Loop over the dofs once, to calculate the grad_ph_u
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    symmetric_grad_phi[k] =
                      fe_values[displacements].symmetric_gradient(k, q);
                    phi[k] = fe_values[displacements].value(k, q);
                  }
                // Loop over the dofs again, to assemble
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        if (is_initial)
                          {
                            local_matrix[i][j] +=
                              rho * phi[i] * phi[j] * fe_values.JxW(q);
                          }
                        else
                          {
                            local_matrix[i][j] +=
                              (rho * phi[i] * phi[j] +
                               symmetric_grad_phi[i] * elasticity *
                                 symmetric_grad_phi[j] * beta * dt * dt) *
                              fe_values.JxW(q);
                            local_stiffness[i][j] +=
                              symmetric_grad_phi[i] * elasticity *
                              symmetric_grad_phi[j] * fe_values.JxW(q);
                          }
                      }
                    // zero body force
                    Tensor<1, dim> gravity;
                    local_rhs[i] += phi[i] * gravity * rho * fe_values.JxW(q);
                  }
              }

            cell->get_dof_indices(local_dof_indices);

            // Traction or Pressure
            for (unsigned int face = 0;
                 face < GeometryInfo<dim>::faces_per_cell;
                 ++face)
              {
                if (cell->face(face)->at_boundary())
                  {
                    unsigned int id = cell->face(face)->boundary_id();
                    if (parameters.solid_neumann_bcs.find(id) !=
                        parameters.solid_neumann_bcs.end())
                      {
                        std::vector<double> value =
                          parameters.solid_neumann_bcs[id];
                        Tensor<1, dim> traction;
                        if (parameters.solid_neumann_bc_type == "Traction")
                          {
                            for (unsigned int i = 0; i < dim; ++i)
                              {
                                traction[i] = value[i];
                              }
                          }

                        fe_face_values.reinit(cell, face);
                        for (unsigned int q = 0; q < n_f_q_points; ++q)
                          {
                            if (parameters.solid_neumann_bc_type == "Pressure")
                              {
                                // The normal is w.r.t. reference configuration!
                                traction = fe_face_values.normal_vector(q);
                                traction *= value[0];
                              }
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const unsigned int component_j =
                                  fe.system_to_component_index(j).first;
                                // +external force
                                local_rhs(j) +=
                                  fe_face_values.shape_value(j, q) *
                                  traction[component_j] * fe_face_values.JxW(q);
                              }
                          }
                      }
                  }
              }

            // Now distribute local data to the system, and apply the
            // hanging node constraints at the same time.
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
            constraints.distribute_local_to_global(
              local_stiffness, local_dof_indices, stiffness_matrix);
          }
      }
    // Synchronize with other processors.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
  }

  // Solve linear system \f$Ax = b\f$ using CG solver.
  template <int dim>
  std::pair<unsigned int, double> ParallelLinearElasticity<dim>::solve(
    const PETScWrappers::MPI::SparseMatrix &A,
    PETScWrappers::MPI::Vector &x,
    const PETScWrappers::MPI::Vector &b)
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    SolverControl solver_control(dof_handler.n_dofs(), tolerance);

    PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

    PETScWrappers::PreconditionBlockJacobi preconditioner(A);

    cg.solve(A, x, b, preconditioner);
    constraints.distribute(x);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void ParallelLinearElasticity<dim>::output_results(
    const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");
    pcout << "Writing solid results..." << std::endl;

    std::vector<std::string> solution_names(dim, "displacements");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    // DataOut needs more than locally owned dofs, so we have to construct a
    // ghosted vector to store the solution.
    PETScWrappers::MPI::Vector solution(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    solution = current_displacement;

    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      {
        subdomain(i) = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    std::string basename =
      "linearelasticity-" + Utilities::int_to_string(output_index, 6) + "-";

    std::string filename =
      basename +
      Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
      ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    // Processor 0 writes the pvd file that tells ParaView filenames and time.
    static std::vector<std::pair<double, std::string>> times_and_names;
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
        std::ofstream pvd_output("linearelasticity.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

  template <int dim>
  void
  ParallelLinearElasticity<dim>::refine_mesh(const unsigned int min_grid_level,
                                             const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    pcout << "Refining mesh..." << std::endl;

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    // In order to estimate error, the distributed vector must be ghosted.
    PETScWrappers::MPI::Vector solution;
    solution.reinit(
      locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    solution = current_displacement;
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       face_quad_formula,
                                       typename FunctionMap<dim>::type(),
                                       solution,
                                       estimated_error_per_cell);

    // Set the refine and coarsen flag
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

    // Prepare to transfer previous solutions
    std::vector<
      parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
      trans(3,
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
  void ParallelLinearElasticity<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    triangulation.refine_global(parameters.global_refinement);
    setup_dofs();
    initialize_system();

    std::cout.precision(6);
    std::cout.width(12);

    // Neet to compute the initial acceleration, \f$ Ma_n = F \f$,
    // at this point set system_matrix to mass_matrix.
    assemble_system(true);
    solve(system_matrix, previous_acceleration, system_rhs);

    // Update the system_matrix
    assemble_system(false);

    const double dt = time.get_delta_t();

    // Temporary vectors used to solve the system
    PETScWrappers::MPI::Vector tmp1(locally_owned_dofs, mpi_communicator);
    PETScWrappers::MPI::Vector tmp2(locally_owned_dofs, mpi_communicator);
    PETScWrappers::MPI::Vector tmp3(locally_owned_dofs, mpi_communicator);

    // Time loop
    output_results(time.get_timestep());
    while (time.end() - time.current() > 1e-12)
      {
        time.increment();
        pcout << std::string(91, '*') << std::endl
              << "Time step = " << time.get_timestep()
              << ", at t = " << std::scientific << time.current() << std::endl;

        // Modify the RHS
        tmp1 = system_rhs;
        tmp2 = previous_displacement;
        tmp2.add(
          dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);
        stiffness_matrix.vmult(tmp3, tmp2);
        tmp1 -= tmp3;

        auto state = solve(system_matrix, current_acceleration, tmp1);

        // update the current velocity
        // \f$ v_{n+1} = v_n + (1-\gamma)\Delta{t}a_n + \gamma\Delta{t}a_{n+1}
        // \f$
        current_velocity = previous_velocity;
        current_velocity.add(dt * (1 - gamma),
                             previous_acceleration,
                             dt * gamma,
                             current_acceleration);

        // update the current displacement
        current_displacement = previous_displacement;
        current_displacement.add(dt, previous_velocity);
        current_displacement.add(dt * dt * (0.5 - beta),
                                 previous_acceleration,
                                 dt * dt * beta,
                                 current_acceleration);

        // update the previous values
        previous_acceleration = current_acceleration;
        previous_velocity = current_velocity;
        previous_displacement = current_displacement;

        pcout << std::scientific << std::left
              << " CG iteration: " << std::setw(3) << state.first
              << " CG residual: " << state.second << std::endl;

        if (time.time_to_output())
          {
            output_results(time.get_timestep());
          }

        if (time.time_to_refine())
          {
            refine_mesh(1, 4);
            tmp1.reinit(locally_owned_dofs, mpi_communicator);
            tmp2.reinit(locally_owned_dofs, mpi_communicator);
            tmp3.reinit(locally_owned_dofs, mpi_communicator);
            assemble_system(false);
          }
      }
  }

  template class ParallelLinearElasticity<2>;
  template class ParallelLinearElasticity<3>;
} // namespace Solid

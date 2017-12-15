#include "linearElasticSolver.h"

namespace Solid
{
  using namespace dealii;

  template <int dim>
  LinearElasticSolver<dim>::LinearElasticSolver(
    Triangulation<dim> &tria, const Parameters::AllParameters &parameters)
    : material(parameters.E, parameters.nu, parameters.rho),
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
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
      parameters(parameters)
  {
  }

  template <int dim>
  void LinearElasticSolver<dim>::setup_dofs()
  {
    TimerOutput::Scope timer_section(timer, "Setup system");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

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

    std::cout << "  Number of active solid cells: "
              << triangulation.n_active_cells() << std::endl
              << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
  }

  template <int dim>
  void LinearElasticSolver<dim>::initialize_system()
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    pattern.copy_from(dsp);

    system_matrix.reinit(pattern);
    stiffness_matrix.reinit(pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void LinearElasticSolver<dim>::assemble_system(const bool is_initial)
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
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
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
                            local_rhs(j) += fe_face_values.shape_value(j, q) *
                                            traction[component_j] *
                                            fe_face_values.JxW(q);
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

  // Solve linear system \f$Ax = b\f$ using CG solver.
  template <int dim>
  std::pair<unsigned int, double> LinearElasticSolver<dim>::solve(
    const SparseMatrix<double> &A, Vector<double> &x, const Vector<double> &b)
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    SolverControl solver_control(A.m(), tolerance);
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);

    cg.solve(A, x, b, preconditioner);
    constraints.distribute(x);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void LinearElasticSolver<dim>::output_results(
    const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    std::cout << "Writing solid results..." << std::endl;
    std::vector<std::string> solution_names(dim, "displacements");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_displacement,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::string basename = "linearelastic";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  template <int dim>
  void LinearElasticSolver<dim>::refine_mesh(const unsigned int min_grid_level,
                                             const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       face_quad_formula,
                                       typename FunctionMap<dim>::type(),
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

    std::vector<SolutionTransfer<dim>> solution_trans(
      3, SolutionTransfer<dim>(dof_handler));
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

  template <int dim>
  void LinearElasticSolver<dim>::run()
  {
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

    // Time loop
    output_results(time.get_timestep());
    while (time.end() - time.current() > 1e-12)
      {
        time.increment();
        std::cout << std::string(91, '*') << std::endl
                  << "Time step = " << time.get_timestep()
                  << ", at t = " << std::scientific << time.current()
                  << std::endl;

        // Modify the RHS
        Vector<double> tmp1(system_rhs);
        auto tmp2 = previous_displacement;
        tmp2.add(
          dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);
        Vector<double> tmp3(dof_handler.n_dofs());
        stiffness_matrix.vmult(tmp3, tmp2);
        tmp1 -= tmp3;

        auto state = solve(system_matrix, current_acceleration, tmp1);

        // update the current velocity
        // \f$ v_{n+1} = v_n + (1-\gamma)\Delta{t}a_n + \gamma\Delta{t}a_{n+1}
        // \f$
        current_velocity = previous_velocity;
        current_velocity.add(dt * (1 - gamma), previous_acceleration);
        current_velocity.add(dt * gamma, current_acceleration);

        // update the current displacement
        current_displacement = previous_displacement;
        current_displacement.add(dt, previous_velocity);
        current_displacement.add(dt * dt * (0.5 - beta), previous_acceleration);
        current_displacement.add(dt * dt * beta, current_acceleration);

        // update the previous values
        previous_acceleration = current_acceleration;
        previous_velocity = current_velocity;
        previous_displacement = current_displacement;

        std::cout << std::scientific << std::left
                  << " CG iteration: " << std::setw(3) << state.first
                  << " CG residual: " << state.second << std::endl;

        if (time.time_to_output())
          {
            output_results(time.get_timestep());
          }

        if (time.time_to_refine())
          {
            refine_mesh(1, 4);
            assemble_system(false);
          }
      }
  }

  template class LinearElasticSolver<2>;
  template class LinearElasticSolver<3>;
}

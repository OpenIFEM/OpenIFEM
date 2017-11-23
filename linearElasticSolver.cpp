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
      tolerance(parameters.tol_f),
      triangulation(tria),
      fe(FE_Q<dim>(degree), dim),
      dof_handler(triangulation),
      volume_quad_formula(degree + 1),
      face_quad_formula(degree + 1),
      time(
        parameters.end_time, parameters.time_step, parameters.output_interval),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {
  }

  template <int dim>
  void LinearElasticSolver<dim>::setup_dofs()
  {
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
    VectorTools::interpolate_boundary_values(
      dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraints);
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
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());
  }

  /**
   * system_matrix: \f$A = \frac{M}{\beta{\Delta{t}}^2} + K\f$;
   * system_rhs:
   * \f$F + \frac{M}{\beta{\Delta{t}}^2}[u_n + \Delta{t}v_n + (0.5-\beta){\Delta
   * t}^2a_n\f$
   * Solving the linear system with constraints yields \f$u_{n+1}\f$.
   * Then one can calculate the acceleration and velocity using
   * \f$a_{n+1} = \frac{1}{\beta{\Delta{t}}^2}
   *              [u_{n+1} - u_n - \Delta{t}v_n - {\Delta{t}}^2(0.5 -
   * \beta)a_n]\f$
   * \f$v_{n+1} = v_n + \Delta{t}[(1-\gamma)a_n + \gamma a_{n+1}]\f$ and
   */
  template <int dim>
  void LinearElasticSolver<dim>::assemble_system(const bool assemble_lhs,
                                                 const bool is_initial)
  {
    if (assemble_lhs)
      {
        system_matrix = 0;
      }
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
    const double dt = time.get_delta_t();
    const double rho = material.get_density();

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_f_q_points = face_quad_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // The symmetric gradients of the displacement shape functions at a certain
    // point.
    // There are dofs_per_cell shape functions so the size is dofs_per_cell.
    std::vector<SymmetricTensor<2, dim>> symmetric_grad_phi(dofs_per_cell);
    // The shape functions at a certain point.
    std::vector<Tensor<1, dim>> phi(dofs_per_cell);
    // The displacement, velocity, acceleration values.
    std::vector<Tensor<1, dim>> displacement_values(n_q_points);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    std::vector<Tensor<1, dim>> acceleration_values(n_q_points);

    // A "viewer" to describe the nodal dofs as a vector.
    FEValuesExtractors::Vector displacements(0);

    // Loop over cells
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);

        fe_values[displacements].get_function_values(previous_displacement,
                                                     displacement_values);
        fe_values[displacements].get_function_values(previous_velocity,
                                                     velocity_values);
        fe_values[displacements].get_function_values(previous_acceleration,
                                                     acceleration_values);

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
                    // rhs involves matrix-vector multiplication, we must map
                    // dof to
                    // the vector component.
                    if (assemble_lhs)
                      {
                        if (is_initial)
                          {
                            cell_matrix[i][j] += rho * phi[i] * phi[j] * fe_values.JxW(q);
                          }
                        else
                          {
                            cell_matrix[i][j] +=
                              (rho * phi[i] * phi[j] +
                               symmetric_grad_phi[i] * elasticity *
                                 symmetric_grad_phi[j] * beta * dt * dt) *
                              fe_values.JxW(q);
                          }
                      }
                    // rhs due to time discretization
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;
                    cell_rhs[i] -= symmetric_grad_phi[i] * elasticity *
                                   symmetric_grad_phi[j] *
                                   (displacement_values[q][component_j] +
                                    dt * velocity_values[q][component_j] +
                                    (0.5 - beta) * dt * dt *
                                      acceleration_values[q][component_j]) *
                                   fe_values.JxW(q);
                  }
                // body force
                Tensor<1, dim> gravity;
                cell_rhs[i] += phi[i] * gravity * rho * fe_values.JxW(q);
              }
          }

        // traction
        // TODO: use FEValueExtractors
        Tensor<1, dim> traction;
        traction[dim - 1] = -1e0;
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == 3)
              {
                fe_face_values.reinit(cell, f);
                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        cell_rhs[i] += traction[component_i] *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                      }
                  }
              }
          }

        // Now distribute local data to the system, and take care of the
        // Dirchlet
        // boundary conditions at the same time.
        cell->get_dof_indices(local_dof_indices);
        if (assemble_lhs)
          {
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
        else
          {
            constraints.distribute_local_to_global(
              cell_rhs, local_dof_indices, system_rhs);
          }
      }
  }

  // In Newmark-beta method, we solve for the acceleration.
  template <int dim>
  std::pair<unsigned int, double> LinearElasticSolver<dim>::solve()
  {
    SolverControl solver_control(system_matrix.m(), tolerance);
    SolverCG<> cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, current_acceleration, system_rhs, preconditioner);
    constraints.distribute(current_acceleration);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void LinearElasticSolver<dim>::output_results(
    const unsigned int output_index) const
  {
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

  template <int dim>
  void LinearElasticSolver<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       current_acceleration,
                                       estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, Vector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(
      current_acceleration);
    triangulation.execute_coarsening_and_refinement();
    setup_dofs();
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(current_acceleration, tmp);
    constraints.distribute(tmp);
    initialize_system();
    current_acceleration = tmp;
  }

  template <int dim>
  void LinearElasticSolver<dim>::run()
  {
    triangulation.refine_global(2);
    setup_dofs();
    initialize_system();

    std::cout.precision(6);
    std::cout.width(12);

    // Neet to compute the initial acceleration
    assemble_system(true, true);
    auto state = solve();
    previous_acceleration = current_acceleration;
    current_acceleration = 0;

    std::cout << std::scientific << std::left << "Initialization..."
              << std::endl
              << " CG iteration: " << std::setw(3) << state.first
              << " CG residual: " << state.second << std::endl;

    output_results(time.get_timestep());
    while (time.current() < time.end())
      {
        time.increment();
        std::cout << std::string(91, '*') << std::endl
                  << "Time step = " << time.get_timestep()
                  << ", at t = " << std::scientific << time.current()
                  << std::endl;

        // if (time.get_timestep() != 1)
        //  refine_mesh();

        // solve for the new acceleration, namely current_acceleration
        assemble_system(true, false);
        auto state = solve();

        const double dt = time.get_delta_t();

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
      }
  }

  template class LinearElasticSolver<2>;
  template class LinearElasticSolver<3>;
}

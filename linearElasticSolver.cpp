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
    TimerOutput::Scope timer_section(timer, "Setup system");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    // The Dirichlet boundary conditions are stored in the ConstraintMatrix
    // object. It does not need to modify the sparse matrix after assembly,
    // because it is applied in the assembly process,
    // therefore is better compared with apply_boundary_values approach.
    // Note that ZeroFunction is used here for convenience. In more
    // complicated applications, write a BoundaryValue class to replace it.

    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
    hanging_node_constraints.close();

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);

    std::cout << "  Number of active solid cells: "
              << triangulation.n_active_cells() << std::endl
              << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
  }

  template <int dim>
  void LinearElasticSolver<dim>::initialize_system()
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints);
    pattern.copy_from(dsp);

    system_matrix.reinit(pattern);
    stiffness_matrix.reinit(pattern);
    mass_matrix.reinit(pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void LinearElasticSolver<dim>::assemble_system()
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    stiffness_matrix = 0;
    mass_matrix = 0;
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

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_f_q_points = face_quad_formula.size();

    FullMatrix<double> local_stiffness (dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass (dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs (dofs_per_cell);

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
        local_mass = 0;
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
                    local_mass[i][j] += rho * phi[i] * phi[j] * fe_values.JxW(q);
                    local_stiffness[i][j] += symmetric_grad_phi[i] * elasticity *
                      symmetric_grad_phi[j] * fe_values.JxW(q);
                  }
                  // body force
                  Tensor<1, dim> gravity;
                  local_rhs[i] += phi[i] * gravity * rho * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);

        // traction
        // TODO: use FEValueExtractors
        Tensor<1, dim> traction;
        traction[1] = -1e-5;
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
                        local_rhs[i] += traction[component_i] *
                                       fe_face_values.shape_value(i, q) *
                                       fe_face_values.JxW(q);
                      }
                  }
              }
          }

        // Now distribute local data to the system, and apply the
        // hanging node constraints at the same time.
        hanging_node_constraints.distribute_local_to_global(local_stiffness,
                                                            local_rhs,
                                                            local_dof_indices,
                                                            stiffness_matrix,
                                                            system_rhs);
        hanging_node_constraints.distribute_local_to_global(local_mass,
                                                            local_dof_indices,
                                                            mass_matrix);
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
    hanging_node_constraints.distribute(x);

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
  void LinearElasticSolver<dim>::run()
  {
    triangulation.refine_global(2);
    setup_dofs();
    initialize_system();

    std::cout.precision(6);
    std::cout.width(12);

    // Assemble system only once
    assemble_system();

    // Neet to compute the initial acceleration, \f$ Ma_n = F \f$, 
    // at this point set system_matrix to mass_matrix.
    system_matrix.copy_from(mass_matrix);
    // apply Dirichlet boundary conditions
    MatrixTools::apply_boundary_values(boundary_values, 
                                       system_matrix,
                                       previous_acceleration,
                                       system_rhs);
    solve(system_matrix, previous_acceleration, system_rhs);
    
    // Next add the stiffness to the system_matrix
    const double dt = time.get_delta_t();
    system_matrix.add(beta*dt*dt, stiffness_matrix);

    Vector<double> tmp1 (dof_handler.n_dofs());
    Vector<double> tmp2 (dof_handler.n_dofs());
    Vector<double> tmp3 (system_rhs); // Cache the system_rhs

    // Time loop
    output_results(time.get_timestep());
    while (time.current() < time.end())
      {
        time.increment();
        std::cout << std::string(91, '*') << std::endl
                  << "Time step = " << time.get_timestep()
                  << ", at t = " << std::scientific << time.current()
                  << std::endl;

        // Modify the rhs to account for the time discretization
        tmp1 = 0;
        tmp1 += previous_displacement;
        tmp1.add(dt, previous_velocity, dt*dt*(0.5-beta), previous_acceleration);

        tmp2 = 0;
        stiffness_matrix.vmult(tmp2, tmp1);
        system_rhs -= tmp2;

        // Solve for the new acceleration, namely current_acceleration
        // apply Dirichlet boundary conditions, again, this is inefficient.
        MatrixTools::apply_boundary_values(boundary_values, 
                                           system_matrix,
                                           current_acceleration,
                                           system_rhs);
        auto state = solve(system_matrix, current_acceleration, system_rhs);
        // Reset system_rhs
        system_rhs = tmp3;

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

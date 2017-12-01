#include "hyperelasticSolver.h"

namespace
{
  using namespace dealii;

  template <int dim>
  void PointHistory<dim>::setup(const Parameters::AllParameters &parameters)
  {
    if (parameters.solid_type == "NeoHookean")
      {
        Assert(parameters.C.size() >= 2, ExcInternalError());
        material.reset(new Solid::NeoHookean<dim>(
          parameters.C[0], parameters.C[1], parameters.rho));
        update(parameters, Tensor<2, dim>());
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void PointHistory<dim>::update(const Parameters::AllParameters &parameters,
                                 const Tensor<2, dim> &Grad_u)
  {
    const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u);
    material->update_data(F);
    F_inv = invert(F);
    if (parameters.solid_type == "NeoHookean")
      {
        auto nh = std::dynamic_pointer_cast<Solid::NeoHookean<dim>>(material);
        Assert(nh, ExcInternalError());
        tau = nh->get_tau();
        Jc = nh->get_Jc();
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
    dPsi_vol_dJ = material->get_dPsi_vol_dJ();
    d2Psi_vol_dJ2 = material->get_d2Psi_vol_dJ2();
  }
}

namespace Solid
{
  using namespace dealii;

  template <int dim>
  HyperelasticSolver<dim>::HyperelasticSolver(
    Triangulation<dim> &tria, const Parameters::AllParameters &params)
    : parameters(params),
      vol(0.),
      time(
        parameters.end_time, parameters.time_step, parameters.output_interval),
      timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
      degree(parameters.solid_degree),
      fe(FE_Q<dim>(degree), dim),
      triangulation(tria),
      dof_handler(triangulation),
      dofs_per_cell(fe.dofs_per_cell),
      volume_quad_formula(degree + 1),
      face_quad_formula(degree + 1),
      n_q_points(volume_quad_formula.size()),
      n_f_q_points(face_quad_formula.size()),
      displacement(0),
      gamma(0.5 + parameters.damping),
      beta(gamma / 2)
  {
  }

  template <int dim>
  void HyperelasticSolver<dim>::run()
  {
    triangulation.refine_global(3);
    vol = GridTools::volume(triangulation);
    std::cout << "Grid:\n\t Reference volume: " << vol << std::endl;

    // The boundary id is hardcoded for now
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->center()[1] == 1.0 * 0.001)
              {
                if (dim == 3)
                  {
                    if (cell->face(face)->center()[0] < 0.5 * 0.001 &&
                        cell->face(face)->center()[2] < 0.5 * 0.001)
                      {
                        cell->face(face)->set_boundary_id(6);
                      }
                  }
                else
                  {
                    if (cell->face(face)->center()[0] < 0.5 * 0.001)
                      {
                        cell->face(face)->set_boundary_id(6);
                      }
                  }
              }
          }
      }

    setup_dofs();
    initialize_system();

    // Solve for the initial acceleration
    assemble(true);
    solve_linear_system(mass_matrix, previous_acceleration, system_rhs);
    output_results(time.get_timestep());

    Vector<double> predicted_displacement(dof_handler.n_dofs());
    Vector<double> newton_update(dof_handler.n_dofs());
    Vector<double> tmp(dof_handler.n_dofs());

    while (time.end() - time.current() > 1e-12)
      {
        time.increment();

        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ "
                  << time.current() << "s" << std::endl;

        // Reset the errors, iteration counter, and the solution increment
        newton_update = 0;
        unsigned int newton_iteration = 0;
        error_residual = 1.0;
        initial_error_residual = 1.0;
        normalized_error_residual = 1.0;
        error_update = 1.0;
        initial_error_update = 1.0;
        normalized_error_update = 1.0;
        const double dt = time.get_delta_t();

        // The prediction of the current displacement,
        // which is what we want to solve.
        predicted_displacement = previous_displacement;
        predicted_displacement.add(
          dt, previous_velocity, (0.5 - beta) * dt * dt, previous_acceleration);

        std::cout << std::string(100, '_') << std::endl;

        while (normalized_error_update > parameters.tol_d ||
               normalized_error_residual > parameters.tol_f)
          {
            AssertThrow(newton_iteration < parameters.solid_max_iterations,
                        ExcMessage("Too many Newton iterations!"));

            // Compute the displacement, velocity and acceleration
            current_acceleration = current_displacement;
            current_acceleration -= predicted_displacement;
            current_acceleration /= (beta * dt * dt);
            current_velocity = previous_velocity;
            current_velocity.add(dt * (1 - gamma),
                                 previous_acceleration,
                                 dt * gamma,
                                 current_acceleration);

            // Assemble the system, and modify the RHS to account for
            // the time-discretization.
            assemble(false);
            mass_matrix.vmult(tmp, current_acceleration);
            system_rhs -= tmp;

            // Solve linear system
            const std::pair<unsigned int, double> lin_solver_output =
              solve_linear_system(system_matrix, newton_update, system_rhs);

            // Error evaluation
            {
              get_error_residual(error_residual);
              if (newton_iteration == 0)
                {
                  initial_error_residual = error_residual;
                }
              normalized_error_residual =
                error_residual / initial_error_residual;

              get_error_update(newton_update, error_update);
              if (newton_iteration == 0)
                {
                  initial_error_update = error_update;
                }
              normalized_error_update = error_update / initial_error_update;
            }

            current_displacement += newton_update;
            // Update the quadrature point history with the newest displacement
            update_qph(current_displacement);

            std::cout << "Newton iteration = " << newton_iteration
                      << ", CG itr = " << lin_solver_output.first << std::fixed
                      << std::setprecision(3) << std::setw(7) << std::scientific
                      << ", CG res = " << lin_solver_output.second
                      << ", res_F = " << error_residual
                      << ", res_U = " << error_update << std::endl;

            newton_iteration++;
          }
        // Once converged, update current acceleration and velocity again.
        current_acceleration = current_displacement;
        current_acceleration -= predicted_displacement;
        current_acceleration /= (beta * dt * dt);
        current_velocity = previous_velocity;
        current_velocity.add(dt * (1 - gamma),
                             previous_acceleration,
                             dt * gamma,
                             current_acceleration);
        // Update the previous values
        previous_acceleration = current_acceleration;
        previous_velocity = current_velocity;
        previous_displacement = current_displacement;

        std::cout << std::string(100, '_') << std::endl
                  << "Relative errors:" << std::endl
                  << "Displacement:\t" << normalized_error_update << std::endl
                  << "Force: \t\t" << normalized_error_residual << std::endl;

        if (time.time_to_output())
          {
            output_results(time.get_timestep());
          }
      }
  }

  template <int dim>
  void HyperelasticSolver<dim>::setup_dofs()
  {
    timer.enter_subsection("Setup system");
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    const FEValuesExtractors::Scalar z_displacement(2);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             fe.component_mask(x_displacement));

    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints,
                                             fe.component_mask(y_displacement));

    if (dim == 2) // 2D case: 0-x, 2-y, 3-x, 6-x
      {
        VectorTools::interpolate_boundary_values(
          dof_handler,
          3,
          Functions::ZeroFunction<dim>(dim),
          constraints,
          fe.component_mask(x_displacement));

        VectorTools::interpolate_boundary_values(
          dof_handler,
          6,
          Functions::ZeroFunction<dim>(dim),
          constraints,
          fe.component_mask(x_displacement));
      }
    else // 3D case: 0-x, 2-y, 4-z, 3-xz, 6-xz
      {
        VectorTools::interpolate_boundary_values(
          dof_handler,
          4,
          Functions::ZeroFunction<dim>(dim),
          constraints,
          fe.component_mask(z_displacement));

        VectorTools::interpolate_boundary_values(
          dof_handler,
          3,
          Functions::ZeroFunction<dim>(dim),
          constraints,
          (fe.component_mask(x_displacement) |
           fe.component_mask(z_displacement)));

        VectorTools::interpolate_boundary_values(
          dof_handler,
          6,
          Functions::ZeroFunction<dim>(dim),
          constraints,
          (fe.component_mask(x_displacement) |
           fe.component_mask(z_displacement)));
      }
    constraints.close();

    timer.leave_subsection();
  }

  template <int dim>
  void HyperelasticSolver<dim>::initialize_system()
  {
    system_matrix.clear();
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    pattern.copy_from(dsp);
    system_matrix.reinit(pattern);
    mass_matrix.reinit(pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    current_acceleration.reinit(dof_handler.n_dofs());
    current_velocity.reinit(dof_handler.n_dofs());
    current_displacement.reinit(dof_handler.n_dofs());
    previous_acceleration.reinit(dof_handler.n_dofs());
    previous_velocity.reinit(dof_handler.n_dofs());
    previous_displacement.reinit(dof_handler.n_dofs());
    setup_qph();
  }

  template <int dim>
  void HyperelasticSolver<dim>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quad_point_history.initialize(
      triangulation.begin_active(), triangulation.end(), n_q_points);
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          quad_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            lqph[q]->setup(parameters);
          }
      }
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::update_qph(const Vector<double> &evaluation_point)
  {
    timer.enter_subsection("Update QPH data");

    // displacement gradient at quad points
    std::vector<Tensor<2, dim>> grad_u(volume_quad_formula.size());
    FEValues<dim> fe_values(
      fe, volume_quad_formula, update_values | update_gradients);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          quad_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        fe_values.reinit(cell);
        fe_values[displacement].get_function_gradients(evaluation_point,
                                                       grad_u);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            lqph[q]->update(parameters, grad_u[q]);
          }
      }
    timer.leave_subsection();
  }

  template <int dim>
  double HyperelasticSolver<dim>::compute_volume() const
  {
    double volume = 0.0;
    FEValues<dim> fe_values(fe, volume_quad_formula, update_JxW_values);
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        fe_values.reinit(cell);
        const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
          quad_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double det = lqph[q]->get_det_F();
            const double JxW = fe_values.JxW(q);
            volume += det * JxW;
          }
      }
    Assert(volume > 0.0, ExcInternalError());
    return volume;
  }

  template <int dim>
  void HyperelasticSolver<dim>::get_error_residual(double &error_residual)
  {
    Vector<double> res(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (!constraints.is_constrained(i))
          {
            res(i) = system_rhs(i);
          }
      }
    error_residual = res.l2_norm();
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::get_error_update(const Vector<double> &newton_update,
                                            double &error_update)
  {
    Vector<double> error(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (!constraints.is_constrained(i))
          {
            error(i) = newton_update(i);
          }
      }
    error_update = error.l2_norm();
  }

  template <int dim>
  void HyperelasticSolver<dim>::assemble(bool initial_step)
  {
    timer.enter_subsection("Assemble tangent matrix");

    if (initial_step)
      {
        mass_matrix = 0.0;
      }
    system_matrix = 0.0;
    system_rhs = 0.0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_gradients |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_JxW_values);

    std::vector<std::vector<Tensor<1, dim>>> phi(
      n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>> grad_phi(
      n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi(
      n_q_points, std::vector<SymmetricTensor<2, dim>>(dofs_per_cell));

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        local_mass = 0;
        local_matrix = 0;
        local_rhs = 0;

        const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
          quad_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<2, dim> F_inv = lqph[q]->get_F_inv();
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                phi[q][k] = fe_values[displacement].value(k, q);
                grad_phi[q][k] = fe_values[displacement].gradient(k, q) * F_inv;
                sym_grad_phi[q][k] = symmetrize(grad_phi[q][k]);
              }

            const SymmetricTensor<2, dim> tau = lqph[q]->get_tau();
            const SymmetricTensor<4, dim> Jc = lqph[q]->get_Jc();
            const double rho = lqph[q]->get_density();
            const double dt = time.get_delta_t();
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                for (unsigned int j = 0; j <= i; ++j)
                  {
                    if (initial_step)
                      {
                        local_mass(i, j) += rho * phi[q][i] * phi[q][j] * JxW;
                      }
                    else
                      {
                        const unsigned int component_j =
                          fe.system_to_component_index(j).first;
                        local_matrix(i, j) +=
                          (phi[q][i] * phi[q][j] * rho / (beta * dt * dt) +
                           sym_grad_phi[q][i] * Jc * sym_grad_phi[q][j]) *
                          JxW;
                        if (component_i == component_j)
                          {
                            local_matrix(i, j) +=
                              grad_phi[q][i][component_i] * tau *
                              grad_phi[q][j][component_j] * JxW;
                          }
                      }
                  }
                local_rhs(i) -=
                  sym_grad_phi[q][i] * tau * JxW; // -internal force
              }
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              {
                local_matrix(i, j) = local_matrix(j, i);
                if (initial_step)
                  {
                    local_mass(i, j) = local_mass(j, i);
                  }
              }
          }

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            // apply pressure
            if (cell->face(face)->at_boundary() &&
                cell->face(face)->boundary_id() == 6)
              {
                fe_face_values.reinit(cell, face);
                const double p0 = -40 / (0.001 * 0.001);
                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    const Tensor<1, dim> &N = fe_face_values.normal_vector(q);
                    const Tensor<1, dim> traction = p0 * N;
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const unsigned int component_j =
                          fe.system_to_component_index(j).first;
                        const double phi = fe_face_values.shape_value(j, q);
                        const double JxW = fe_face_values.JxW(q);
                        local_rhs(j) += (phi * traction[component_j]) *
                                        JxW; // +external force
                      }
                  }
                break;
              }
          }

        if (initial_step)
          {
            constraints.distribute_local_to_global(local_mass,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   mass_matrix,
                                                   system_rhs);
          }
        else
          {
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
      }

    timer.leave_subsection();
  }

  template <int dim>
  std::pair<unsigned int, double> HyperelasticSolver<dim>::solve_linear_system(
    SparseMatrix<double> &A, Vector<double> &x, Vector<double> &b)
  {
    timer.enter_subsection("Linear solver");

    SolverControl solver_control(A.m(), 1e-6 * b.l2_norm());
    GrowingVectorMemory<Vector<double>> GVM;
    SolverCG<Vector<double>> solver_CG(solver_control, GVM);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(A, 1.2);
    solver_CG.solve(A, x, b, preconditioner);

    constraints.distribute(x);
    timer.leave_subsection();

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::output_results(const unsigned int output_index) const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_names(dim, "displacement");
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_displacement,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<double> soln(current_displacement.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      {
        soln(i) = current_displacement(i);
      }
    // Map the solution to the deformed mesh, not necessary!
    MappingQEulerian<dim> q_mapping(degree, dof_handler, soln);
    data_out.build_patches(q_mapping, degree);

    std::string basename = "hyperelastic";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  template class HyperelasticSolver<2>;
  template class HyperelasticSolver<3>;
}

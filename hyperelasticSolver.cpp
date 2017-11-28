#include "hyperelasticSolver.h"

namespace
{
  template <int dim>
  void PointHistory<dim>::setup(const Parameters::AllParameters &parameters)
  {
    if (parameters.solid_type == "NeoHookean")
      {
        Assert(parameters.C.size() >= 2, dealii::ExcInternalError());
        material.reset(new Solid::NeoHookean<dim>(
          parameters.C[0], parameters.C[1], parameters.rho));
        update(parameters, dealii::Tensor<2, dim>());
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }
  }

  template <int dim>
  void PointHistory<dim>::update(const Parameters::AllParameters &parameters,
                                 const dealii::Tensor<2, dim> &Grad_u)
  {
    const dealii::Tensor<2, dim> F =
      dealii::Physics::Elasticity::Kinematics::F(Grad_u);
    material->update_data(F);
    F_inv = dealii::invert(F);
    if (parameters.solid_type == "NeoHookean")
      {
        auto nh = std::dynamic_pointer_cast<Solid::NeoHookean<dim>>(material);
        Assert(nh, dealii::ExcInternalError());
        tau = nh->get_tau();
        Jc = nh->get_Jc();
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
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
      displacement(0)
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
    output_results(time.get_timestep());
    time.increment();
    Vector<double> solution_delta(dof_handler.n_dofs());
    while (time.current() < time.end())
      {
        solution_delta = 0.0;
        solve_nonlinear_step(solution_delta);
        solution += solution_delta;
        if (time.time_to_output())
          {
            output_results(time.get_timestep());
          }
        time.increment();
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
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());
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
  void HyperelasticSolver<dim>::update_qph(const Vector<double> &solution_delta)
  {
    timer.enter_subsection("Update QPH data");
    std::cout << " UQPH " << std::flush;

    const Vector<double> solution_total(get_total_solution(solution_delta));

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
        fe_values[displacement].get_function_gradients(solution_total, grad_u);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            lqph[q]->update(parameters, grad_u[q]);
          }
      }
    timer.leave_subsection();
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::solve_nonlinear_step(Vector<double> &solution_delta)
  {
    std::cout << std::endl
              << "Timestep " << time.get_timestep() << " @ " << time.current()
              << "s" << std::endl;

    Vector<double> newton_update(dof_handler.n_dofs());

    errorResidual.reset();
    errorResidual0.reset();
    errorResidualNorm.reset();
    errorUpdate.reset();
    errorUpdate0.reset();
    errorUpdateNorm.reset();

    print_conv_header();

    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.solid_max_iterations;
         ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " "
                  << std::flush;

        system_matrix = 0.0;
        system_rhs = 0.0;
        assemble(false);
        get_error_residual(errorResidual);

        if (newton_iteration == 0)
          {
            errorResidual0 = errorResidual;
          }

        errorResidualNorm = errorResidual;
        errorResidualNorm.normalize(errorResidual0);

        if (newton_iteration > 0 && errorUpdateNorm.norm <= parameters.tol_d &&
            errorResidualNorm.norm <= parameters.tol_f)
          {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();
            break;
          }

        assemble(true);

        const std::pair<unsigned int, double> lin_solver_output =
          solve_linear_system(newton_update);

        get_error_update(newton_update, errorUpdate);
        if (newton_iteration == 0)
          {
            errorUpdate0 = errorUpdate;
          }

        errorUpdateNorm = errorUpdate;
        errorUpdateNorm.normalize(errorUpdate0);

        solution_delta += newton_update;
        update_qph(solution_delta);

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "  "
                  << lin_solver_output.second << "  " << errorResidualNorm.norm
                  << "  " << errorResidualNorm.norm << "  "
                  << errorUpdateNorm.norm << "  " << errorUpdateNorm.norm
                  << "  " << std::endl;
      }

    AssertThrow(newton_iteration < parameters.solid_max_iterations,
                ExcMessage("No convergence in nonlinear solver!"));
  }

  template <int dim>
  void HyperelasticSolver<dim>::print_conv_header()
  {
    static const unsigned int width = 100;
    std::string splitter(width, '_');
    std::cout << splitter << std::endl;
    std::cout << "           SOLVER STEP       "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     NU_NORM     "
              << " NU_U       " << std::endl;
    std::cout << splitter << std::endl;
  }

  template <int dim>
  void HyperelasticSolver<dim>::print_conv_footer()
  {
    static const unsigned int width = 100;
    std::string splitter(width, '_');
    std::cout << splitter << std::endl;
    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << errorUpdate.norm / errorUpdate0.norm
              << std::endl
              << "Force: \t\t" << errorResidual.norm / errorResidual0.norm
              << std::endl;
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
  void HyperelasticSolver<dim>::get_error_residual(Errors &residual)
  {
    Vector<double> res(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (!constraints.is_constrained(i))
          {
            res(i) = system_rhs(i);
          }
      }
    residual.norm = res.l2_norm();
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::get_error_update(const Vector<double> &newton_update,
                                            Errors &error_update)
  {
    Vector<double> error(dof_handler.n_dofs());
    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (!constraints.is_constrained(i))
          {
            error(i) = newton_update(i);
          }
      }
    error_update.norm = error.l2_norm();
  }

  template <int dim>
  Vector<double> HyperelasticSolver<dim>::get_total_solution(
    const Vector<double> &solution_delta) const
  {
    Vector<double> solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template <int dim>
  void HyperelasticSolver<dim>::assemble(bool assemble_lhs)
  {
    timer.enter_subsection("Assemble tangent matrix");
    std::cout << " ASM_K " << std::flush;

    if (assemble_lhs)
      {
        system_matrix = 0.0;
      }
    system_rhs = 0.0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_gradients |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_JxW_values);

    std::vector<std::vector<double>> phi(n_q_points,
                                         std::vector<double>(dofs_per_cell));
    std::vector<std::vector<Tensor<2, dim>>> grad_phi(
      n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi(
      n_q_points, std::vector<SymmetricTensor<2, dim>>(dofs_per_cell));

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

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
                grad_phi[q][k] = fe_values[displacement].gradient(k, q) * F_inv;
                sym_grad_phi[q][k] = symmetrize(grad_phi[q][k]);
              }
          }

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const SymmetricTensor<2, dim> tau = lqph[q]->get_tau();
            const SymmetricTensor<4, dim> Jc = lqph[q]->get_Jc();
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (assemble_lhs)
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;
                    for (unsigned int j = 0; j <= i; ++j)
                      {
                        const unsigned int component_j =
                          fe.system_to_component_index(j).first;
                        local_matrix(i, j) +=
                          sym_grad_phi[q][i] * Jc * sym_grad_phi[q][j] * JxW;
                        if (component_i == component_j)
                          {
                            local_matrix(i, j) +=
                              grad_phi[q][i][component_i] * tau *
                              grad_phi[q][j][component_j] * JxW;
                          }
                      }
                  }
                local_rhs(i) -= sym_grad_phi[q][i] * tau * JxW;
              }
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              {
                local_matrix(i, j) = local_matrix(j, i);
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
                const double p0 = -400.0 / (0.001 * 0.001);
                const double time_ramp = (time.current() / time.end());
                const double pressure = p0 * time_ramp;
                for (unsigned int q = 0; q < n_f_q_points; ++q)
                  {
                    const Tensor<1, dim> &N = fe_face_values.normal_vector(q);
                    const Tensor<1, dim> traction = pressure * N;
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const unsigned int component_j =
                          fe.system_to_component_index(j).first;
                        const double Ni = fe_face_values.shape_value(j, q);
                        const double JxW = fe_face_values.JxW(q);
                        local_rhs(j) +=
                          (Ni * traction[component_j]) * JxW; // +external force
                      }
                  }
                break;
              }
          }

        if (assemble_lhs)
          {
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
          }
        else
          {
            constraints.distribute_local_to_global(
              local_rhs, local_dof_indices, system_rhs);
          }
      }

    timer.leave_subsection();
  }

  template <int dim>
  std::pair<unsigned int, double>
  HyperelasticSolver<dim>::solve_linear_system(Vector<double> &newton_update)
  {
    timer.enter_subsection("Linear solver");
    std::cout << " SLV " << std::flush;

    unsigned int lin_it = 0;
    double lin_res = 0.0;

    const int solver_its = system_matrix.m() * parameters.solid_max_iterations;
    const double tol_sol = 1e-6 * system_rhs.l2_norm();
    SolverControl solver_control(solver_its, tol_sol);
    GrowingVectorMemory<Vector<double>> GVM;
    SolverCG<Vector<double>> solver_CG(solver_control, GVM);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver_CG.solve(system_matrix, newton_update, system_rhs, preconditioner);
    lin_it = solver_control.last_step();
    lin_res = solver_control.last_value();

    timer.leave_subsection();
    constraints.distribute(newton_update);
    return std::make_pair(lin_it, lin_res);
  }

  template <int dim>
  void
  HyperelasticSolver<dim>::output_results(const unsigned int output_index) const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    Vector<double> soln(solution.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      {
        soln(i) = solution(i);
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

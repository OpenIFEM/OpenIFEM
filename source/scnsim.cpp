#include "scnsim.h"

namespace Fluid
{
  /**
   * The initialization of the direct solver is expensive as it allocates
   * a lot of memory. The preconditioner is going to be applied several
   * times before it is re-initialized. Therefore initializing the direct
   * solver in the constructor saves time. However, it is pointless to do
   * this to iterative solvers.
   */
  template <int dim>
  SCnsIM<dim>::BlockIncompSchurPreconditioner::SchurComplementTpp::
    SchurComplementTpp(TimerOutput &timer,
                       const BlockSparseMatrix<double> &system,
                       const SparseILU<double> &Pvvinv)
    : timer(timer), system_matrix(&system), Pvv_inverse(&Pvvinv)
  {
  }

  template <int dim>
  void SCnsIM<dim>::BlockIncompSchurPreconditioner::SchurComplementTpp::vmult(
    Vector<double> &dst, const Vector<double> &src) const
  {
    TimerOutput::Scope timer_section(timer, "Tpp vmult");
    // this is the exact representation of Tpp = App - Apv * Avv * Avp.
    Vector<double> tmp1(this->Avv().m()), tmp2(this->Avv().m()),
      tmp3(src.size());
    this->Avp().vmult(tmp1, src);
    Pvv_inverse->vmult(tmp2, tmp1);
    this->Apv().vmult(tmp3, tmp2);
    this->App().vmult(dst, src);
    dst -= tmp3;
  }

  template <int dim>
  SCnsIM<dim>::BlockIncompSchurPreconditioner::BlockIncompSchurPreconditioner(
    TimerOutput &timer,
    const BlockSparseMatrix<double> &system,
    SparseMatrix<double> &schur,
    SparseMatrix<double> &B2pp)
    : timer(timer),
      system_matrix(&system),
      schur_matrix(&schur),
      B2pp_matrix(&B2pp),
      Tpp_itr(0)
  {
    // Initialize the Pvv inverse (the ILU(0) factorization of Avv)
    Pvv_inverse.initialize(this->Avv());
    // Initialize Tpp
    Tpp.reset(new SchurComplementTpp(timer, *system_matrix, Pvv_inverse));
    // Compute B2pp matrix App - Apv*rowsum(|Avv|)^(-1)*Avp
    // as the preconditioner to solve Tpp^-1
    Vector<double> RowSumAvv(this->Avv().m());
    Vector<double> IdentityVector(this->Avv().m());
    IdentityVector = 1;
    // iterate the Avv matrix to set everything to positive.
    SparseMatrix<double> tmp_Avv;
    tmp_Avv.reinit(this->Avv().get_sparsity_pattern());
    tmp_Avv.copy_from(this->Avv());
    for (auto itr = tmp_Avv.begin(); itr != tmp_Avv.end(); ++itr)
      {
        itr->value() = std::abs(itr->value());
      }
    // Compute the diag vector rowsum(|Avv|)^(-1)
    tmp_Avv.vmult(RowSumAvv, IdentityVector);
    for (auto itr = RowSumAvv.begin(); itr != RowSumAvv.end(); ++itr)
      { // Inverse the vector
        *itr = 1 / *itr;
      }
    // Compute Schur matrix Apv*rowsum(|Avv|)^(-1)*Avp
    this->Apv().mmult(*schur_matrix, this->Avp(), RowSumAvv, false);
    *B2pp_matrix = 0.0;
    // Add in numbers to B2pp
    *schur_matrix *= -1;
    for (auto itr = schur_matrix->begin(); itr != schur_matrix->end(); ++itr)
      {
        B2pp_matrix->set(itr->row(), itr->column(), itr->value());
      }
    for (auto itr = this->App().begin(); itr != App().end(); ++itr)
      {
        B2pp_matrix->add(itr->row(), itr->column(), itr->value());
      }
    B2pp_inverse.initialize(*B2pp_matrix);
  }

  template <int dim>
  void SCnsIM<dim>::BlockIncompSchurPreconditioner::vmult(
    BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    // Compute the intermediate vector:
    //      |I           0|*|src(0)| = |src(0)|
    //      |-ApvPvv^-1  I| |src(1)|   |ptmp  |
    /////////////////////////////////////////
    Vector<double> ptmp1(src.block(0).size()), ptmp(src.block(1).size());
    Pvv_inverse.vmult(ptmp1, src.block(0));
    this->Apv().vmult(ptmp, ptmp1);
    ptmp *= -1.0;
    ptmp += src.block(1);
    // Compute the final vector:
    //      |Pvv^-1     -Pvv^-1*Avp*Tpp^-1|*|src(0)|
    //      |0          Tpp^-1            | |ptmp  |
    //                        =   |Pvv^-1*src(0) - Pvv^-1*Avp*Tpp^-1*ptmp|
    //                            |Tpp^-1 * ptmp                         |
    //////////////////////////////////////////
    // Compute Tpp^-1 * ptmp first, which is equal to the problem Tpp*x = ptmp
    // Set up initial guess first
    {
      Vector<double> c(ptmp), Sc;
      Sc.reinit(c);
      Tpp->vmult(Sc, c);
      double alpha = (ptmp * c) / (Sc * c);
      c *= alpha;
      dst.block(1) = c;
    }

    // Compute the multiplication
    timer.enter_subsection("Solving Tpp");

    SolverControl solver_control(
      ptmp.size(), 1e-6 * ptmp.l2_norm(), true, true);
    SolverGMRES<Vector<double>> gmres(
      solver_control, SolverGMRES<Vector<double>>::AdditionalData(200));
    gmres.solve(*Tpp, dst.block(1), ptmp, B2pp_inverse);
    // Count iterations for this solver solving Tpp inverse
    Tpp_itr += solver_control.last_step();

    timer.leave_subsection("Solving Tpp");

    // Compute Pvv^-1*src(0) - Pvv^-1*Avp*dst(1)
    Vector<double> utmp1(src.block(0).size()), utmp2(src.block(0).size());
    this->Avp().vmult(utmp1, dst.block(1));
    Pvv_inverse.vmult(utmp2, utmp1);
    Pvv_inverse.vmult(dst.block(0), src.block(0));
    dst.block(0) -= utmp2;
  }

  template <int dim>
  SCnsIM<dim>::SCnsIM(Triangulation<dim> &tria,
                      const Parameters::AllParameters &parameters,
                      std::shared_ptr<Function<dim>> bc,
                      std::shared_ptr<Function<dim>> pml)
    : FluidSolver<dim>(tria, parameters, bc), sigma_pml_field(pml)
  {
    AssertThrow(parameters.fluid_velocity_degree ==
                  parameters.fluid_pressure_degree,
                ExcMessage("Velocity degree must the same as pressure!"));
  }

  template <int dim>
  void SCnsIM<dim>::initialize_system()
  {
    preconditioner.reset();
    system_matrix.clear();
    schur_matrix.clear();
    B2pp_matrix.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    present_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    evaluation_point.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);

    // Compute the sparsity pattern for schur in advance.
    // It should be the same as \f$BC^T\f$.
    DynamicSparsityPattern schur_dsp(dofs_per_block[1], dofs_per_block[1]);
    schur_dsp.compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                    sparsity_pattern.block(0, 1));
    schur_pattern.copy_from(schur_dsp);
    schur_matrix.reinit(schur_pattern);

    // Compute the pattern for Tpp
    for (auto itr = sparsity_pattern.block(1, 1).begin();
         itr != sparsity_pattern.block(1, 1).end();
         ++itr)
      {
        schur_dsp.add(itr->row(), itr->column());
      }
    Tpp_pattern.copy_from(schur_dsp);
    B2pp_matrix.reinit(Tpp_pattern);

    // Cell property
    setup_cell_property();
  }

  template <int dim>
  void SCnsIM<dim>::assemble(const bool use_nonzero_constraints)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system");

    const double viscosity = parameters.viscosity;
    const double rho = parameters.fluid_rho;
    Tensor<1, dim> gravity;
    for (unsigned int i = 0; i < dim; ++i)
      gravity[i] = parameters.gravity[i];

    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_face_q_points = face_quad_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // For the linearized system, we create temporary storage for current
    // velocity
    // and gradient, current pressure, and present velocity. In practice, they
    // are
    // all obtained through their shape functions at quadrature points.

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);
    std::vector<Tensor<1, dim>> current_pressure_gradients(n_q_points);
    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<double> present_pressure_values(n_q_points);
    std::vector<double> sigma_pml(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

    // The parameters that is used in isentropic continuity equation:
    // heat capacity ratio and atmospheric pressure.
    double cp_to_cv = 1.4;
    double atm = 1013250;

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        auto p = cell_property.get_data(cell);

        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs = 0;

        fe_values[velocities].get_function_values(evaluation_point,
                                                  current_velocity_values);

        fe_values[velocities].get_function_gradients(
          evaluation_point, current_velocity_gradients);

        fe_values[pressure].get_function_values(evaluation_point,
                                                current_pressure_values);

        fe_values[pressure].get_function_gradients(evaluation_point,
                                                   current_pressure_gradients);

        fe_values[velocities].get_function_values(present_solution,
                                                  present_velocity_values);

        fe_values[pressure].get_function_values(present_solution,
                                                present_pressure_values);

        sigma_pml_field->value_list(
          fe_values.get_quadrature_points(), sigma_pml, 0);

        // Assemble the system matrix
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const int ind = p[q]->indicator;
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
                grad_phi_p[k] = fe_values[pressure].gradient(k, q);
              }
            double current_velocity_divergence =
              trace(current_velocity_gradients[q]);

            // Define the UGN based SUPG parameters (Tezduyar):
            // tau_SUPG and tau_PSPG. They are
            // evaluated based on the results from the last Newton iteration.
            double tau_SUPG, tau_PSPG;
            // the length scale h is the length of the element in the direction
            // of convection
            double h = 0;
            for (unsigned int a = 0; a < dofs_per_cell / fe.dofs_per_vertex;
                 ++a)
              {
                h +=
                  abs(present_velocity_values[q] * fe_values.shape_grad(a, q));
              }
            if (h)
              h = 2 * present_velocity_values[q].norm() / h;
            else
              h = 0;
            double nu = viscosity / rho;
            double v_norm = present_velocity_values[q].norm();
            if (h)
              tau_SUPG =
                1 / sqrt((pow(2 / time.get_delta_t(), 2) +
                          pow(2 * v_norm / h, 2) + pow(4 * nu / pow(h, 2), 2)));
            else
              tau_SUPG = time.get_delta_t() / 2;
            tau_PSPG = tau_SUPG / rho;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // The bilinear term for linearized Navier Stokes
                    // equations
                    local_matrix(i, j) +=
                      ((viscosity *
                          scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                        rho * current_velocity_gradients[q] * phi_u[j] *
                          phi_u[i] +
                        rho * grad_phi_u[j] * current_velocity_values[q] *
                          phi_u[i] -
                        div_phi_u[i] * phi_p[j]) +
                       rho * phi_u[i] * phi_u[j] / time.get_delta_t()) *
                      fe_values.JxW(q);
                    // PML attenuation
                    local_matrix(i, j) +=
                      (rho * sigma_pml[q] * phi_u[j] * phi_u[i] +
                       sigma_pml[q] * phi_p[j] * phi_p[i] / (cp_to_cv * atm)) *
                      fe_values.JxW(q);
                    // Add SUPG and PSPG stabilization
                    local_matrix(i, j) +=
                      // SUPG Convection
                      (tau_SUPG * rho *
                         (current_velocity_values[q] * grad_phi_u[i]) *
                         (phi_u[j] * current_velocity_gradients[q]) +
                       tau_SUPG * rho *
                         (current_velocity_values[q] * grad_phi_u[i]) *
                         (current_velocity_values[q] * grad_phi_u[j]) +
                       tau_SUPG * rho * (phi_u[j] * grad_phi_u[i]) *
                         (current_velocity_values[q] *
                          current_velocity_gradients[q]) +
                       // SUPG Acceleration
                       tau_SUPG * rho * current_velocity_values[q] *
                         grad_phi_u[i] * phi_u[j] / time.get_delta_t() +
                       tau_SUPG * rho * phi_u[j] * grad_phi_u[i] *
                         (current_velocity_values[q] -
                          present_velocity_values[q]) /
                         time.get_delta_t() +
                       // SUPG Pressure
                       tau_SUPG * current_velocity_values[q] * grad_phi_u[i] *
                         grad_phi_p[j] +
                       tau_SUPG * phi_u[j] * grad_phi_u[i] *
                         current_pressure_gradients[q] +
                       // SUPG PML
                       tau_SUPG * rho * current_velocity_values[q] *
                         grad_phi_u[i] * sigma_pml[q] * phi_u[j] +
                       tau_SUPG * rho * phi_u[j] * grad_phi_u[i] *
                         sigma_pml[q] * current_velocity_values[q] +
                       // PSPG Convection
                       tau_PSPG * rho * grad_phi_p[i] *
                         (phi_u[j] * current_velocity_gradients[q]) +
                       tau_PSPG * rho * grad_phi_p[i] *
                         (current_velocity_values[q] * grad_phi_u[j]) +
                       // PSPG Acceleration
                       tau_PSPG * rho * grad_phi_p[i] * phi_u[j] /
                         time.get_delta_t() +
                       // PSPG Pressure
                       tau_PSPG * grad_phi_p[i] * grad_phi_p[j] +
                       // PSPG PML
                       tau_PSPG * rho * grad_phi_p[i] * sigma_pml[q] *
                         phi_u[j]) *
                      fe_values.JxW(q);
                    // For more clear demonstration, write continuity
                    // equation
                    // separately.
                    // The original strong form is:
                    // \f$p_{,t} + \frac{C_p}{C_v} * (p_0 + p) * (\nabla
                    // \times u) + u (\nabla p) = 0\f$
                    local_matrix(i, j) +=
                      (cp_to_cv * (atm + current_pressure_values[q]) *
                         div_phi_u[j] * phi_p[i] +
                       cp_to_cv * phi_p[j] * current_velocity_divergence *
                         phi_p[i] +
                       current_velocity_values[q] * grad_phi_p[j] * phi_p[i] +
                       phi_u[j] * current_pressure_gradients[q] * phi_p[i] +
                       phi_p[i] * phi_p[j] / time.get_delta_t()) /
                      (cp_to_cv * atm) * fe_values.JxW(q);
                  }

                // RHS is \f$-(A_{current} + C_{current}) -
                // M_{present-current}/\Delta{t}\f$.
                local_rhs(i) +=
                  ((-viscosity * scalar_product(current_velocity_gradients[q],
                                                grad_phi_u[i]) -
                    rho * current_velocity_gradients[q] *
                      current_velocity_values[q] * phi_u[i] +
                    current_pressure_values[q] * div_phi_u[i]) -
                   rho *
                     (current_velocity_values[q] - present_velocity_values[q]) *
                     phi_u[i] / time.get_delta_t() +
                   gravity * phi_u[i] * rho) *
                  fe_values.JxW(q);
                local_rhs(i) +=
                  -(rho * sigma_pml[q] * current_velocity_values[q] * phi_u[i] +
                    sigma_pml[q] * current_pressure_values[q] * phi_p[i] /
                      (cp_to_cv * atm)) *
                  fe_values.JxW(q);
                local_rhs(i) +=
                  -(cp_to_cv * (atm + current_pressure_values[q]) *
                      current_velocity_divergence * phi_p[i] +
                    current_velocity_values[q] * current_pressure_gradients[q] *
                      phi_p[i] +
                    (current_pressure_values[q] - present_pressure_values[q]) *
                      phi_p[i] / time.get_delta_t()) /
                  (cp_to_cv * atm) * fe_values.JxW(q);
                // Add SUPG and PSPS rhs terms.
                local_rhs(i) +=
                  -((tau_SUPG * current_velocity_values[q] * grad_phi_u[i]) *
                      (rho * ((current_velocity_values[q] -
                               present_velocity_values[q]) /
                                time.get_delta_t() +
                              current_velocity_values[q] *
                                current_velocity_gradients[q]) +
                       current_pressure_gradients[q] +
                       rho * sigma_pml[q] * current_velocity_values[q]) +
                    (tau_PSPG * grad_phi_p[i]) *
                      (rho * ((current_velocity_values[q] -
                               present_velocity_values[q]) /
                                time.get_delta_t() +
                              current_velocity_values[q] *
                                current_velocity_gradients[q]) +
                       current_pressure_gradients[q] +
                       rho * sigma_pml[q] * current_velocity_values[q])) *
                  fe_values.JxW(q);
                if (ind == 1)
                  {
                    local_rhs(i) +=
                      (scalar_product(grad_phi_u[i], p[q]->fsi_stress) +
                       (p[q]->fsi_acceleration * rho * phi_u[i])) *
                      fe_values.JxW(q);
                  }
              }

            // Impose pressure boundary here if specified, loop over faces on
            // the cell and apply pressure boundary conditions:
            // \f$\int_{\Gamma_n} -p\bold{n}d\Gamma\f$
            if (parameters.n_fluid_neumann_bcs != 0)
              {
                for (unsigned int face_n = 0;
                     face_n < GeometryInfo<dim>::faces_per_cell;
                     ++face_n)
                  {
                    if (cell->at_boundary(face_n) &&
                        parameters.fluid_neumann_bcs.find(
                          cell->face(face_n)->boundary_id()) !=
                          parameters.fluid_neumann_bcs.end())
                      {
                        fe_face_values.reinit(cell, face_n);
                        unsigned int p_bc_id =
                          cell->face(face_n)->boundary_id();
                        double boundary_values_p =
                          parameters.fluid_neumann_bcs[p_bc_id];
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                          {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                              {
                                local_rhs(i) +=
                                  -(fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);

            const AffineConstraints<double> &constraints_used =
              use_nonzero_constraints ? nonzero_constraints : zero_constraints;

            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs,
                                                        true);
          }
      }
  }

  template <int dim>
  std::pair<unsigned int, double>
  SCnsIM<dim>::solve(const bool use_nonzero_constraints)
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    preconditioner.reset(new BlockIncompSchurPreconditioner(
      timer, system_matrix, schur_matrix, B2pp_matrix));

    // NOTE: SolverFGMRES only applies the preconditioner from the right,
    // as opposed to SolverGMRES which allows both left and right
    // preconditoners.
    SolverControl solver_control(
      system_matrix.m(), 1e-8 * system_rhs.l2_norm(), true);
    GrowingVectorMemory<BlockVector<double>> vector_memory;
    SolverFGMRES<BlockVector<double>> gmres(solver_control, vector_memory);

    gmres.solve(system_matrix, newton_update, system_rhs, *preconditioner);

    const AffineConstraints<double> &constraints_used =
      use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute(newton_update);

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void SCnsIM<dim>::run_one_step(bool apply_nonzero_constraints,
                                 bool assemble_system)
  {
    (void)assemble_system;
    std::cout.precision(6);
    std::cout.width(12);

    if (time.get_timestep() == 0)
      {
        output_results(0);
      }

    time.increment();
    std::cout << std::string(96, '*') << std::endl
              << "Time step = " << time.get_timestep()
              << ", at t = " << std::scientific << time.current() << std::endl;

    // Resetting
    double current_residual = 1.0;
    double initial_residual = 1.0;
    double relative_residual = 1.0;
    unsigned int outer_iteration = 0;
    evaluation_point = present_solution;
    while (relative_residual > parameters.fluid_tolerance &&
           current_residual > 1e-14)
      {
        AssertThrow(outer_iteration < parameters.fluid_max_iterations,
                    ExcMessage("Too many Newton iterations!"));

        newton_update = 0.0;

        // Since evaluation_point changes at every iteration,
        // we have to reassemble both the lhs and rhs of the system
        // before solving it.
        // If the Dirichlet BCs are time-dependent, nonzero_constraints
        // should be applied at the first iteration of every time step;
        // if they are time-independent, nonzero_constraints should be
        // applied only at the first iteration of the first time step.
        assemble(apply_nonzero_constraints && outer_iteration == 0);
        auto state = solve(apply_nonzero_constraints && outer_iteration == 0);
        current_residual = system_rhs.l2_norm();

        // Update evaluation_point. Since newton_update has been set to
        // the correct bc values, there is no need to distribute the
        // evaluation_point again.
        evaluation_point.add(1.0, newton_update);

        // Update the relative residual
        if (outer_iteration == 0)
          {
            initial_residual = current_residual;
          }
        relative_residual = current_residual / initial_residual;

        std::cout << std::scientific << std::left << " ITR = " << std::setw(2)
                  << outer_iteration << " ABS_RES = " << current_residual
                  << " REL_RES = " << relative_residual
                  << " GMRES_ITR = " << std::setw(3) << state.first
                  << " GMRES_RES = " << state.second
                  << " INNER_GMRES_ITR = " << std::setw(3)
                  << preconditioner->get_Tpp_itr_count() << std::endl;

        outer_iteration++;
      }
    // Newton iteration converges, update time and solution
    present_solution = evaluation_point;
    // Output
    if (time.time_to_output())
      {
        output_results(time.get_timestep());
      }
    if (time.time_to_refine())
      {
        refine_mesh(1, 3);
      }
  }

  template <int dim>
  void SCnsIM<dim>::run()
  {
    triangulation.refine_global(parameters.global_refinements[0]);
    setup_dofs();
    make_constraints();
    initialize_system();

    // Time loop.
    // use_nonzero_constraints is set to true only at the first time step,
    // which means nonzero_constraints will be applied at the first iteration
    // in the first time step only, and never be used again.
    // This corresponds to time-independent Dirichlet BCs.
    run_one_step(true);
    while (time.end() - time.current() > 1e-12)
      {
        if (parameters.use_hard_coded_values)
          {
            // Only for time dependent BCs!
            // Advance the time by delta_t and make constraints
            boundary_values->advance_time(time.get_delta_t());
            make_constraints();
            run_one_step(true);
          }
        else
          run_one_step(false);
      }
  }

  template class SCnsIM<2>;
  template class SCnsIM<3>;
} // namespace Fluid

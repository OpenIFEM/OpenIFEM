#include "navierstokes.h"

namespace Fluid
{
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[0] - 0.3) < 1e-10)
      {
        double U = 1.5;
        double y = p[1];
        return 4 * U * y * (0.41 - y) / (0.41 * 0.41);
      }
    return 0;
  }

  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }

  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &m, const PreconditionerType &preconditioner)
    : matrix(&m), preconditioner(&preconditioner)
  {
  }

  template <class MatrixType, class PreconditionerType>
  void InverseMatrix<MatrixType, PreconditionerType>::vmult(
    Vector<double> &dst, const Vector<double> &src) const
  {
    SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    SolverCG<> cg(solver_control);
    dst = 0;
    cg.solve(*matrix, dst, src, *preconditioner);
  }

  ApproximateMassSchur::ApproximateMassSchur(
    const BlockSparseMatrix<double> &M)
    : mass_matrix(&M), tmp1(M.block(0, 0).m()), tmp2(M.block(0, 0).m())
  {
  }

  void ApproximateMassSchur::vmult(Vector<double> &dst,
                                         const Vector<double> &src) const
  {
    mass_matrix->block(0, 1).vmult(tmp1, src);
    mass_matrix->block(0, 0).precondition_Jacobi(tmp2, tmp1);
    mass_matrix->block(1, 0).vmult(dst, tmp2);
  }

  template <class PreconditionerSm, class PreconditionerMp>
  SchurComplementInverse<PreconditionerSm, PreconditionerMp>::SchurComplementInverse(
    double gamma, double viscosity, double dt,
    const InverseMatrix<ApproximateMassSchur, PreconditionerSm> &Sm_inv,
    const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mp_inv) :
    gamma(gamma), viscosity(viscosity), dt(dt), Sm_inverse(&Sm_inv), Mp_inverse(&Mp_inv)
  {
  }

  template <class PreconditionerSm, class PreconditionerMp>
  void SchurComplementInverse<PreconditionerSm, PreconditionerMp>::vmult(
    Vector<double> &dst, const Vector<double> &src) const
  {
    Vector<double> tmp(src.size());
    Mp_inverse->vmult(tmp, src);
    tmp *= -(viscosity + gamma);
    Sm_inverse->vmult(dst, src);
    dst *= -1/dt;
    dst += tmp;
  }

  /**
   * We can notice that the initialization of the inverse of the matrix at (0,0) corner
   * is completed in the constructor. Thus the initialization of this class is expensive,
   * but every application of the preconditioner then no longer requires
   * the computation of the matrix factors.
   */
  template <class PreconditionerSm, class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerSm, PreconditionerMp>::
  BlockSchurPreconditioner (
    const BlockSparseMatrix<double> &system,
    const SchurComplementInverse<PreconditionerSm, PreconditionerMp> &S_inv)
    :
    system_matrix (&system),
    S_inverse (&S_inv)
  {
    A_inverse.initialize(system_matrix->block(0,0));
  }

  template <class PreconditionerSm, class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerSm, PreconditionerMp>::vmult
    (BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));

    {
      S_inverse->vmult(dst.block(1), src.block(1));
    }

    {
      system_matrix->block(0,1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }

    A_inverse.vmult (dst.block(0), utmp);
  }

  template <int dim>
  NavierStokes<dim>::NavierStokes(Triangulation<dim> &tria, const Parameters::AllParameters &parameters)
    :
    viscosity(parameters.viscosity),
    gamma(parameters.grad_div),
    degree(parameters.fluid_degree),
    triangulation(tria),
    fe(FE_Q<dim>(degree+1), dim,
       FE_Q<dim>(degree),   1),
    dof_handler(triangulation),
    volume_quad_formula(degree+2),
    face_quad_formula(degree+2),
    tolerance(parameters.fluid_tolerance),
    max_iteration(parameters.fluid_max_iterations),
    time(parameters.end_time, parameters.time_step, parameters.output_interval),
    timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
  {}

  template <int dim>
  void NavierStokes<dim>::setup_dofs()
  {
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs (fe);

    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);
    std::vector<unsigned int> block_component(dim+1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise (dof_handler, block_component);

    dofs_per_block.resize (2);
    DoFTools::count_dofs_per_block (dof_handler, dofs_per_block, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    // In Newton's scheme, we first apply the boundary condition on the solution
    // obtained from the initial step. To make sure the boundary conditions remain
    // satisfied during Newton's iteration, zero boundary conditions are used for
    // the update \f$\delta u^k\f$. Therefore we set up two different constraint objects.
    // Dirichlet boundary conditions are applied to both boundaries 0 and 1.
    FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(dim+1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               BoundaryValues<dim>(),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }
    zero_constraints.close();

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << " (" << dof_u << '+' << dof_p << ')'
              << std::endl;

  }

  template <int dim>
  void NavierStokes<dim>::initialize_system()
  {
    BlockDynamicSparsityPattern dsp (dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern (dof_handler, dsp, nonzero_constraints);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);
    mass_matrix.reinit (sparsity_pattern);

    present_solution.reinit (dofs_per_block);
    newton_update.reinit (dofs_per_block);
    evaluation_point.reinit (dofs_per_block);
    system_rhs.reinit (dofs_per_block);
  }

  template <int dim>
  void NavierStokes<dim>::assemble(const bool initial_step,
                                   const bool assemble_matrix)
  {
    TimerOutput::Scope timer_section(timer, "Assemble system"); 

    if (assemble_matrix)
    {
      system_matrix = 0;
      mass_matrix = 0;
    }

    system_rhs = 0;

    FEValues<dim> fe_values (fe,
                             volume_quad_formula,
                             update_values |
                             update_quadrature_points |
                             update_JxW_values |
                             update_gradients );

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = volume_quad_formula.size();

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dim);

    FullMatrix<double>   local_matrix      (dofs_per_cell, dofs_per_cell);
    FullMatrix<double>   local_mass_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs         (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // For the linearized system, we create temporary storage for current velocity
    // and gradient, current pressure, and present velocity. In practice, they are
    // all obtained through their shape functions at quadrature points.

    std::vector<Tensor<1, dim> >  current_velocity_values    (n_q_points);
    std::vector<Tensor<2, dim> >  current_velocity_gradients (n_q_points);
    std::vector<double>           current_pressure_values    (n_q_points);
    std::vector<Tensor<1, dim> >  present_velocity_values    (n_q_points);

    std::vector<double>           div_phi_u                 (dofs_per_cell);
    std::vector<Tensor<1, dim> >  phi_u                     (dofs_per_cell);
    std::vector<Tensor<2, dim> >  grad_phi_u                (dofs_per_cell);
    std::vector<double>           phi_p                     (dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_mass_matrix =0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(evaluation_point,
                                                  current_velocity_values);

        fe_values[velocities].get_function_gradients(evaluation_point,
                                                     current_velocity_gradients);

        fe_values[pressure].get_function_values(evaluation_point,
                                                current_pressure_values);

        fe_values[velocities].get_function_values(present_solution,
                                                  present_velocity_values);

        // Assemble the system matrix and mass matrix simultaneouly.
        // We assemble \f$B^T\f$ and \f$B\f$ into the mass matrix for convience
        // because they will be used in the preconditioner.
        //
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
              {
                div_phi_u[k]  =  fe_values[velocities].divergence (k, q);
                grad_phi_u[k] =  fe_values[velocities].gradient(k, q);
                phi_u[k]      =  fe_values[velocities].value(k, q);
                phi_p[k]      =  fe_values[pressure]  .value(k, q);
              }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                if (assemble_matrix)
                  {
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      {
                        // Let the linearized diffusion, continuity and Grad-Div term be written as
                        // the bilinear operator: \f$A = a((\delta{u}, \delta{p}), (\delta{v}, \delta{q}))\f$,
                        // the linearized convection term be: \f$C = c(u;\delta{u}, \delta{v})\f$,
                        // and the linearized inertial term be:
                        // \f$M = m(\delta{u}, \delta{v})$, then LHS is: $(A + C) + M/{\Delta{t}}\f$
                        local_matrix(i, j) += ( (viscosity*scalar_product(grad_phi_u[j], grad_phi_u[i])
                                                 + current_velocity_gradients[q]*phi_u[j]*phi_u[i]
                                                 + grad_phi_u[j]*current_velocity_values[q]*phi_u[i]
                                                 - div_phi_u[i]*phi_p[j]
                                                 - phi_p[i]*div_phi_u[j]
                                                 + gamma*div_phi_u[j]*div_phi_u[i])
                                                +
                                                phi_u[i]*phi_u[j] / time.get_delta_t()
                                              )
                                              * fe_values.JxW(q);
                        local_mass_matrix(i, j) += (phi_u[i]*phi_u[j] + phi_p[i]*phi_p[j]
                                                    - div_phi_u[i]*phi_p[j] - phi_p[i]*div_phi_u[j])
                                                   * fe_values.JxW(q);
                      }
                  }

                // RHS is \f$-(A_{current} + C_{current}) - M_{present-current}/\Delta{t}\f$.
                double current_velocity_divergence =  trace(current_velocity_gradients[q]);
                local_rhs(i) += ( (- viscosity*scalar_product(current_velocity_gradients[q],grad_phi_u[i])
                                   - current_velocity_gradients[q]*current_velocity_values[q]*phi_u[i]
                                   + current_pressure_values[q]*div_phi_u[i]
                                   + current_velocity_divergence*phi_p[i]
                                   - gamma*current_velocity_divergence*div_phi_u[i])
                                  - (current_velocity_values[q] - present_velocity_values[q])*phi_u[i]
                                     / time.get_delta_t()
                                )
                                * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices (local_dof_indices);

        const ConstraintMatrix &constraints_used = initial_step ? nonzero_constraints : zero_constraints;

        if (assemble_matrix)
          {
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
            constraints_used.distribute_local_to_global(local_mass_matrix,
                                                        local_dof_indices,
                                                        mass_matrix);
          }
        else
          {
            constraints_used.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs);
          }
      }
  }

  template <int dim>
  void NavierStokes<dim>::assemble_system(const bool initial_step)
  {
    assemble(initial_step, true);
  }

  template <int dim>
  void NavierStokes<dim>::assemble_rhs(const bool initial_step)
  {
    assemble(initial_step, false);
  }

  template <int dim>
  std::pair<unsigned int, double> NavierStokes<dim>::solve (const bool initial_step)
  {
    // The only reason that the initialization of the preconditioners is timed 
    // separately from solving the system is that it calls a direct solver
    // to get \f$\tilde{A}^{-1}\f$ which is time-consuming. Other than that,
    // initializing preconditioners does not do real work.
    {
      TimerOutput::Scope timer_section(timer, "Initialize preconditioners"); 

      preconditioner.reset();
      S_inverse.reset();
      Mp_inverse.reset();
      preconditioner_Mp.reset();
      Sm_inverse.reset();
      preconditioner.reset();
      approximate_Sm.reset();

      approximate_Sm.reset(new ApproximateMassSchur(mass_matrix));
      preconditioner_Sm.reset(new PreconditionIdentity());
      Sm_inverse.reset(new InverseMatrix<ApproximateMassSchur, PreconditionIdentity>
        (*approximate_Sm, *preconditioner_Sm));
      preconditioner_Mp.reset(new SparseILU<double>());
      preconditioner_Mp->initialize(mass_matrix.block(1,1), SparseILU<double>::AdditionalData());
      Mp_inverse.reset(new InverseMatrix<SparseMatrix<double>, SparseILU<double>>
        (mass_matrix.block(1,1), *preconditioner_Mp));
      S_inverse.reset(new SchurComplementInverse<PreconditionIdentity,
        SparseILU<double>>(gamma, viscosity, time.get_delta_t(), *Sm_inverse, *Mp_inverse));
      preconditioner.reset(new BlockSchurPreconditioner
        <PreconditionIdentity, SparseILU<double>>(system_matrix, *S_inverse));
    }

    // NOTE: SolverFGMRES only applies the preconditioner from the right,
    // as opposed to SolverGMRES which allows both left and right preconditoners.
    SolverControl solver_control (system_matrix.m(), 1e-8*system_rhs.l2_norm(), true);
    GrowingVectorMemory<BlockVector<double>> vector_memory;
    SolverFGMRES<BlockVector<double>> gmres(solver_control, vector_memory);
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system"); 

      gmres.solve (system_matrix,
                  newton_update,
                  system_rhs,
                  *preconditioner);

      const ConstraintMatrix &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
      constraints_used.distribute(newton_update);
    }

    return {solver_control.last_step(), solver_control.last_value()};
  }

  template <int dim>
  void NavierStokes<dim>::refine_mesh()
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh"); 

    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(degree+1),
                                        typename FunctionMap<dim>::type(),
                                        present_solution,
                                        estimated_error_per_cell,
                                        fe.component_mask(velocity));

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3, 0.0);

    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, BlockVector<double> > solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement ();

    // First the DoFHandler is set up and constraints are generated. Then we
    // create a temporary vector whose size is according with the
    // solution on the new mesh.
    setup_dofs();

    BlockVector<double> tmp (dofs_per_block);

    // Transfer solution from coarse to fine mesh and apply boundary value
    // constraints to the new transfered solution. Note that present_solution
    // is still a vector corresponding to the old mesh.
    solution_transfer.interpolate(present_solution, tmp);
    nonzero_constraints.distribute(tmp);

    // Finally set up matrix and vectors and set the present_solution to the
    // interpolated data.
    initialize_system();
    present_solution = tmp;
  }

  template <int dim>
  void NavierStokes<dim>::run()
  {
    triangulation.refine_global(2);
    setup_dofs();
    initialize_system();

    bool first_step = true;

    // Time loop.
    std::cout.precision(6);
    std::cout.width(12);
    output_results(time.get_timestep());
    while (time.current() < time.end())
    {
      time.increment();
      std::cout << std::string(91, '*') << std::endl << "Time step = "
        << time.get_timestep() << ", at t = " << std::scientific
        << time.current() << std::endl;

      // Resetting current_res to a number greater than tolerance.
      double current_res = 1.0;
      unsigned int outer_iteration = 0;
      while (first_step || current_res > tolerance)
      {
        AssertThrow(outer_iteration < max_iteration,
          ExcMessage("Too many Newton iterations!"));

        // Since evaluation_point changes at every iteration,
        // we have to reassemble both the lhs and rhs of the system
        // before solving it.
        assemble_system(first_step);
        auto state = solve(first_step);
        current_res = system_rhs.l2_norm();

        // Update evaluation_point and do not forget to modify it
        // with constraints.
        evaluation_point.add(1.0, newton_update);
        nonzero_constraints.distribute(evaluation_point);
        first_step = false;

        std::cout << std::scientific << std::left
          << " Iteration = " << std::setw(2) << outer_iteration
          << " Residual = "<< current_res
          << " FGMRES iteration: " << std::setw(3) << state.first
          << " FGMRES residual: " << state.second << std::endl;

        outer_iteration++;
      }
      // Newton iteration converges, update time and solution
      present_solution = evaluation_point;
      // Output
      if (time.time_to_output())
      {
        output_results(time.get_timestep());
      }
    }
  }

  template <int dim>
  void NavierStokes<dim>::output_results (const unsigned int output_index)  const
  {
    TimerOutput::Scope timer_section(timer, "Output results"); 

    std::cout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names (dim, "velocity");
    solution_names.push_back ("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
    .push_back (DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::string basename = "flow_around_cylinder";
    std::string filename = basename + "-" + Utilities::int_to_string (output_index, 6) + ".vtu";

    std::ofstream output (filename);
    data_out.write_vtu (output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }

  template class NavierStokes<2>;
  template class NavierStokes<3>;
}

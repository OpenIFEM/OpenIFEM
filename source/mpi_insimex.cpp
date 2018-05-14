#include "mpi_insimex.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    double
    InsIMEX<dim>::BoundaryValues::value(const Point<dim> &p,
                                        const unsigned int component) const
    {
      Assert(component < this->n_components,
             ExcIndexRange(component, 0, this->n_components));
      double left_boundary = (dim == 2 ? 0.3 : 0.0);
      if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
        {
          // For a parabolic velocity profile, Uavg = 2/3 * Umax in 2D,
          // and 4/9 * Umax in 3D. If nu = 0.001, D = 0.1,
          // then Re = 100 * Uavg
          double Uavg = 0.2;
          double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
          double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
          if (dim == 3)
            {
              value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
            }
          return value;
        }
      return 0;
    }

    template <int dim>
    void
    InsIMEX<dim>::BoundaryValues::vector_value(const Point<dim> &p,
                                               Vector<double> &values) const
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = BoundaryValues::value(p, c);
    }

    template <int dim>
    PETScWrappers::MPI::BlockVector InsIMEX<dim>::get_current_solution() const
    {
      return present_solution;
    }

    template <int dim>
    InsIMEX<dim>::BlockSchurPreconditioner::BlockSchurPreconditioner(
      TimerOutput &timer,
      double gamma,
      double viscosity,
      double rho,
      double dt,
      const std::vector<IndexSet> &owned_partitioning,
      const PETScWrappers::MPI::BlockSparseMatrix &system,
      const PETScWrappers::MPI::BlockSparseMatrix &mass,
      PETScWrappers::MPI::BlockSparseMatrix &schur)
      : timer(timer),
        gamma(gamma),
        viscosity(viscosity),
        rho(rho),
        dt(dt),
        system_matrix(&system),
        mass_matrix(&mass),
        mass_schur(&schur)
    {
      TimerOutput::Scope timer_section(timer, "CG for Sm");
      // The sparsity pattern of mass_schur is already set,
      // we calculate its value in the following.
      PETScWrappers::MPI::BlockVector tmp1, tmp2;
      tmp1.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
      tmp2.reinit(owned_partitioning, mass_matrix->get_mpi_communicator());
      tmp1 = 1;
      tmp2 = 0;
      // Jacobi preconditioner of matrix A is by definition inverse diag(A),
      // this is exactly what we want to compute.
      // Note that the mass matrix and mass schur do not include the density.
      PETScWrappers::PreconditionJacobi jacobi(mass_matrix->block(0, 0));
      jacobi.vmult(tmp2.block(0), tmp1.block(0));
      // The sparsity pattern has already been set correctly, so explicitly
      // tell mmult not to rebuild the sparsity pattern.
      system_matrix->block(1, 0).mmult(
        mass_schur->block(1, 1), system_matrix->block(0, 1), tmp2.block(0));
    }

    /**
     * The vmult operation strictly follows the definition of
     * BlockSchurPreconditioner. Conceptually it computes \f$u = P^{-1}v\f$.
     */
    template <int dim>
    void InsIMEX<dim>::BlockSchurPreconditioner::vmult(
      PETScWrappers::MPI::BlockVector &dst,
      const PETScWrappers::MPI::BlockVector &src) const
    {
      // Temporary vectors
      PETScWrappers::MPI::Vector utmp(src.block(0));
      PETScWrappers::MPI::Vector tmp(src.block(1));
      tmp = 0;

      // This function is part of "solve linear system", but it
      // is further profiled to get a better idea of how time
      // is spent on different solvers.

      // This block computes \f$u_1 = \tilde{S}^{-1} v_1\f$,
      // where CG solvers are used for \f$M_p^{-1}\f$ and \f$S_m^{-1}\f$.
      {
        TimerOutput::Scope timer_section(timer, "CG for Mp");
        SolverControl mp_control(src.block(1).size(),
                                 1e-6 * src.block(1).l2_norm());
        PETScWrappers::SolverCG cg_mp(mp_control,
                                      mass_schur->get_mpi_communicator());
        // \f$-(\mu + \gamma\rho)M_p^{-1}v_1\f$
        PETScWrappers::PreconditionBlockJacobi Mp_preconditioner;
        Mp_preconditioner.initialize(mass_matrix->block(1, 1));
        cg_mp.solve(
          mass_matrix->block(1, 1), tmp, src.block(1), Mp_preconditioner);
        tmp *= -(viscosity + gamma * rho);
      }

      // FIXME: There is a mysterious bug here. After refine_mesh is called,
      // the initialization of Sm_preconditioner will complain about zero
      // entries on the diagonal which causes division by 0 since
      // PreconditionBlockJacobi uses ILU(0) underneath. This is similar to the
      // serial code where SparseILU is used. However, 1. if we do not use a
      // preconditioner here, the code runs fine, suggesting that mass_schur is
      // correct; 2. if we do not call refine_mesh, the code also runs fine. So
      // the question is, why would refine_mesh generate diagonal zeros?
      //
      // \f$-\frac{1}{dt}S_m^{-1}v_1\f$
      {
        TimerOutput::Scope timer_section(timer, "CG for Sm");
        SolverControl sm_control(src.block(1).size(),
                                 1e-6 * src.block(1).l2_norm());
        PETScWrappers::SolverCG cg_sm(sm_control,
                                      mass_schur->get_mpi_communicator());
        PETScWrappers::PreconditionBlockJacobi Sm_preconditioner;
        Sm_preconditioner.initialize(mass_schur->block(1, 1));
        cg_sm.solve(mass_schur->block(1, 1),
                    dst.block(1),
                    src.block(1),
                    Sm_preconditioner);
        dst.block(1) *= -rho / dt;
        // Adding up these two, we get \f$\tilde{S}^{-1}v_1\f$.
        dst.block(1) += tmp;
      }

      // Compute \f$v_0 - B^T\tilde{S}^{-1}v_1\f$ based on \f$u_1\f$.
      system_matrix->block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);

      // Finally, compute the product of \f$\tilde{A}^{-1}\f$ and utmp
      // using another CG solver.
      {
        TimerOutput::Scope timer_section(timer, "CG for A");
        SolverControl a_control(src.block(0).size(),
                                1e-6 * src.block(0).l2_norm());
        PETScWrappers::SolverCG cg_a(a_control,
                                     mass_schur->get_mpi_communicator());
        PETScWrappers::PreconditionNone A_preconditioner;
        A_preconditioner.initialize(system_matrix->block(0, 0));
        cg_a.solve(
          system_matrix->block(0, 0), dst.block(0), utmp, A_preconditioner);
      }
    }

    template <int dim>
    InsIMEX<dim>::InsIMEX(parallel::distributed::Triangulation<dim> &tria,
                          const Parameters::AllParameters &parameters)
      : viscosity(parameters.viscosity),
        rho(parameters.fluid_rho),
        gamma(parameters.grad_div),
        degree(parameters.fluid_pressure_degree),
        triangulation(tria),
        fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1),
        dof_handler(triangulation),
        volume_quad_formula(degree + 2),
        face_quad_formula(degree + 2),
        parameters(parameters),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout,
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
        time(parameters.end_time,
             parameters.time_step,
             parameters.output_interval,
             parameters.refinement_interval),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)

    {
    }

    template <int dim>
    void InsIMEX<dim>::setup_dofs()
    {
      // The first step is to associate DoFs with a given mesh.
      dof_handler.distribute_dofs(fe);

      // We renumber the components to have all velocity DoFs come before
      // the pressure DoFs to be able to split the solution vector in two blocks
      // which are separately accessed in the block preconditioner.
      DoFRenumbering::Cuthill_McKee(dof_handler);
      std::vector<unsigned int> block_component(dim + 1, 0);
      block_component[dim] = 1;
      DoFRenumbering::component_wise(dof_handler, block_component);

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

      pcout << "   Number of active fluid cells: "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << dof_u << '+' << dof_p << ')' << std::endl;
    }

    template <int dim>
    void InsIMEX<dim>::make_constraints()
    {
      // Because the equation is written in incremental form, two constraints
      // are needed: nonzero constraint and zero constraint.

      // For inhomogeneous BC, only constant input values can be read from
      // the input file. If time or space dependent Dirichlet BCs are
      // desired, they must be implemented in BoundaryValues.
      nonzero_constraints.clear();
      zero_constraints.clear();
      nonzero_constraints.reinit(locally_relevant_dofs);
      zero_constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
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
          if (parameters.use_hard_coded_values == 1)
            {
              VectorTools::interpolate_boundary_values(dof_handler,
                                                       id,
                                                       BoundaryValues(),
                                                       nonzero_constraints,
                                                       ComponentMask(mask));
            }
          else
            {
              VectorTools::interpolate_boundary_values(
                dof_handler,
                id,
                Functions::ConstantFunction<dim>(augmented_value),
                nonzero_constraints,
                ComponentMask(mask));
            }
          VectorTools::interpolate_boundary_values(
            dof_handler,
            id,
            Functions::ZeroFunction<dim>(dim + 1),
            zero_constraints,
            ComponentMask(mask));
        }
      nonzero_constraints.close();
      zero_constraints.close();
    }

    template <int dim>
    void InsIMEX<dim>::setup_cell_property()
    {
      const unsigned int n_q_points = volume_quad_formula.size();
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell_property.initialize(cell, n_q_points);
              const std::vector<std::shared_ptr<CellProperty>> p =
                cell_property.get_data(cell);
              Assert(p.size() == n_q_points,
                     ExcMessage("Wrong number of cell property!"));
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  p[q]->indicator = 0;
                  p[q]->fsi_acceleration = 0;
                  p[q]->fsi_stress = 0;
                }
            }
        }
    }

    template <int dim>
    void InsIMEX<dim>::initialize_system()
    {
      preconditioner.reset();
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
      // newton_update is non-ghosted because the linear solver needs
      // a completely distributed vector.
      solution_increment.reinit(owned_partitioning, mpi_communicator);
      // system_rhs is non-ghosted because it is only used in the linear
      // solver and residual evaluation.
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      // Cell property
      setup_cell_property();
    }

    template <int dim>
    void InsIMEX<dim>::assemble(bool use_nonzero_constraints,
                                bool assemble_system)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      if (assemble_system)
        {
          system_matrix = 0;
          mass_matrix = 0;
        }
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
      FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
      std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
      std::vector<double> current_velocity_divergences(n_q_points);
      std::vector<double> current_pressure_values(n_q_points);

      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);

              fe_values.reinit(cell);

              if (assemble_system)
                {
                  local_matrix = 0;
                  local_mass_matrix = 0;
                }
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                present_solution, current_velocity_values);

              fe_values[velocities].get_function_gradients(
                present_solution, current_velocity_gradients);

              fe_values[velocities].get_function_divergences(
                present_solution, current_velocity_divergences);

              fe_values[pressure].get_function_values(present_solution,
                                                      current_pressure_values);

              // Assemble the system matrix and mass matrix simultaneouly.
              // The mass matrix only uses the (0, 0) and (1, 1) blocks.
              //
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const int ind = p[q]->indicator;
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      if (assemble_system)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              local_matrix(i, j) +=
                                (viscosity * scalar_product(grad_phi_u[j],
                                                            grad_phi_u[i]) -
                                 div_phi_u[i] * phi_p[j] -
                                 phi_p[i] * div_phi_u[j] +
                                 gamma * div_phi_u[j] * div_phi_u[i] * rho +
                                 phi_u[i] * phi_u[j] / time.get_delta_t() *
                                   rho) *
                                fe_values.JxW(q);
                              local_mass_matrix(i, j) +=
                                (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j]) *
                                fe_values.JxW(q);
                            }
                        }
                      local_rhs(i) -=
                        (viscosity *
                           scalar_product(current_velocity_gradients[q],
                                          grad_phi_u[i]) -
                         current_velocity_divergences[q] * phi_p[i] -
                         current_pressure_values[q] * div_phi_u[i] +
                         gamma * current_velocity_divergences[q] *
                           div_phi_u[i] * rho +
                         current_velocity_gradients[q] *
                           current_velocity_values[q] * phi_u[i] * rho) *
                        fe_values.JxW(q);
                      if (ind == 1)
                        {
                          local_rhs(i) +=
                            (scalar_product(grad_phi_u[i], p[q]->fsi_stress) +
                             (p[q]->fsi_acceleration * rho * phi_u[i])) *
                            fe_values.JxW(q);
                        }
                    }
                }

              // Impose pressure boundary here if specified, loop over faces on
              // the
              // cell
              // and apply pressure boundary conditions:
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
                                  local_rhs(i) += -(
                                    fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                                }
                            }
                        }
                    }
                }

              cell->get_dof_indices(local_dof_indices);

              const ConstraintMatrix &constraints_used = use_nonzero_constraints
                                                           ? nonzero_constraints
                                                           : zero_constraints;
              if (assemble_system)
                {
                  constraints_used.distribute_local_to_global(local_matrix,
                                                              local_rhs,
                                                              local_dof_indices,
                                                              system_matrix,
                                                              system_rhs);
                  constraints_used.distribute_local_to_global(
                    local_mass_matrix, local_dof_indices, mass_matrix);
                }
              else
                {
                  constraints_used.distribute_local_to_global(
                    local_rhs, local_dof_indices, system_rhs);
                }
            }
        }

      if (assemble_system)
        {
          system_matrix.compress(VectorOperation::add);
          mass_matrix.compress(VectorOperation::add);
        }
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned int, double>
    InsIMEX<dim>::solve(bool use_nonzero_constraints, bool assemble_system)
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      if (assemble_system)
        {
          preconditioner.reset(new BlockSchurPreconditioner(timer,
                                                            gamma,
                                                            viscosity,
                                                            rho,
                                                            time.get_delta_t(),
                                                            owned_partitioning,
                                                            system_matrix,
                                                            mass_matrix,
                                                            mass_schur));
        }

      SolverControl solver_control(
        system_matrix.m(), 1e-8 * system_rhs.l2_norm(), true);
      // Because PETScWrappers::SolverGMRES requires preconditioner derived
      // from PETScWrappers::PreconditionBase, we use dealii SolverFGMRES.
      GrowingVectorMemory<PETScWrappers::MPI::BlockVector> vector_memory;
      SolverFGMRES<PETScWrappers::MPI::BlockVector> gmres(solver_control,
                                                          vector_memory);

      // The solution vector must be non-ghosted
      gmres.solve(
        system_matrix, solution_increment, system_rhs, *preconditioner);

      const ConstraintMatrix &constraints_used =
        use_nonzero_constraints ? nonzero_constraints : zero_constraints;
      constraints_used.distribute(solution_increment);

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void InsIMEX<dim>::refine_mesh(const unsigned int min_grid_level,
                                   const unsigned int max_grid_level)
    {
      TimerOutput::Scope timer_section(timer, "Refine mesh");

      Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
      FEValuesExtractors::Vector velocity(0);
      KellyErrorEstimator<dim>::estimate(dof_handler,
                                         face_quad_formula,
                                         typename FunctionMap<dim>::type(),
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
      PETScWrappers::MPI::BlockVector tmp(solution_increment);
      tmp = 0;
      trans.interpolate(tmp);
      nonzero_constraints.distribute(tmp); // Is this line necessary?
      present_solution = tmp;
    }

    template <int dim>
    void InsIMEX<dim>::run_one_step()
    {
      std::cout.precision(6);
      std::cout.width(12);

      if (time.get_timestep() == 0)
        {
          output_results(0);
        }

      time.increment();
      pcout << std::string(96, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;

      // Resetting
      solution_increment = 0;
      // Only use nonzero constraints at the very first time step
      bool apply_nonzero_constraints = (time.get_timestep() == 1);
      // We have to assemble the LHS twice: once using nonzero_constraints,
      // once using zero_constraints.
      bool assemble_system = (time.get_timestep() < 3);
      assemble(apply_nonzero_constraints, assemble_system);
      auto state = solve(apply_nonzero_constraints, assemble_system);

      // Note we have to use a non-ghosted vector in order to do addition.
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);
      tmp = present_solution;
      tmp += solution_increment;
      present_solution = tmp;

      pcout << std::scientific << std::left << " GMRES_ITR = " << std::setw(3)
            << state.first << " GMRES_RES = " << state.second << std::endl;

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
    void InsIMEX<dim>::run()
    {
      pcout << "Running with PETSc on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      triangulation.refine_global(parameters.global_refinement);
      setup_dofs();
      make_constraints();
      initialize_system();

      // Time loop.
      while (time.end() - time.current() > 1e-12)
        {
          run_one_step();
        }
    }

    template <int dim>
    void InsIMEX<dim>::output_results(const unsigned int output_index) const
    {
      TimerOutput::Scope timer_section(timer, "Output results");

      pcout << "Writing results..." << std::endl;
      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.push_back("pressure");

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

      // Partition
      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");

      // Indicator
      Vector<float> ind(triangulation.n_active_cells());
      int cnt = 0;
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);
              bool artificial = false;
              for (auto ptr : p)
                {
                  if (ptr->indicator == 1)
                    {
                      artificial = true;
                      break;
                    }
                }
              ind[cnt++] = artificial;
            }
        }
      data_out.add_data_vector(ind, "Indicator");
      data_out.build_patches(parameters.fluid_velocity_degree);

      std::string basename =
        "navierstokes" + Utilities::int_to_string(output_index, 6) + "-";

      std::string filename =
        basename +
        Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
        ".vtu";

      std::ofstream output(filename);
      data_out.write_vtu(output);

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
          std::ofstream pvd_output("navierstokes.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template class InsIMEX<2>;
    template class InsIMEX<3>;
  } // namespace MPI
} // namespace Fluid

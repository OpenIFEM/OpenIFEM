#include "mpi_stokes.h"

namespace Fluid
{
  namespace MPI
  {

    // Define numerical constants to replace magic numbers
    namespace NumericalConstants
    {
      constexpr double DOMAIN_HEIGHT = 0.41; // Height of the domain
      constexpr double DOMAIN_LENGTH = 2.2;  // Length of the domain
      constexpr double INLET_U_MAX = 100;    // Maximum inlet velocity
      constexpr double TOLERANCE = 1e-10;    // Tolerance for boundary checks
    }                                        // namespace NumericalConstants
                                             // namespace NumericalConstants

    namespace LinearSolvers
    {
      template <class Matrix, class Preconditioner>
      class InverseMatrix : public Subscriptor
      {
      public:
        InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

        template <typename VectorType>
        void vmult(VectorType &dst, const VectorType &src) const;

      private:
        const SmartPointer<const Matrix> matrix;
        const Preconditioner &preconditioner;
      };

      template <class Matrix, class Preconditioner>
      InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
        const Matrix &m, const Preconditioner &preconditioner)
        : matrix(&m), preconditioner(preconditioner)
      {
      }

      template <class Matrix, class Preconditioner>
      template <typename VectorType>
      void
      InverseMatrix<Matrix, Preconditioner>::vmult(VectorType &dst,
                                                   const VectorType &src) const
      {
        SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
        SolverCG<VectorType> cg(solver_control);
        dst = 0;

        try
          {
            cg.solve(*matrix, dst, src, preconditioner);
          }
        catch (std::exception &e)
          {
            Assert(false, ExcMessage(e.what()));
          }
      }

      template <class PreconditionerA, class PreconditionerS>
      class BlockDiagonalPreconditioner : public Subscriptor
      {
      public:
        BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                    const PreconditionerS &preconditioner_S);

        // void vmult(PETScWrappers::MPI::BlockVector       &dst,
        void
        vmult(dealii::LinearAlgebraPETSc::MPI::BlockVector &dst,
              //  const PETScWrappers::MPI::BlockVector &src) const;
              const dealii::LinearAlgebraPETSc::MPI::BlockVector &src) const;

      private:
        const PreconditionerA &preconditioner_A;
        const PreconditionerS &preconditioner_S;
      };

      template <class PreconditionerA, class PreconditionerS>
      BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
        BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                    const PreconditionerS &preconditioner_S)
        : preconditioner_A(preconditioner_A), preconditioner_S(preconditioner_S)
      {
      }

      template <class PreconditionerA, class PreconditionerS>
      void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
        //  PETScWrappers::MPI::BlockVector       &dst,
        //  const PETScWrappers::MPI::BlockVector &src) const
        dealii::LinearAlgebraPETSc::MPI::BlockVector &dst,
        const dealii::LinearAlgebraPETSc::MPI::BlockVector &src) const
      {
        preconditioner_A.vmult(dst.block(0), src.block(0));
        preconditioner_S.vmult(dst.block(1), src.block(1));
      }

    } // namespace LinearSolvers
      /*
        template <class MatrixType, class PreconditionerType>
        class InverseMatrix : public Subscriptor
        {
           public:
           InverseMatrix(const MatrixType &m,
                        const PreconditionerType &preconditioner);
  
          void vmult(PETScWrappers::MPI::Vector &dst, const
      PETScWrappers::MPI::Vector &src) const;
  
          private:
          const SmartPointer<const MatrixType> matrix;
          const SmartPointer<const PreconditionerType> preconditioner;
  
        };
  
       template <class MatrixType, class PreconditionerType>
        InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
          const MatrixType &m, const PreconditionerType &preconditioner)
          : matrix(&m), preconditioner(&preconditioner)
        {
        }
  
        template <class MatrixType, class PreconditionerType>
        void InverseMatrix<MatrixType, PreconditionerType>::vmult(
         PETScWrappers::MPI::Vector &dst, const PETScWrappers::MPI::Vector &src)
      const
         {
             SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
             PETScWrappers::SolverCG cg(solver_control);
  
             dst = 0.0;
  
             cg.solve(*matrix, dst, src, *preconditioner);
         }
  
        template <class PreconditionerType>
        class SchurComplement : public Subscriptor
        {
          public:
          SchurComplement(
            const PETScWrappers::MPI::BlockSparseMatrix &system_matrix,
            const InverseMatrix<PETScWrappers::MPI::SparseMatrix,
      PreconditionerType> &A_inverse);
  
          void vmult(PETScWrappers::MPI::Vector &dst, const
      PETScWrappers::MPI::Vector &src) const;
  
          private:
          const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
      system_matrix;   const SmartPointer<   const
      InverseMatrix<PETScWrappers::MPI::SparseMatrix, PreconditionerType>>
            A_inverse;
  
          mutable PETScWrappers::MPI::Vector tmp1, tmp2;
        };
  
        template <class PreconditionerType>
        SchurComplement<PreconditionerType>::SchurComplement(
          const PETScWrappers::MPI::BlockSparseMatrix &system_matrix,
          const InverseMatrix<PETScWrappers::MPI::SparseMatrix,
      PreconditionerType> &A_inverse)   : system_matrix(&system_matrix),
            A_inverse(&A_inverse)
          //const std::vector<IndexSet> &owned_partitioning  // not sure need to
      not for now
          //tmp1(system_matrix.block(0, 0).m(),
      system_matrix.get_mpi_communicator()),
          //tmp2(system_matrix.block(0, 0).m(),
      system_matrix.get_mpi_communicator())
        {
          //const std::vector<IndexSet> &owned_partitioning;
          const IndexSet &owned_range_indices = system_matrix.block(0,
      0).locally_owned_range_indices();   tmp1.reinit(owned_range_indices,
      system_matrix.get_mpi_communicator());   tmp2.reinit(owned_range_indices,
      system_matrix.get_mpi_communicator());
           //tmp1.reinit(owned_partitioning,
      system_matrix.get_mpi_communicator());
           //tmp2.reinit(owned_partitioning,
      system_matrix.get_mpi_communicator());
        }
  
      template <class PreconditionerType>
      void
      SchurComplement<PreconditionerType>::vmult(PETScWrappers::MPI::Vector &dst,
                                                 const PETScWrappers::MPI::Vector
      &src) const
      {
        system_matrix->block(0, 1).vmult(tmp1, src);
        A_inverse->vmult(tmp2, tmp1);
        system_matrix->block(1, 0).vmult(dst, tmp2);
      }
      */

    template <int dim>
    Stokes<dim>::Stokes(parallel::distributed::Triangulation<dim> &tria,
                        const Parameters::AllParameters &parameters)
      : FluidSolver<dim>(tria, parameters)
    {
      Assert(
        parameters.fluid_velocity_degree - parameters.fluid_pressure_degree ==
          1,
        ExcMessage(
          "Velocity finite element should be one order higher than pressure!"));
    }

    template <int dim>
    class InletVelocity : public Function<dim>
    {
    public:
      InletVelocity()
        : Function<dim>(dim) {
      } // in serial code I used dim+1, only one is correct; check in the future

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override
      {

        const double y = p[1];
        const double H =
          NumericalConstants::DOMAIN_HEIGHT; // Height of the domain

        values(0) = NumericalConstants::INLET_U_MAX *
                    (1.0 - std::pow((2.0 * y / H - 1), 2));
        // values(0) = NumericalConstants::INLET_U_MAX ; //uniform inlet
        // velocity

        for (unsigned int i = 1; i < dim + 1; ++i)
          values(i) = 0.0; // No velocity in other components
      }

    private:
    };

    template <int dim>
    void Stokes<dim>::set_up_boundary_values()
    {
      constraints.clear();

      constraints.reinit(locally_relevant_dofs);
      // constraints.reinit(locally_owned_dofs, locally_relevant_dofs); from
      // step 55

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(
        dim); // for fixing pressure only

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      Functions::ZeroFunction<dim> zero_velocity(
        dim); // in serial code I used dim+1, only one is correct; check in the
              // future

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               zero_velocity,
                                               constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               zero_velocity,
                                               constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               4,
                                               zero_velocity,
                                               constraints,
                                               fe.component_mask(velocities));

      // Inlet (Left Boundary) with Parabolic Velocity Profile
      InletVelocity<dim> inlet_velocity;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               inlet_velocity,
                                               constraints,
                                               fe.component_mask(velocities));

      // fix pressure at a given point
      // Build a map from DoFs to their support points

      // std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      std::map<types::global_dof_index, Point<dim>> support_points;

      // Use a suitable mapping; if you're using higher-order elements, adjust
      // accordingly
      MappingQGeneric<dim> mapping(parameters.fluid_pressure_degree);

      DoFTools::map_dofs_to_support_points(
        mapping, dof_handler, support_points);

      // Now extract pressure DoFs (serial version)
      IndexSet pressure_dofs =
        DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure));

      IndexSet locally_owned_pressure_dofs(dof_handler.n_dofs());
      locally_owned_pressure_dofs.clear();

      // Manually compute the intersection of pressure_dofs and locally owned
      // DoFs
      for (IndexSet::ElementIterator index = pressure_dofs.begin();
           index != pressure_dofs.end();
           ++index)
        {
          types::global_dof_index i = *index;
          if (dof_handler.locally_owned_dofs().is_element(i))
            {
              locally_owned_pressure_dofs.add_index(i);
            }
        }
      locally_owned_pressure_dofs.compress();

      Point<dim> target_point(NumericalConstants::DOMAIN_LENGTH, 0.0);

      types::global_dof_index local_fixed_pressure_dof =
        numbers::invalid_dof_index;

      double local_min_distance = std::numeric_limits<double>::max();

      Point<dim> local_fixed_pressure_dof_location;

      // Loop over locally owned pressure DoFs
      for (IndexSet::ElementIterator index =
             locally_owned_pressure_dofs.begin();
           index != locally_owned_pressure_dofs.end();
           ++index)
        {
          types::global_dof_index dof = *index;

          auto it = support_points.find(dof);
          if (it != support_points.end())
            {
              double distance = it->second.distance(target_point);
              if (distance < local_min_distance)
                {
                  local_min_distance = distance;
                  local_fixed_pressure_dof = dof;
                  local_fixed_pressure_dof_location = it->second;
                }
            }
        }

      // Obtain the MPI rank as an int using standard MPI
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // Use MPI to find the global minimum distance and the rank of the process
      // that has it
      struct
      {
        double distance;
        int rank;
      } local_data, global_data;

      local_data.distance = local_min_distance;
      local_data.rank = rank;

      // Perform an MPI_Allreduce with MPI_MINLOC operation
      MPI_Allreduce(&local_data,
                    &global_data,
                    1,
                    MPI_DOUBLE_INT,
                    MPI_MINLOC,
                    MPI_COMM_WORLD);

      // Now, global_data.distance is the global minimum distance
      // and global_data.rank is the rank of the process that has it

      // Initialize the global fixed pressure DoF index
      types::global_dof_index global_fixed_pressure_dof =
        numbers::invalid_dof_index;

      // The process with rank == global_data.rank sets the
      // global_fixed_pressure_dof
      if (rank == global_data.rank)
        {
          global_fixed_pressure_dof = local_fixed_pressure_dof;
        }

      // Broadcast the global_fixed_pressure_dof to all processes
      MPI_Bcast(&global_fixed_pressure_dof,
                1,
                MPI_UNSIGNED_LONG_LONG,
                global_data.rank,
                MPI_COMM_WORLD);

      // Add the constraint if we found a local DoF that is the closest
      if (global_fixed_pressure_dof != numbers::invalid_dof_index)
        {
          if (dof_handler.locally_owned_dofs().is_element(
                global_fixed_pressure_dof))
            {
              constraints.add_line(global_fixed_pressure_dof);
              constraints.set_inhomogeneity(global_fixed_pressure_dof,
                                            0.0); // Set pressure to zero

              std::cout << "Process "
                        << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                        << " constrains pressure DoF at: "
                        << support_points[global_fixed_pressure_dof]
                        << std::endl;
            }
        }

      constraints.close();
    }

    template <int dim>
    void Stokes<dim>::initialize_system()
    {
      FluidSolver<dim>::initialize_system();
      // A_preconditioner.reset();
      preconditioner_matrix.clear();

      // BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      BlockDynamicSparsityPattern dsp(relevant_partitioning);

      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

      // sparsity_pattern.copy_from(dsp);

      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);

      BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                     dofs_per_block);

      DoFTools::make_sparsity_pattern(
        dof_handler, preconditioner_dsp, constraints);

      // preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);

      SparsityTools::distribute_sparsity_pattern(
        preconditioner_dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   dof_handler.locally_owned_dofs()),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
      preconditioner_matrix.reinit(
        owned_partitioning, preconditioner_dsp, mpi_communicator);
      solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      system_rhs.reinit(owned_partitioning, mpi_communicator);

      // Cell property
      setup_cell_property();
    }

    template <int dim>
    void Stokes<dim>::assemble()
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");
      const double viscosity = parameters.viscosity;

      const double rho = parameters.fluid_rho;

      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        gravity[i] = parameters.gravity[i];

      system_matrix = 0;
      preconditioner_matrix = 0;
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
      FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                     dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);

      std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);

              fe_values.reinit(cell);

              local_matrix = 0;
              local_preconditioner_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                present_solution, current_velocity_values);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                      symgrad_phi_u[k] =
                        fe_values[velocities].symmetric_gradient(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      // for (unsigned int j = 0; j <= i; ++j)
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          local_matrix(i, j) +=

                            (rho / time.get_delta_t()) * (phi_u[i] * phi_u[j]) *
                            fe_values.JxW(q); // time derivative term

                          local_matrix(i, j) +=
                            //  (2 * viscosity *
                            //     (symgrad_phi_u[i] * symgrad_phi_u[j]) //
                            //     viscous term
                            (viscosity *
                               scalar_product(grad_phi_u[i],
                                              grad_phi_u[j]) // viscous term
                             - div_phi_u[i] * phi_p[j]       // pressure term
                             - phi_p[i] * div_phi_u[j])      // divergence term
                                                        // (incompressibility)
                            * fe_values.JxW(q);

                          local_preconditioner_matrix(i, j) +=
                            ((rho / time.get_delta_t()) * phi_u[i] * phi_u[j] *
                             fe_values.JxW(q)) +
                            // ((2 / viscosity) * phi_p[i] * phi_p[j] *
                            //   fe_values.JxW(q));
                            1.0 / viscosity * phi_p[i] * phi_p[j] *
                              fe_values.JxW(q);
                        }

                      local_rhs(i) += phi_u[i] * gravity * fe_values.JxW(q);

                      local_rhs(i) += (rho / time.get_delta_t()) *
                                      (phi_u[i] * current_velocity_values[q]) *
                                      fe_values.JxW(q); // time derivative term
                    }
                }

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

              // for (unsigned int i = 0; i < dofs_per_cell; ++i)
              // for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              // {
              // local_matrix(i, j) = local_matrix(j, i);
              // local_preconditioner_matrix(i, j) =
              //  local_preconditioner_matrix(j, i);
              // }

              cell->get_dof_indices(local_dof_indices);

              constraints.distribute_local_to_global(local_matrix,
                                                     local_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);

              constraints.distribute_local_to_global(
                local_preconditioner_matrix,
                local_dof_indices,
                preconditioner_matrix);
            }
        }

      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
      preconditioner_matrix.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned int, double> Stokes<dim>::solve()
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG prec_A;

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data_A;

      data_A.symmetric_operator = true;

      prec_A.initialize(system_matrix.block(0, 0), data_A);

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG prec_S;

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data_S;

      data_S.symmetric_operator = true;

      prec_S.initialize(preconditioner_matrix.block(1, 1), data_S);

      using mp_inverse_t = LinearSolvers::InverseMatrix<
        dealii::LinearAlgebraPETSc::MPI::SparseMatrix,
        dealii::LinearAlgebraPETSc::MPI::PreconditionAMG>;

      const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

      const LinearSolvers::BlockDiagonalPreconditioner<
        dealii::LinearAlgebraPETSc::MPI::PreconditionAMG,
        mp_inverse_t>
        preconditioner(prec_A, mp_inverse);

      SolverControl solver_control(system_matrix.m(),
                                   1e-12 * system_rhs.l2_norm());

      // SolverMinRes<PETScWrappers::MPI::BlockVector> solver(solver_control);
      SolverMinRes<dealii::LinearAlgebraPETSc::MPI::BlockVector> solver(
        solver_control);

      // GrowingVectorMemory<BlockVector<double> > vector_memory;

      // SolverGMRES<BlockVector<double> > solver(solver_control,
      // vector_memory);

      // PETScWrappers::MPI::BlockVector
      // distributed_solution(owned_partitioning,
      // mpi_communicator);
      dealii::LinearAlgebraPETSc::MPI::BlockVector distributed_solution(
        owned_partitioning, mpi_communicator);

      constraints.set_zero(distributed_solution);

      solver.solve(
        system_matrix, distributed_solution, system_rhs, preconditioner);

      constraints.distribute(distributed_solution);

      solution = distributed_solution;

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>

    void Stokes<dim>::output_results(const unsigned int output_index) const
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

      data_out.add_data_vector(solution,
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
      data_out.build_patches(parameters.fluid_pressure_degree);

      data_out.write_vtu_with_pvtu_record(
        "./", "fluid", output_index, mpi_communicator, 6, 0);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          pvd_writer.write_current_timestep("fluid_", 6);
        }
    }

    // override the run_one_step function to aviod compilation error
    template <int dim>
    void Stokes<dim>::run_one_step(bool apply_nonzero_constraints,
                                   bool assemble_system)
    {
      (void)assemble_system;
      (void)apply_nonzero_constraints;
    }

    template <int dim>
    void Stokes<dim>::run_one_step()
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

      solution = 0;
      assemble();

      auto state = solve();

      present_solution = solution;

      pcout << std::scientific << std::left << " ITR = " << std::setw(3)
            << state.first << " RES = " << state.second << std::endl;

      if (time.time_to_output())
        {
          output_results(time.get_timestep());
        }
      if (parameters.simulation_type == "Fluid" && time.time_to_refine())
        {
          refine_mesh(1, 3);
        }
    }

    template <int dim>
    void Stokes<dim>::run()
    {
      pcout << "Running with PETSc on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      triangulation.refine_global(parameters.global_refinements[0]);
      setup_dofs();

      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   ++face)
                {
                  if (cell->face(face)->at_boundary())
                    {
                      const auto center = cell->face(face)->center();
                      const double x = center[0];
                      const double y = center[1];
                      const double tol = NumericalConstants::TOLERANCE;

                      // Determine and set boundary IDs based on face center
                      // coordinates
                      if (std::abs(y - 0.41) < tol)
                        {
                          cell->face(face)->set_boundary_id(3); // Top
                        }

                      else if (std::abs(y) < tol)
                        {
                          cell->face(face)->set_boundary_id(2); // Bottom
                        }
                      else if (std::abs(x) < tol)
                        {
                          cell->face(face)->set_boundary_id(0); // Left
                        }

                      else if (std::abs(x - 2.2) > tol)
                        {
                          cell->face(face)->set_boundary_id(4); // cylinder
                        }
                      /*
                      else if (std::abs(x - 2.2) > tol)
                      {
                        cell->face(face)->set_boundary_id(4); // Cylinder
                      }
                      */
                    }
                }
            }
        }
      set_up_boundary_values();
      initialize_system();

      run_one_step();

      while (time.end() - time.current() > 1e-12)
        {
          run_one_step();
        }
    }

    template class Stokes<2>;
    template class Stokes<3>;
  } // namespace MPI
} // namespace Fluid
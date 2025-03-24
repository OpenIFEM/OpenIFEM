#include "mpi_stokes.h"

namespace Fluid
{
  namespace MPI
  {

    // Define numerical constants to replace magic numbers
    /*
    namespace NumericalConstants
    {
      constexpr double DOMAIN_HEIGHT = 0.41; // Height of the domain
      constexpr double DOMAIN_LENGTH = 2.2;  // Length of the domain
      constexpr double INLET_U_MAX = 1;    // Maximum inlet velocity
      //constexpr double TOLERANCE = 1e-12;    // Tolerance for boundary checks
    } */                                      // namespace NumericalConstants
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
        SolverControl solver_control(src.size(), 1e-10 * src.l2_norm());
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
    template <int dim>
    class InletVelocity : public Function<dim>
    {
    public:
      InletVelocity(const double ramp_time)
        : Function<dim>(dim)
        , ramp_time(ramp_time)
        {
        } // in serial code I used dim+1, only one is correct; check in the
    future

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override
      {
        const double t = this->get_time();

        const double alpha = (ramp_time > 1e-14)
        ? std::min(1.0, t / ramp_time)
        : 1.0;


        const double y = p[1];
        const double H =
          NumericalConstants::DOMAIN_HEIGHT; // Height of the domain

        values(0) = alpha* 4.0 * NumericalConstants::INLET_U_MAX *
                    (y/H) *
                    (1-y/H);
        //values(0) = NumericalConstants::INLET_U_MAX ; //uniform inlet velocity

        for (unsigned int i = 1; i < dim + 1; ++i)
          values(i) = 0.0; // No velocity in other components
      }

    private:
           const double ramp_time;
    };*/

    template <int dim>
    Stokes<dim>::Stokes(parallel::distributed::Triangulation<dim> &tria,
                        const Parameters::AllParameters &parameters)
      : FluidSolver<dim>(tria, parameters), inlet_velocity(0.1)
    {
      Assert(
        parameters.fluid_velocity_degree - parameters.fluid_pressure_degree ==
          1,
        ExcMessage(
          "Velocity finite element should be one order higher than pressure!"));
    }

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

      // Inlet (Left Boundary) with Parabolic Velocity Profile
      // InletVelocity<dim> inlet_velocity;
      inlet_velocity.set_time(time.current());

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               inlet_velocity,
                                               constraints,
                                               fe.component_mask(velocities));
      /*
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               zero_velocity,
                                               constraints,
                                               fe.component_mask(velocities));
    */

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

      //  Point<dim> target_point(NumericalConstants::DOMAIN_LENGTH, 0.0);
      Point<dim> target_point(2.2, 0.0);

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
      system_matrix.clear();

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

      present_solution.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      fsi_acceleration.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      fsi_force_acceleration_part.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      fsi_force_stress_part.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      fsi_force.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      int stress_vec_size = dim + dim * (dim - 1) * 0.5;

      fsi_stress = std::vector<PETScWrappers::MPI::Vector>(
        stress_vec_size,
        PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                   locally_relevant_scalar_dofs,
                                   mpi_communicator));

      stress = std::vector<std::vector<PETScWrappers::MPI::Vector>>(
        dim,
        std::vector<PETScWrappers::MPI::Vector>(
          dim,
          PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                     mpi_communicator)));

      // Cell property
      setup_cell_property();

      if (initial_condition_field)
        {
          apply_initial_condition();
          constraints.distribute(present_solution);
        }
    }

    /*
    template <int dim>
    void Stokes<dim>::apply_initial_condition()
    {
        // 1) Wrap the user std::function in our custom wrapper.
     InternalWrappers::InitialVelocityWrapper<dim> velocity_function(
     *this->initial_condition_field, // your stored std::function
      dim                           // number of components (velocity)
  );
   // 2) Choose quadrature for projection
   const QGauss<dim> quadrature(parameters.fluid_velocity_degree + 1);

   // 3) Make a velocity-only component mask
   const FEValuesExtractors::Vector velocities(0);
   ComponentMask velocity_mask = fe.component_mask(velocities);

   PETScWrappers::MPI::BlockVector projection;
   projection.reinit(owned_partitioning, mpi_communicator);

    //projection.reinit(owned_partitioning,
    //relevant_partitioning,
    //mpi_communicator);
   VectorTools::project(
    MappingQGeneric<dim>(this->parameters.fluid_velocity_degree),
    dof_handler,
    constraints,
    quadrature,
    velocity_function,
    projection,
    &velocity_mask
  );
     projection.compress(VectorOperation::insert);
     constraints.distribute(projection);
     present_solution = projection;

    }

    */

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
      fsi_force_acceleration_part = 0;
      fsi_force_stress_part = 0;
      fsi_force = 0;
      /*
      std::vector<std::vector<PETScWrappers::MPI::Vector>>
        relevant_partition_stress =
          std::vector<std::vector<PETScWrappers::MPI::Vector>>(
            dim,
            std::vector<PETScWrappers::MPI::Vector>(
              dim,
              PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                         locally_relevant_scalar_dofs,
                                         mpi_communicator)));
      relevant_partition_stress = stress; */

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);

      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_normal_vectors |
                                         update_quadrature_points |
                                         update_JxW_values);

      FEValues<dim> scalar_fe_values(scalar_fe,
                                     volume_quad_formula,
                                     update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_face_q_points = face_quad_formula.size();

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                     dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);
      Vector<double> local_rhs_acceleration_part(dofs_per_cell);
      Vector<double> local_rhs_stress_part(dofs_per_cell);
      Vector<double> local_fsi_force(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);

      std::vector<Tensor<1, dim>> fsi_acc_values(n_q_points);

      std::vector<double> fsi_stress_value(n_q_points);

      std::vector<std::vector<double>> fsi_cell_stress =
        std::vector<std::vector<double>>(fsi_stress.size(),
                                         std::vector<double>(n_q_points));

      std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

      auto scalar_cell = scalar_dof_handler.begin_active();

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell, ++scalar_cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);

              const int ind = p[0]->indicator;

              fe_values.reinit(cell);
              scalar_fe_values.reinit(scalar_cell);

              local_matrix = 0;
              local_preconditioner_matrix = 0;
              local_rhs_acceleration_part = 0;
              local_rhs_stress_part = 0;
              local_rhs = 0;
              local_fsi_force = 0;

              fe_values[velocities].get_function_values(
                present_solution, current_velocity_values);

              for (unsigned int i = 0; i < fsi_stress.size(); i++)
                {
                  scalar_fe_values.get_function_values(fsi_stress[i],
                                                       fsi_stress_value);

                  fsi_cell_stress[i] = fsi_stress_value;
                }

              fe_values[velocities].get_function_values(fsi_acceleration,
                                                        fsi_acc_values);

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

                      grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                    }

                  SymmetricTensor<2, dim> fsi_stress_tensor;

                  if (ind != 0)
                    {
                      int stress_index = 0;
                      for (unsigned int k = 0; k < dim; k++)
                        {
                          for (unsigned int m = 0; m < k + 1; m++)
                            {
                              fsi_stress_tensor[k][m] =
                                fsi_cell_stress[stress_index][q];
                              stress_index++;
                            }
                        }
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
                            (2 * viscosity *
                               (symgrad_phi_u[i] *
                                symgrad_phi_u[j])      // viscous term
                             - div_phi_u[i] * phi_p[j] // pressure term
                             -
                             phi_p[i] * div_phi_u[j]) // incompressibility term
                            * fe_values.JxW(q);

                          if (fe.system_to_component_index(i).first < dim &&
                              fe.system_to_component_index(j).first < dim)
                            {

                              local_preconditioner_matrix(i, j) +=

                                ((rho / time.get_delta_t()) * phi_u[i] *
                                   phi_u[j]

                                 +
                                 (viscosity * scalar_product(grad_phi_u[i],
                                                             grad_phi_u[j]))) *

                                fe_values.JxW(q);
                            }

                          else if (fe.system_to_component_index(i).first ==
                                     dim &&
                                   fe.system_to_component_index(j).first == dim)
                            {
                              local_preconditioner_matrix(i, j) +=
                                //(1.0 / viscosity * phi_p[i] * phi_p[j]) *
                                (1.0 / viscosity) *
                                scalar_product(grad_phi_p[i], grad_phi_p[j]) *
                                fe_values.JxW(q);
                            }
                        }

                      /*
                      local_preconditioner_matrix(i, j) +=
                        ((rho / time.get_delta_t()) * phi_u[i] * phi_u[j] *
                         fe_values.JxW(q)) +
                        // ((2 / viscosity) * phi_p[i] * phi_p[j] *
                        //   fe_values.JxW(q));
                        1.0 / viscosity * phi_p[i] * phi_p[j] *
                          fe_values.JxW(q);
                      */

                      local_rhs(i) += phi_u[i] * gravity * fe_values.JxW(q);

                      local_rhs(i) += (rho / time.get_delta_t()) *
                                      (phi_u[i] * current_velocity_values[q]) *
                                      fe_values.JxW(q); // time derivative term

                      if (ind != 0)
                        {
                          local_rhs(i) +=
                            (scalar_product(grad_phi_u[i], fsi_stress_tensor) +
                             (fsi_acc_values[q] * parameters.solid_rho *
                              phi_u[i])) *
                            fe_values.JxW(q);

                          local_rhs_acceleration_part(i) +=
                            (fsi_acc_values[q] * parameters.solid_rho *
                             phi_u[i]) *
                            fe_values.JxW(q);

                          local_rhs_stress_part(i) +=
                            scalar_product(grad_phi_u[i], fsi_stress_tensor) *
                            fe_values.JxW(q);

                          local_fsi_force(i) +=
                            (scalar_product(grad_phi_u[i], fsi_stress_tensor) +
                             (fsi_acc_values[q] * parameters.solid_rho *
                              phi_u[i])) *
                            fe_values.JxW(q);
                        }
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

              constraints.distribute_local_to_global(
                local_rhs_acceleration_part,
                local_dof_indices,
                fsi_force_acceleration_part);

              constraints.distribute_local_to_global(local_rhs_stress_part,
                                                     local_dof_indices,
                                                     fsi_force_stress_part);

              constraints.distribute_local_to_global(
                local_fsi_force, local_dof_indices, fsi_force);
            }
        }

      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
      preconditioner_matrix.compress(VectorOperation::add);
      fsi_force_acceleration_part.compress(VectorOperation::add);
      fsi_force_stress_part.compress(VectorOperation::add);
      fsi_force.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned int, double> Stokes<dim>::solve()
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");
      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG prec_A;

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data_A;

      data_A.symmetric_operator = true;

      // data_A.relaxation_type_up =
      // dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
      // data_A.relaxation_type_down =
      // dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;

      prec_A.initialize(system_matrix.block(0, 0), data_A);

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG prec_S;

      dealii::LinearAlgebraPETSc::MPI::PreconditionAMG::AdditionalData data_S;

      data_S.symmetric_operator = true;

      // data_S.relaxation_type_up =
      // dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;
      // data_S.relaxation_type_down =
      // dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData::RelaxationType::Chebyshev;

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
                                   1e-10 * system_rhs.l2_norm());

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
    void Stokes<dim>::compute_energy_estimates()
    {
      TimerOutput::Scope timer_section(timer, "Compute energy estimates");

      // Set up FEValues
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_JxW_values);

      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values);

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_face_q_points = face_quad_formula.size();

      // Local accumulators
      double local_ke = 0.0;      // Kinetic energy
      double local_visc = 0.0;    // Viscous dissipation
      double local_p_div_u = 0.0; // Pressure-divergence term
      // double local_boundary_work = 0.0;
      double local_boundary_work_inlet = 0.0;
      double local_boundary_work_outlet = 0.0;

      // Quadrature point data
      std::vector<Tensor<1, dim>> velocity_values(n_q_points);
      std::vector<SymmetricTensor<2, dim>> sym_grad_u(n_q_points);
      std::vector<double> pressure_values(n_q_points);
      std::vector<double> div_u_values(n_q_points);

      // Loop over cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (!cell->is_locally_owned())
            continue;

          auto p = cell_property.get_data(cell);
          if (p[0]->indicator == 1) // Skip solid regions if applicable
            continue;

          fe_values.reinit(cell);

          // Extract field values
          fe_values[velocities].get_function_values(solution, velocity_values);
          fe_values[velocities].get_function_symmetric_gradients(solution,
                                                                 sym_grad_u);
          fe_values[pressure].get_function_values(solution, pressure_values);
          fe_values[velocities].get_function_divergences(solution,
                                                         div_u_values);

          // Quadrature loop
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Kinetic energy: 1/2 * rho * |u|^2
              double u_sq = velocity_values[q].norm_square();
              local_ke += 0.5 * parameters.fluid_rho * u_sq * fe_values.JxW(q);

              // Viscous dissipation: 2 * mu * eps:eps
              double eps_eps =
                sym_grad_u[q] * sym_grad_u[q]; // Double contraction
              local_visc +=
                2.0 * parameters.viscosity * eps_eps * fe_values.JxW(q);

              // Pressure-divergence term (diagnostic)
              local_p_div_u +=
                pressure_values[q] * div_u_values[q] * fe_values.JxW(q);
            }

          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell;
               ++face_no)
            {
              if (cell->at_boundary(face_no))
                {
                  fe_face_values.reinit(cell, face_no);

                  // We need the velocity, gradient(velocity), and pressure on
                  // the face
                  std::vector<Tensor<1, dim>> face_velocity(n_face_q_points);
                  std::vector<Tensor<2, dim>> face_grad_u(n_face_q_points);
                  std::vector<double> face_pressure(n_face_q_points);

                  fe_face_values[velocities].get_function_values(solution,
                                                                 face_velocity);
                  fe_face_values[velocities].get_function_gradients(
                    solution, face_grad_u);
                  fe_face_values[pressure].get_function_values(solution,
                                                               face_pressure);

                  for (unsigned int qf = 0; qf < n_face_q_points; ++qf)
                    {
                      const Tensor<1, dim> &u_face = face_velocity[qf];
                      const Tensor<1, dim> &n_face =
                        fe_face_values.normal_vector(qf);

                      SymmetricTensor<2, dim> symgrad_u_face;

                      for (unsigned int i = 0; i < dim; ++i)
                        {
                          for (unsigned int j = 0; j < dim; ++j)
                            {
                              symgrad_u_face[i][j] =
                                0.5 *
                                (face_grad_u[qf][i][j] + face_grad_u[qf][j][i]);
                            }
                        }

                      SymmetricTensor<2, dim> stress_face =
                        -face_pressure[qf] *
                          Physics::Elasticity::StandardTensors<dim>::I +
                        2.0 * parameters.viscosity * symgrad_u_face;

                      Tensor<1, dim> traction = stress_face * n_face;

                      double integrand = u_face * traction;

                      // record inlet and outlet seperately
                      const types::boundary_id b_id =
                        cell->face(face_no)->boundary_id();

                      if (b_id == 0) // Inlet boundary
                        {
                          local_boundary_work_inlet +=
                            integrand * fe_face_values.JxW(qf);
                        }
                      else if (b_id == 1)
                        {
                          local_boundary_work_outlet +=
                            integrand * fe_face_values.JxW(qf);
                        }
                      // local_boundary_work += integrand *
                      // fe_face_values.JxW(qf);
                    }
                }
            }
        }

      // Global reduction
      double global_kinetic_energy = 0.0;
      double global_viscous_energy = 0.0;
      double global_divergence_residual = 0.0;
      // double global_boundary_work = 0.0;
      double global_boundary_work_inlet = 0.0;
      double global_boundary_work_outlet = 0.0;

      MPI_Allreduce(&local_ke,
                    &global_kinetic_energy,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_visc,
                    &global_viscous_energy,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_p_div_u,
                    &global_divergence_residual,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_boundary_work_inlet,
                    &global_boundary_work_inlet,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      MPI_Allreduce(&local_boundary_work_outlet,
                    &global_boundary_work_outlet,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      // Output results
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          // --- BEGIN CHANGES ---
          // Modify the output columns to include inlet and outlet separately
          std::ofstream file("energy_estimates.txt",
                             time.current() == 0 ? std::ios::out
                                                 : std::ios::app);

          if (time.current() == 0)
            {
              file
                << "Time\tKinetic_Energy\tViscous_Dissipation\tPressure_Div_"
                   "Term"
                << "\tBoundary_Work_Inlet\tBoundary_Work_Outlet\n"; // <<<
                                                                    // CHANGED
            }

          file << time.current() << "\t" << global_kinetic_energy << "\t"
               << global_viscous_energy << "\t" << global_divergence_residual
               << "\t" << global_boundary_work_inlet << "\t" // <<< CHANGED
               << global_boundary_work_outlet << "\n";       // <<< CHANGED

          file.close();
        }
    }

    template <int dim>
    void Stokes<dim>::compute_ind_norms() const
    {
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);

      const FEValuesExtractors::Vector velocities(0);
      double local_sum = 0.0;
      double local_max = 0.0;

      std::vector<Tensor<1, dim>> velocity_values(volume_quad_formula.size());

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);
              if (p[0]->indicator == 1)
                {
                  fe_values.reinit(cell);

                  fe_values[velocities].get_function_values(solution,
                                                            velocity_values);

                  for (unsigned int q = 0; q < volume_quad_formula.size(); ++q)
                    {
                      const double vel_norm = velocity_values[q].norm();
                      local_sum += vel_norm * vel_norm * fe_values.JxW(q);
                      if (vel_norm > local_max)
                        local_max = vel_norm;
                    }
                }
            }
        }

      double global_sum = 0.0;
      double global_max = 0.0;

      MPI_Allreduce(&local_sum,  // sendbuf
                    &global_sum, // recvbuf
                    1,           // count
                    MPI_DOUBLE,  // datatype
                    MPI_SUM,     // operation
                    mpi_communicator);

      MPI_Allreduce(
        &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);

      global_sum = std::sqrt(global_sum);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream file_fluid_vel;

          if (time.current() == 0)
            {
              file_fluid_vel.open("ind_vel_norm.txt");
              file_fluid_vel << "Time"
                             << "\t"
                             << "L2-norm"
                             << "\t"
                             << "Max-norm"
                             << "\t"
                             << "\n";
            }
          else
            {
              file_fluid_vel.open("ind_vel_norm.txt", std::ios_base::app);
            }

          file_fluid_vel << time.current() << "\t" << global_sum << "\t"
                         << global_max << "\n";

          file_fluid_vel.close();
        }
    }

    template <int dim>
    void Stokes<dim>::compute_fluid_norms()
    {
      // Set up FEValues with necessary update flags
      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_gradients |
                                update_JxW_values);

      const FEValuesExtractors::Vector velocities(0);
      const unsigned int n_q_points = volume_quad_formula.size();

      // Local sums for velocity and divergence norms
      double local_sum_vel = 0.0;
      double local_sum_div = 0.0;

      // Vectors to store velocity and divergence values at quadrature points
      std::vector<Tensor<1, dim>> velocity_values(n_q_points);
      std::vector<double> divergence_values(n_q_points);

      // Loop over all locally owned cells
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {

              auto p = cell_property.get_data(cell);

              if (p[0]->indicator == 1)
                {
                  continue;
                }

              fe_values.reinit(cell);

              // Get velocity and divergence values from the solution
              fe_values[velocities].get_function_values(solution,
                                                        velocity_values);
              fe_values[velocities].get_function_divergences(solution,
                                                             divergence_values);

              // Compute contributions to the integrals
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // Velocity L2 norm: integrate |u|^2
                  double vel_norm_sq = velocity_values[q].norm_square();
                  local_sum_vel += vel_norm_sq * fe_values.JxW(q);

                  // Divergence L2 norm: integrate (div u)^2
                  double div_u = divergence_values[q];
                  local_sum_div += div_u * div_u * fe_values.JxW(q);
                }
            }
        }

      // Global reduction to sum contributions across all processes
      double global_sum_vel = 0.0;
      double global_sum_div = 0.0;
      MPI_Allreduce(&local_sum_vel,
                    &global_sum_vel,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);
      MPI_Allreduce(&local_sum_div,
                    &global_sum_div,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      // Compute the L2 norms
      double L2_norm_vel = std::sqrt(global_sum_vel);
      double L2_norm_div = std::sqrt(global_sum_div);

      // Write results to files from process 0
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream file_vel, file_div;

          // If first time step, write headers
          if (time.current() == 0)
            {
              file_vel.open("velocity_L2_norm.txt");
              file_vel << "Time\tL2_norm_velocity\n";
              file_div.open("divergence_L2_norm.txt");
              file_div << "Time\tL2_norm_divergence\n";
            }
          else
            {
              file_vel.open("velocity_L2_norm.txt", std::ios_base::app);
              file_div.open("divergence_L2_norm.txt", std::ios_base::app);
            }

          // Write current time and norms
          file_vel << time.current() << "\t" << L2_norm_vel << "\n";
          file_div << time.current() << "\t" << L2_norm_div << "\n";

          file_vel.close();
          file_div.close();
        }
    }

    template <int dim>
    void Stokes<dim>::compute_drag_lift_coefficients()
    {

      // 1) Set up FEValues on boundary faces
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_gradients |
                                         update_normal_vectors |
                                         update_JxW_values);

      const unsigned int n_face_q_points = face_quad_formula.size();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      // 2) Local accumulators for drag and lift
      double local_drag = 0.0;
      double local_lift = 0.0;

      // 3) Vectors to store shape function info at each quadrature
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);

      // 4) For evaluating final solution at quadrature
      std::vector<Tensor<1, dim>> velocity_values(n_face_q_points);
      std::vector<Tensor<2, dim>> velocity_gradients(n_face_q_points);
      std::vector<double> pressure_values(n_face_q_points);

      // For each locally owned cell:
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (!cell->is_locally_owned())
            continue;

          // Check each face
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->at_boundary(f))
                {
                  // If boundary_id == 4, we assume it's the cylinder boundary
                  if (cell->face(f)->boundary_id() == 2)
                    {
                      fe_face_values.reinit(cell, f);

                      // Get velocity, pressure, velocity gradients at face
                      // quadrature
                      fe_face_values[velocities].get_function_values(
                        solution, velocity_values);
                      fe_face_values[velocities].get_function_gradients(
                        solution, velocity_gradients);
                      fe_face_values[pressure].get_function_values(
                        solution, pressure_values);

                      // Integration over the face quadrature points
                      for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                          const Tensor<1, dim> &normal =
                            -fe_face_values.normal_vector(
                              q); // negative because its pointing inward the
                                  // cylinder hole!

                          // Compute stress tensor sigma = -p I + 2 mu
                          // symgrad(u) For Stokes, let mu =
                          // parameters.viscosity
                          SymmetricTensor<2, dim> grad_sym;
                          for (unsigned int i = 0; i < dim; ++i)
                            for (unsigned int j = 0; j < dim; ++j)
                              grad_sym[i][j] =
                                0.5 * (velocity_gradients[q][i][j] +
                                       velocity_gradients[q][j][i]);

                          const double p = pressure_values[q];
                          SymmetricTensor<2, dim> sigma =
                            -p * Physics::Elasticity::StandardTensors<dim>::I +
                            2.0 * parameters.viscosity * grad_sym;

                          // Traction vector t = sigma * n
                          // (SymmetricTensor<2,dim> times Tensor<1,dim>)
                          const Tensor<1, dim> traction = sigma * normal;

                          // Drag is x-component, Lift is y-component
                          // In 3D, you'd typically define an axis convention or
                          // use z for lift, etc.
                          const double tx = traction[0];
                          const double ty = traction[1];

                          // Accumulate the integrals
                          local_drag += tx * fe_face_values.JxW(q);
                          local_lift += ty * fe_face_values.JxW(q);
                        } // end face quadrature
                    }     // end if boundary_id == 4
                }         // end if at_boundary
            }             // end face loop
        }                 // end cell loop

      // 5) Sum drag and lift across all MPI processes
      double global_drag = 0.0;
      double global_lift = 0.0;

      MPI_Allreduce(
        &local_drag, &global_drag, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      MPI_Allreduce(
        &local_lift, &global_lift, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

      // 6) Compute dimensionless coefficients (example in 2D)
      // Suppose the cylinder diameter is D, and reference velocity is U_ref.
      // Then  (drag coefficient) =  F_x / (0.5 * rho * U_ref^2 * D)
      //        lift coefficient =  F_y / (0.5 * rho * U_ref^2 * D)
      // Here "rho" might be your parameters.fluid_rho
      // Be sure to use consistent units in your code!

      const double D = 0.1;        // example cylinder diameter
      const double U_ref = 0.9796; // mean velocity
      const double rho = parameters.fluid_rho;

      // Denominator for the usual 2D definition
      const double denominator = 0.5 * rho * U_ref * U_ref * D;

      const double drag_coefficient = global_drag / denominator;
      const double lift_coefficient = global_lift / denominator;

      // 7) Print or store the results
      pcout << std::endl
            << "----------------------------------------------------------"
            << std::endl
            << " Drag  = " << global_drag << "   -> C_D = " << drag_coefficient
            << std::endl
            << " Lift  = " << global_lift << "   -> C_L = " << lift_coefficient
            << std::endl
            << "----------------------------------------------------------"
            << std::endl
            << std::endl;

      // Optionally, write to a file if you like:
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          std::ofstream coeffs("drag_lift_history.txt", std::ios::app);
          coeffs << time.get_timestep() << "\t" << time.current() << "\t"
                 << drag_coefficient << "\t" << lift_coefficient << std::endl;
          coeffs.close();
        }
    }

    template <int dim>
    void Stokes<dim>::output_results(const unsigned int output_index) const
    {
      TimerOutput::Scope timer_section(timer, "Output results");
      pcout << "Writing results..." << std::endl;
      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.push_back("pressure");

      std::vector<std::string> fsi_force_names(dim,
                                               "fsi_force_acceleration_part");
      fsi_force_names.push_back("dummy_fsi_force_acc");

      std::vector<std::string> fsi_force_names_stress(dim,
                                                      "fsi_force_stress_part");
      fsi_force_names_stress.push_back("dummy_fsi_force_str");

      std::vector<std::string> fsi_force_names_total(dim, "fsi_force_total");
      fsi_force_names_total.push_back("dummy_fsi_force_total");

      std::vector<std::vector<PETScWrappers::MPI::Vector>> tmp_stress =
        std::vector<std::vector<PETScWrappers::MPI::Vector>>(
          dim,
          std::vector<PETScWrappers::MPI::Vector>(
            dim,
            PETScWrappers::MPI::Vector(locally_owned_scalar_dofs,
                                       locally_relevant_scalar_dofs,
                                       mpi_communicator)));
      tmp_stress = stress;

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);

      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(present_solution,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      data_out.add_data_vector(fsi_force_acceleration_part,
                               fsi_force_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      data_out.add_data_vector(fsi_force_stress_part,
                               fsi_force_names_stress,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      data_out.add_data_vector(fsi_force,
                               fsi_force_names_total,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      /*
      // Partition
      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");
      */

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

      // viscous stress
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[0][0], "Txx");
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[0][1], "Txy");
      data_out.add_data_vector(scalar_dof_handler, tmp_stress[1][1], "Tyy");

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
      run_one_step_new();
    }

    template <int dim>
    void Stokes<dim>::run_one_step_new()
    {

      std::cout.precision(6);
      std::cout.width(12);
      solution = present_solution;

      if (time.get_timestep() == 0)
        {
          output_results(0);
          compute_ind_norms();
          compute_fluid_norms();
          compute_energy_estimates();
        }

      time.increment();

      pcout << std::string(96, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;

      set_up_boundary_values(); // Update boundary conditions with current time
      assemble();

      auto state = solve();

      // compute_drag_lift_coefficients();
      // compute_ind_norms();
      compute_fluid_norms();
      compute_energy_estimates();

      update_stress();

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
    void Stokes<dim>::initialize_bcs()
    {
      // setup_dofs();
      FluidSolver<dim>::setup_dofs();
      /*
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

                      // Determine and set boundary IDs based on face center
                      // coordinates
                     if (std::abs(y - 1) < 1e-10)
                       {
                        cell->face(face)->set_boundary_id(3); // Top
                       }

                     else if (std::abs(y) < 1e-10)
                      {
                        cell->face(face)->set_boundary_id(2); // Bottom
                      }

                      else if (std::abs(x) < 1e-10)

                        {
                          cell->face(face)->set_boundary_id(0); // Left
                        }

                       else if (std::abs(x - 1) < 1e-10)
                        {
                          cell->face(face)->set_boundary_id(1); // right
                        }


                    }
                }
            }
        } */
      set_up_boundary_values();
      initialize_system();
    }

    template <int dim>
    void Stokes<dim>::run()
    {
      pcout << "Running with PETSc on "
            << Utilities::MPI::n_mpi_processes(mpi_communicator)
            << " MPI rank(s)..." << std::endl;

      triangulation.refine_global(parameters.global_refinements[0]);

      initialize_bcs();

      run_one_step_new();

      while (time.end() - time.current() > 1e-12)
        {
          run_one_step_new();
        }
    }

    template class Stokes<2>;
    template class Stokes<3>;
  } // namespace MPI
} // namespace Fluid
#include "stokes.h"

namespace Fluid
{

  // Define numerical constants to replace magic numbers
  namespace NumericalConstants
  {
    constexpr double DOMAIN_HEIGHT = 0.41; // Height of the domain
    constexpr double DOMAIN_LENGTH = 2.2; // X-position of the cylinder
    constexpr double INLET_U_MAX = 100;   // Maximum inlet velocity
    constexpr double TOLERANCE = 1e-10;   // Tolerance for boundary checks
  }                                       // namespace NumericalConstants

  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {

  public:
    InverseMatrix(const MatrixType &m,
                  const PreconditionerType &preconditioner);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

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
    Vector<double> &dst, const Vector<double> &src) const
  {
    SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    dst = 0;

    cg.solve(*matrix, dst, src, *preconditioner);
  }

  template <class PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);

    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
      A_inverse;

    mutable Vector<double> tmp1, tmp2;
  };

  template <class PreconditionerType>
  SchurComplement<PreconditionerType>::SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix),
      A_inverse(&A_inverse),
      tmp1(system_matrix.block(0, 0).m()),
      tmp2(system_matrix.block(0, 0).m())
  {
  }

  template <class PreconditionerType>
  void
  SchurComplement<PreconditionerType>::vmult(Vector<double> &dst,
                                             const Vector<double> &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
  }

  template <class PreconditionerA, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      const BlockSparseMatrix<double> &S,
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mpinv,
      const PreconditionerA &Apreconditioner);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp>>
      m_inverse;
    const PreconditionerA &a_preconditioner;

    mutable Vector<double> tmp;
  };

  template <class PreconditionerA, class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::
    BlockSchurPreconditioner(
      const BlockSparseMatrix<double> &S,
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mpinv,
      const PreconditionerA &Apreconditioner)
    : system_matrix(&S),
      m_inverse(&Mpinv),
      a_preconditioner(Apreconditioner),
      tmp(S.block(1, 1).m())
  {
  }

  template <class PreconditionerA, class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerA, PreconditionerMp>::vmult(
    BlockVector<double> &dst, const BlockVector<double> &src) const
  {
    // Form u_new = A^{-1} u
    a_preconditioner.vmult(dst.block(0), src.block(0));
    // Form tmp = - B u_new + p
    // (<code>SparseMatrix::residual</code>
    // does precisely this)
    system_matrix->block(1, 0).residual(tmp, dst.block(0), src.block(1));
    // Change sign in tmp
    tmp *= -1;
    // Multiply by approximate Schur complement
    // (i.e. a pressure mass matrix)
    m_inverse->vmult(dst.block(1), tmp);
  }

  template <int dim>
  Stokes<dim>::Stokes(Triangulation<dim> &tria,
                      const Parameters::AllParameters &parameters,
                      std::shared_ptr<Function<dim>> bc)
    : FluidSolver<dim>(tria, parameters, bc)
  {
    Assert(
      parameters.fluid_velocity_degree - parameters.fluid_pressure_degree == 1,
      ExcMessage(
        "Velocity finite element should be one order higher than pressure!"));
  }

  template <int dim>
  class InletVelocity : public Function<dim>
  {
  public:
    InletVelocity() : Function<dim>(dim + 1) {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override
    {

      const double y = p[1];
      const double H =
        NumericalConstants::DOMAIN_HEIGHT; // Height of the domain

       values(0) = NumericalConstants::INLET_U_MAX *
         (1.0 - std::pow((2.0 * y / H - 1), 2));
      //values(0) = NumericalConstants::INLET_U_MAX; // uniform inlet velocity

      for (unsigned int i = 1; i < dim + 1; ++i)
        values(i) = 0.0; // No velocity in other components
    }

  private:
  };

  /*
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
      BoundaryValues() : Function<dim>(dim + 1) {}

      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const override;

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &value) const override;
    };

    template <int dim>
    double BoundaryValues<dim>::value(const Point<dim> &p,
                                      const unsigned int component) const
    {
      Assert(component < this->n_components,
             ExcIndexRange(component, 0, this->n_components));

      if (component == 0)
        {
          if (p[0] < 0)
            return -1;
          else if (p[0] > 0)
            return 1;
          else
            return 0;
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
    */

  template <int dim>
  void Stokes<dim>::set_up_boundary_values()
  {
    constraints.clear();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim); // for fixing pressure only

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    /*
    // try to exclude top and bot corner nodes from the no-slip condition
     ComponentMask component_mask = fe.component_mask(velocities);


    // Create a set of boundary IDs to extract
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(2); // Bottom Wall
    boundary_ids.insert(3); // Top Wall

    IndexSet boundary_dofs = DoFTools::extract_boundary_dofs(dof_handler,
    component_mask, boundary_ids);

    std::vector<Point<dim>> v_support_points(dof_handler.n_dofs());
    MappingQGeneric<dim> v_mapping(parameters.fluid_velocity_degree);
    DoFTools::map_dofs_to_support_points(v_mapping, dof_handler,
    v_support_points);


    // Apply zero-velocity constraints, excluding corner nodes at x=2.2
    for (auto dof = boundary_dofs.begin(); dof != boundary_dofs.end(); ++dof)
    {
        const Point<dim> &p = v_support_points[*dof];
        if (std::abs(p[0] - 2.2) > 1e-12) // Exclude points at x=2.2
        {
            constraints.add_line(*dof);
            constraints.set_inhomogeneity(*dof, 0.0); // Zero velocity
        }
    }

    */

    Functions::ZeroFunction<dim> zero_velocity(dim);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             zero_velocity,
                                             constraints,
                                             fe.component_mask(velocities));

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

    // Inlet (Left Boundary) with Parabolic Velocity Profile
    InletVelocity<dim> inlet_velocity; // Set U_max as needed
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             inlet_velocity,
                                             constraints,
                                             fe.component_mask(velocities));

    // fix pressure at a given point
    // Build a map from DoFs to their support points
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());

    // Use a suitable mapping; if you're using higher-order elements, adjust
    // accordingly
    MappingQGeneric<dim> mapping(parameters.fluid_pressure_degree);

    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    // Now extract pressure DoFs
    IndexSet pressure_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure));

    Point<dim> target_point(NumericalConstants::DOMAIN_LENGTH, 0.0);

    types::global_dof_index fixed_pressure_dof = numbers::invalid_dof_index;

    double min_distance = std::numeric_limits<double>::max();

    for (auto dof = pressure_dofs.begin(); dof != pressure_dofs.end(); ++dof)
      {
        double distance = support_points[*dof].distance(target_point);
        if (distance < min_distance)
          {
            min_distance = distance;
            fixed_pressure_dof = *dof;
          }
      }

    // Ensure we found a valid DoF
    Assert(fixed_pressure_dof != numbers::invalid_dof_index,
           ExcInternalError());

    std::cout << "Constrained pressure DoF at: "
              << support_points[fixed_pressure_dof] << std::endl;

    constraints.add_line(fixed_pressure_dof);
    constraints.set_inhomogeneity(fixed_pressure_dof,
                                  0.0); // Set pressure to zero

    constraints.close();
  }

  template <int dim>
  void Stokes<dim>::initialize_system()
  {
    FluidSolver<dim>::initialize_system();
    A_preconditioner.reset();
    system_matrix.clear();
    preconditioner_matrix.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

    // not in fluid_solver.cpp, need to test if necessary
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
    //

    DoFTools::make_sparsity_pattern(
      dof_handler, coupling, dsp, constraints, false);

    sparsity_pattern.copy_from(dsp);

    BlockDynamicSparsityPattern preconditioner_dsp(dofs_per_block,
                                                   dofs_per_block);

    // not in fluid_solver.cpp, need to test if necessary
    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (((c == dim) && (d == dim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;

    //

    DoFTools::make_sparsity_pattern(dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    constraints,
                                    false);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);

    system_matrix.reinit(sparsity_pattern);
    preconditioner_matrix.reinit(preconditioner_sparsity_pattern);
    solution.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);

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
    // mass_matrix = 0; needs to be initialized!!
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
    const unsigned int u_dofs = fe.base_element(0).dofs_per_cell;
    const unsigned int p_dofs = fe.base_element(1).dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();
    const unsigned int n_face_q_points = face_quad_formula.size();

    AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
                ExcMessage("Wrong partitioning of dofs!"));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    // FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_preconditioner_matrix(dofs_per_cell,
                                                   dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);

    /*

    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);
    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    */

    std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);
    // std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)

      {
        auto p = cell_property.get_data(cell);
        // const int ind = p[0]->indicator;

        fe_values.reinit(cell);

        local_matrix = 0;
        local_preconditioner_matrix = 0;
        local_rhs = 0;
        // local_mass_matrix = 0;

        /*
        fe_values[velocities].get_function_values(evaluation_point,
                                              current_velocity_values);

        fe_values[velocities].get_function_gradients(
      evaluation_point, current_velocity_gradients);

        fe_values[pressure].get_function_values(evaluation_point,
                                            current_pressure_values);

        fe_values[velocities].get_function_values(present_solution,
                                              present_velocity_values);
        */

        fe_values[velocities].get_function_values(present_solution,
                                                  current_velocity_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
                symgrad_phi_u[k] =
                  fe_values[velocities].symmetric_gradient(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j <= i; ++j)
                  {

                    local_matrix(i, j) +=

                      (rho / time.get_delta_t()) * (phi_u[i] * phi_u[j]) *
                      fe_values.JxW(q); // time derivative term

                    local_matrix(i, j) +=
                      (2 * viscosity *
                         (symgrad_phi_u[i] * symgrad_phi_u[j]) // viscous term
                       - div_phi_u[i] * phi_p[j]               // pressure term
                       - phi_p[i] *
                           div_phi_u[j]) // divergence term (incompressibility)
                      * fe_values.JxW(q);

                    local_preconditioner_matrix(i, j) +=
                      ((rho / time.get_delta_t()) * phi_u[i] * phi_u[j] *
                       fe_values.JxW(q)) +
                      ((2 / viscosity) * phi_p[i] * phi_p[j] *
                       fe_values.JxW(q));
                    //(2/viscosity) * phi_p[i] * phi_p[j] *
                    // fe_values.JxW(q);
                  }

                local_rhs(i) += phi_u[i] * gravity * fe_values.JxW(q);

                local_rhs(i) += (rho / time.get_delta_t()) *
                                (phi_u[i] * current_velocity_values[q]) *
                                fe_values.JxW(q); // time derivative term
              }
          }

        // Impose pressure boundary here if specified, loop over faces on the
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
                    unsigned int p_bc_id = cell->face(face_n)->boundary_id();
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

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
            {
              local_matrix(i, j) = local_matrix(j, i);
              local_preconditioner_matrix(i, j) =
                local_preconditioner_matrix(j, i);
            }

        cell->get_dof_indices(local_dof_indices);

        // const AffineConstraints<double> &constraints_used =
        // use_nonzero_constraints ? nonzero_constraints : zero_constraints;

        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);

        constraints.distribute_local_to_global(local_preconditioner_matrix,
                                               local_dof_indices,
                                               preconditioner_matrix);
      }

    A_preconditioner =
      std::make_shared<typename InnerPreconditioner<dim>::type>();

    A_preconditioner->initialize(
      system_matrix.block(0, 0),
      typename InnerPreconditioner<dim>::type::AdditionalData());
  }

  template <int dim>
  std::pair<unsigned int, double> Stokes<dim>::solve()
  {
    TimerOutput::Scope timer_section(timer, "Solve linear system");

    /*
    // block schur complement method
    const SparseMatrix<double> &pressure_mass_matrix =
  preconditioner_matrix.block(1,1);

   SparseILU<double> pmass_preconditioner;
  pmass_preconditioner.initialize (pressure_mass_matrix,
  SparseILU<double>::AdditionalData());

  InverseMatrix<SparseMatrix<double>,SparseILU<double>>
  m_inverse (pressure_mass_matrix, pmass_preconditioner);

  BlockSchurPreconditioner<typename InnerPreconditioner<dim>::type,
                         SparseILU<double> >
  preconditioner (system_matrix, m_inverse, *A_preconditioner);

  SolverControl solver_control (system_matrix.m(),
                              1e-6*system_rhs.l2_norm());

  GrowingVectorMemory<BlockVector<double> > vector_memory;
  //SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
  //gmres_data.max_basis_size = 100;


  SolverGMRES<BlockVector<double> > gmres(solver_control, vector_memory);

   gmres.solve(system_matrix, solution, system_rhs,
            preconditioner);

    const AffineConstraints<double> &constraints_used =
      use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute(solution);

    return {solver_control.last_step(), solver_control.last_value()};
    */

    // /*
    // Schur complement method
    // solve for P
    const InverseMatrix<SparseMatrix<double>,
                        typename InnerPreconditioner<dim>::type>
      A_inverse(system_matrix.block(0, 0), *A_preconditioner);

    Vector<double> tmp(solution.block(0).size());

    Vector<double> schur_rhs(solution.block(1).size());
    A_inverse.vmult(tmp, system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);

    SchurComplement<typename InnerPreconditioner<dim>::type> schur_complement(
      system_matrix, A_inverse);

    SolverControl solver_control(solution.block(1).size(),
                                 1e-12 * schur_rhs.l2_norm());

    SolverCG<Vector<double>> cg(solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize(preconditioner_matrix.block(1, 1),
                              SparseILU<double>::AdditionalData());

    InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
      preconditioner_matrix.block(1, 1), preconditioner);

    cg.solve(schur_complement, solution.block(1), schur_rhs, m_inverse);

    constraints.distribute(solution);

    // solve for U
    system_matrix.block(0, 1).vmult(tmp, solution.block(1));
    tmp *= -1;
    tmp += system_rhs.block(0);

    A_inverse.vmult(solution.block(0), tmp);
    constraints.distribute(solution);

    return {solver_control.last_step(), solver_control.last_value()};
    //*/
  }

  template <int dim>

  void Stokes<dim>::output_results(const unsigned int output_index) const

  {
    TimerOutput::Scope timer_section(timer, "Output results");
    std::cout << "Writing results..." << std::endl;
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

    // Indicator
    Vector<float> ind(triangulation.n_active_cells());
    int i = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        auto p = cell_property.get_data(cell);
        ind[i++] = p[0]->indicator;
      }
    data_out.add_data_vector(ind, "Indicator");

    data_out.build_patches(parameters.fluid_pressure_degree);

    std::string basename = "fluid";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
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
    // (void)assemble_system;
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

    // solution = 0;

    assemble();

    auto state = solve();

    present_solution = solution;

    std::cout << std::scientific << std::left << " ITR = " << std::setw(3)
              << state.first << " RES = " << state.second << std::endl;

    // std::cout << std::scientific << std::left << " GMRES_ITR = " <<
    // std::setw(3)
    // << state.first << " GMRES_RES = " << state.second << std::endl;

    // update_stress();

    // Output
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
    triangulation.refine_global(parameters.global_refinements[0]);
    setup_dofs();
    // make_constraints();

    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (cell->face(face)->at_boundary())
              {
                const auto center = cell->face(face)->center();
                const double x = center[0];
                const double y = center[1];
                const double tol = NumericalConstants::TOLERANCE;

                if (std::abs(y - 0.41) < tol)
                  {
                    cell->face(face)->set_all_boundary_ids(3); // Top
                  }

                //   else if (std::abs(x - 2.2) < tol)
                // {
                // cell->face(face)->set_all_boundary_ids(1); // outlet is
                // traction-free bc, do nothing, even not set ID, if ID is set
                // here,
                // result will be wrong near the outlet
                //  }

                else if (std::abs(y - 0.0) < tol)
                  {
                    cell->face(face)->set_all_boundary_ids(2); // Bottom
                  }

                else if (std::abs(x - 0) < tol)
                  {
                    cell->face(face)->set_all_boundary_ids(0); // Left
                  }

             //   else if (std::abs(x - 1) < tol)
                //  {
                //    cell->face(face)->set_all_boundary_ids(3); // right
                //  }

                 else if (std::abs(x - 2.2) > tol)
                 {
                cell->face(face)->set_all_boundary_ids(4); // cylinder
                  }
              }
          }
      }

    set_up_boundary_values();

    initialize_system();

    // Time loop.
    // use_nonzero_constraints is set to true only at the first time step,
    // which means nonzero_constraints will be applied at the first iteration
    // in the first time step only, and never be used again.
    // This corresponds to time-independent Dirichlet BCs.

    run_one_step();

    while (time.end() - time.current() > 1e-12)
      {
        run_one_step();
      }
  }

  template class Stokes<2>;
  template class Stokes<3>;

} // namespace Fluid
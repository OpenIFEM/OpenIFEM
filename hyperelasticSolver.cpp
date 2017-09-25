#include "hyperelasticSolver.h"

namespace
{
  template<int dim>
  void PointHistory<dim>::setup(const Parameters::AllParameters& parameters)
  {
    if (parameters.type == "NeoHookean")
    {
      auto nh = std::dynamic_pointer_cast<NeoHookean<dim>>(this->material);
      Assert(nh, dealii::ExcInternalError());
      Assert(!parameters.C.empty(), dealii::ExcInternalError());
      nh.reset(new NeoHookean<dim>(parameters.C[0], parameters.rho));
      this->update(dealii::Tensor<2, dim>());
    }
    else
    {
      Assert(false, dealii::ExcNotImplemented());
    }
  }

  template<int dim>
  void PointHistory<dim>::update(const dealii::Tensor<2, dim>& Grad_u)
  {
    const dealii::Tensor<2, dim> F =
      dealii::Physics::Elasticity::Kinematics::F(Grad_u);
    this->material->updateData(F);
    this->FInv = dealii::invert(F);
    //FIXME: getTau and getJc are calling model specific functions
    // here we don't know the type of model, this is definitely bad.
    {
      auto nh = std::dynamic_pointer_cast<NeoHookean<dim>>(this->material);
      Assert(nh, dealii::ExcInternalError());
      this->tau = nh->getTau();
      this->Jc = nh->getJc();
    }
    this->dPsi_vol_dJ = this->material->get_dPsi_vol_dJ();
    this->d2Psi_vol_dJ2 = this->material->get_d2Psi_vol_dJ2();
  }

  template class PointHistory<2>;
  template class PointHistory<3>;
}

namespace IFEM
{
  using namespace dealii;
  template<int dim>
  HyperelasticSolver<dim>::HyperelasticSolver(const std::string& infile) :
    parameters(infile), vol(0.), 
    time(parameters.end_time, parameters.delta_t),
    timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
    degree(parameters.poly_degree), fe(FE_Q<dim>(parameters.poly_degree, dim)),
    dofHandler(tria), dofsPerCell(fe.dofs_per_cell), quadFormula(parameters.quad_order),
    quadFaceFormula(parameters.quad_order), numQuadPts(quadFormula.size()),
    numFaceQuadPts(quadFaceFormula.size()), uFe(0)
  {
  }

  template<int dim>
  HyperelasticSolver<dim>::~HyperelasticSolver()
  {
    this->dofHandler.clear();
  }

  template<int dim>
  void HyperelasticSolver<dim>::runStatics()
  {
    generateMesh();
    setup();
    output();
    time.increment();
    Vector<double> solution_delta(this->dofHandler.n_dofs());
    while (time.current() < time.end())
    {
      solution_delta = 0.0;
      solveNonlinearTimestep(solution_delta);
      solution += solution_delta;
      output();
      time.increment();
    }
  }

  template<int dim>
  struct HyperelasticSolver<dim>::PerTaskDataK
  {
    // cell_matrix and local_dof_indices are needed to
    // assemble the global matrix
    FullMatrix<double> cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskDataK(const unsigned int dofsPerCell) :
      cell_matrix(dofsPerCell, dofsPerCell),
      local_dof_indices(dofsPerCell) {}
    void reset()
    {
      cell_matrix = 0.0;
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchDataK
  {
    // These are needed to compute local matrices
    FEValues<dim> feValues;
    std::vector<std::vector<double>> Nx;
    std::vector<std::vector<Tensor<2, dim>>> grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;

    ScratchDataK(const FiniteElement<dim>& fe_cell,
      const QGauss<dim>& qf_cell, const UpdateFlag uf_cell) :
      feValues(fe_cell, qf_cell, uf_cell),
      Nx(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(), std::vector<Tensor<2, dim>>(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
        std::vector<SymmetricTensor<2, dim>>(fe_cell.dofs_per_cell)) {}

    void reset()
    {
      Assert(!Nx.empty(), ExcInternalError());
      const unsigned int n_q_points = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        Assert(Nx[q].size() == n_dofs_per_cell, ExcInternalError());
        Assert(grad_Nx[q].size() == n_dofs_per_cell, ExcInternalError());
        Assert(symm_grad_Nx[q].size() = n_dofs_per_cell, ExcInternalError());
        for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
        {
          Nx[q][k] = 0.0;
          grad_Nx[q][k] = 0.0;
          symm_grad_Nx[q][k] = 0.0;
        }
      }
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::PerTaskDataRHS
  {
    // cell_rhs and local_dof_indices are needed in assembly
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskDataRHS(const unsigned int dofs_per_cell) :
      cell_rhs(dofs_per_cell), local_dof_indices(dofs_per_cell) {}
    void reset()
    {
      cell_rhs = 0.0;
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchDataRHS
  {
    // Theses are needed to compute local RHS
    FEValues<dim> feValues;
    FEFaceValues<dim> feFaceValues;
    std::vector<std::vector<double>> Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;

    ScratchDataRHS(const ScratchData_RHS &rhs) :
      feValues(rhs.feValues.get_fe(), rhs.feValues.get_quadrature(),
        rhs.get_update_flags()),
      feFaceValues(rhs.feFaceValues.get_fe(), rhs.feFaceValues.get_quadrature(),
        rhs.feFaceValues.get_update_flags()),
      Nx(rhs.Nx), symm_grad_Nx(rhs.symm_grad_Nx)
    {}

    void reset()
    {
      const unsigned int numQuadPts = Nx.size();
      const unsigned int dofsPerCell = Nx[0].size();
      for (unsigned int q = 0; q < numQuadPts; ++q)
      {
        Assert(Nx[q].size() == dofsPerCell, ExcInternalError());
        Assert(symm_grad_Nx[q].size() == dofsPerCell, ExcInternalError());
        for (unsigned int k = 0; k < numQuadPts; ++k)
        {
          Nx[q][k] = 0.0;
          symm_grad_Nx[q][k] = 0.0;
        }
      }
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::PerTaskDataQPH
  {
    // Updating quadrature point history is purely local
    // No need to write out anything so this structure is empty
    void reset() {}
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchDataQPH
  {
    /**
     * In order to update PointHistory, we need access to the
     * displacement to compute its gradient at current quadrature point,
     * which is saved to feValues.
     * To avoid copy, use a reference.
     */
    const Vector<double> &solution;
    std::vector<Tensor<2, dim>> grad_u;
    FEValues<dim> feValues;

    ScratchDataQPH(const FiniteElement<dim> &fe_cell,
      const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
      const Vector<double> &soln) : solution(soln),
      grad_u(qf_cell.size()), feValues(fe_cell, qf_cell, uf_cell)
    {}

    ScratchDataQPH(const ScratchDataQPH &rhs) :
      solution(rhs.solution), grad_u(rhs.grad_u),
      feValues(rhs.feValues.get_fe(), rhs.feValues.get_quadrature(),
        rhs.feValues.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = grad_u.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        grad_u[q] = 0.0;
      }
    }
  };

  template<int dim>
  void HyperelasticSolver<dim>::generateMesh()
  {
    GridGenerator::hyper_rectangle(this->tria,
      (dim == 3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
      (dim == 3 ? Point<dim>(1.0, 1.0, 1.0) : Point<dim>(1.0, 1.0)),
      true);
    GridTools::scale(parameters.scale, tria);
    tria.refine_global(std::max(1U, parameters.global_refinement));
    this->vol = GridTools::volume(this->tria);
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;

    // The boundary id is hardcoded for now
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if(cell->face(face)->at_boundary() && 
          cell->face(face)->center()[1] == 1.0 * parameters.scale)
        {
          if(dim==3)
          {
            if(cell->face(face)->center()[0] < 0.5 * parameters.scale &&
              cell->face(face)->center()[2] < 0.5 * parameters.scale)
            {
              cell->face(face)->set_boundary_id(6);
            }
          }
          else
          {
            if(cell->face(face)->center()[0] < 0.5 * parameters.scale)
            {
              cell->face(face)->set_boundary_id(6);
            }
          }
        }
      }
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::systemSetup()
  {
    timer.enter_subsection("Setup system");
    dofHandler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dofHandler);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: " << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
              << std::endl;

    tangentMatrix.clear();
    DynamicSparsityPattern dsp(dofHandler.n_dofs(), dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(dofHandler, dsp, constraints, false);
    pattern.copy_from(dsp);
    tangentMatrix.reinit(pattern);
    systemRHS.reinit(dofHandler.n_dofs());
    solution.reinit(dofHandler.n_dofs());
    setupQPH();
    timer.leave_subsection();
  }

  template<int dim>
  void HyperelasticSolver<dim>::setupQPH()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadraturePointHistory.initialize(tria.begin_active(), tria.end(), numQuadPts);
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
        quadraturePointHistory.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());
      for (unsigned int q = 0; q < numQuadPts; ++q)
      {
        lqph[q]->setup(parameters);
      }
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::updateGlobalQPH(const Vector<double> &solution_delta)
  {
    timer.enter_subsection("Update QPH data");
    std::cout << " UQPH " << std::flush;

    const Vector<double> solution_total(getSolution(solution_delta));
    const UpdateFlags uf_QPH(update_values | update_gradients);
    PerTaskDataQPH per_task_data_QPH;
    ScratchDataQPH scratch_data_QPH(fe, qf_cell, uf_QPH, solution_total);

    WorkStream::run(dofHandler.begin_active(),
                    dofHandler.end(),
                    *this,
                    &HyperelasticSolver::updateLocalQPH,
                    &Solid::copyLocalToGlobalQPH,
                    scratch_data_QPH,
                    per_task_data_QPH);

    timer.leave_subsection();
  }

  template<int dim>
  void HyperelasticSolver<dim>::updateLocalQPH(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataQPH &scratch, PerTaskDataQPH &/*data*/)
  {
    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      quadraturePointHistory.get_data(cell);
    Assert(lqph.size() == numQuadPts, ExcInternalError());
    Assert(scratch.grad_u.size() == numQuadPts, ExcInternalError());
    scratch.reset();
    scratch.feValues.reinit(cell);
    scratch.feValues[uFe].get_function_gradients(scratch.solution, scratch.grad_u);
    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      lqph[q]->update(scratch.grad_u[q]);
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::solveNonlinearTimestep(Vector<double> &solution_delta)
  {
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
              << time.current() << "s" << std::endl;

    Vector<double> newton_update(dofHandler.n_dofs());

    errorResidual.reset();
    errorResidual0.reset();
    errorResidualNorm.reset();
    errorUpdate.reset();
    errorUpdate0.reset();
    errorUpdateNorm.reset();

    print_conv_header();

    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
    {
      std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

      tangentMatrix = 0.0;
      systemRHS = 0.0;
      assembleGlobalRHS();
      getErrorResidual(errorResidual);

      if(newton_iteration == 0)
      {
        errorResidual0 = errorResidual;
      }

      errorResidualNorm = errorResidual;
      errorResidualNorm.normalize(errorResidual0);

      if(newton_iteration > 0 && errorUpdateNorm.norm <= parameters.tol_u
          && errorResidualNorm.norm <= parameters.tol_f)
      {
        std::cout << " CONVERGED! " << std::endl;
        print_conv_footer();
        break;
      }

      assembleGlobalTangent();
      make_constraints(newton_iteration);
      constraints.condense(tangentMatrix, systemRHS);

      const std::pair<unsigned int, double>
      lin_solver_output = solveLinearSystem(newton_update);

      getErrorUpdate(newton_update, errorUpdate);
      if (newton_iteration == 0)
      {
        errorUpdate0 = errorUpdate;
      }

      errorUpdateNorm = errorUpdate;
      errorUpdateNorm.normalize(errorUpdate0);

      solution_delta += newton_update;
      updateGlobalQPH(solution_delta);

      std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                << std::scientific << lin_solver_output.first << "  "
                << lin_solver_output.second << "  " << error_residual_norm.norm
                << "  " << error_residual_norm.u << "  " << error_update_norm.norm
                << "  " << error_update_norm.u << "  " << std::endl;
    }

    AssertThrow(newton_iteration < parameters.max_iterations_NR,
      ExcMessage("No convergence in nonlinear solver!"));
  }

  template<int dim>
  void HyperelasticSolver<dim>::printConvHeader()
  {
    static const unsigned int width = 100;
    std::string splitter("_", width);
    std::cout << splitter << std::endl;
    std::cout << "           SOLVER STEP            "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     NU_NORM     "
              << " NU_U       " << std::endl;
    std::cout << splitter << std::endl;
  }

  template<int dim>
  void HyperelasticSolver<dim>::printConvFooter()
  {
    static const unsigned int width = 100;
    std::string splitter("_", width);
    std::cout << splitter << std::endl;
    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl;
  }

  template<int dim>
  double HyperelasticSolver<dim>::computeVolume() const
  {
    double volume = 0.0;
    FEValues<dim> feVals(fe, quadFormula, update_JxW_values);

    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      feVals.reinit(cell);
      const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
        quadraturePointHistory.get_data(cell);
      Assert(lqph.size() == numQuadPts, ExcInternalError());
      for (unsigned int q = 0; q < numQuadPts; ++q)
      {
        const double det = lqph[q]->getDetF();
        const double JxW = feVals.JxW(q);
        volume += det * JxW;
      }
    }
    Assert(volume > 0.0, ExcInternalError());
    return volume;
  }

  template<int dim>
  void HyperelasticSolver<dim>::getErrorResidual(Errors &residual)
  {
    Vector<double> res(dofHandler.n_dofs());
    for (unsigned int i = 0; i < dofHandler.n_dofs(); ++i)
    {
      if (!constraints.is_constrained(i))
      {
        res(i) = systemRHS(i);
      }
    }
    residual.norm = res.l2_norm();
  }


  template<int dim>
  void HyperelasticSolver<dim>::getErrorUpdate(const Vector<double> &newton_update,
    Errors &error_update)
  {
    Vector<double> error(dofHandler.n_dofs());
    for (unsigned int i = 0; i < dofHandler.n_dofs(); ++i)
    {
      if (!constraints.is_constrained(i))
      {
        error(i) = newton_update(i);
      }
    }
    error_update.norm = error.l2_norm();
  }

  template<int dim>
  Vector<double> HyperelasticSolver<dim>::getSolution(
    const Vector<double> &solution_delta) const
  {
    Vector<double> solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
  }

  template<int dim>
  void HyperelasticSolver<dim>::assembleGlobalK()
  {
    timer.enter_subsection("Assemble tangent matrix");
    std::cout << " ASM_K " << std::flush;

    tangentMatrix = 0.0;

    const UpdateFlags uf_cell(update_values    |
                              update_gradients |
                              update_JxW_values);

    PerTaskDataK per_task_data(dofsPerCell);
    ScratchDataK scratch_data(fe, quadFormula, uf_cell);

    WorkStream::run(dofHandler.begin_active(),
                    dofHandler.end(),
                    std::bind(&HyperelasticSolver<dim>::assembleLocalK,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&HyperelasticSolver<dim>::copy_local_to_global_K,
                              this,
                              std::placeholders::_1),
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();
  }

  template<int dim>
  void HyperelasticSolver<dim>::copyLocalToGlobalK(const PerTaskDataK &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        tangent_matrix.add(data.local_dof_indices[i],
                           data.local_dof_indices[j],
                           data.cell_matrix(i, j));
      }
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::assembleLocalK(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataK &scratch, PerTaskDataK &data) const
  {
    data.reset();
    scratch.reset();
    scratch.feValues.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == numQuadPts, ExcInternalError());

    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      const Tensor<2, dim> F_inv = lqph[q]->getFInv();
      for (unsigned int k = 0; k < dofsPerCell; ++k)
      {
        // TODO: remove these two lines
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        Assert(k_group == 0, ExcInternalError());
        scratch.grad_Nx[q][k] = scratch.feValues[uFe].gradient(k, q)*F_inv;
        scratch.symm_grad_Nx[q][k] = symmetrize(scratch.grad_Nx[q][k]);
      }
    }

    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      const Tensor<2, dim> tau = lqph[q]->getTau();
      const SymmetricTensor<4, dim> Jc = lqph[q]->getJc();
      const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q];
      const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q];
      const double JxW = scratch.feValues.JxW(q);

      // TODO: remove i_group j_group
      for (unsigned int i = 0; i < dofsPerCell; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;
        const unsigned int i_group = fe.system_to_base_index(i).first.first;
        for (unsigned int j = 0; j <= i; ++j)
        {
          const unsigned int component_j = fe.system_to_component_index(j).first;
          const unsigned int j_group = fe.system_to_base_index(j).first.first;
          Assert(i_group == 0, ExcInternalError());
          Assert(j_group == 0, ExcInternalError());
          data.cell_matrix(i, j) += symm_grad_Nx[i]*Jc*symm_grad_Nx[j]*JxW;
          if (component_i == component_j)
          {
            data.cell_matrix(i, j) += grad_Nx[i][component_i]*tau
              *grad_Nx[j][component_j]*JxW;
          }
        }
      }
    }

    for (unsigned int i = 0; i < dofsPerCell; ++i)
    {
      for (unsigned int j = i + 1; j < dofsPerCell; ++j)
      {
        data.cell_matrix(i, j) = data.cell_matrix(j, i);
      }
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::assembleGlobalRHS()
  {
    timer.enter_subsection("Assemble system right-hand side");
    std::cout << " ASM_R " << std::flush;
    sysRHS = 0.0;
    const UpdateFlags uf_cell(update_values |
                              update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_normal_vectors |
                              update_JxW_values);
    PerTaskDataRHS per_task_data(dofsPerCell);
    ScratchDataRHS scratch_data(fe, quadFormula, uf_cell, quadFaceFormula, uf_face);
    WorkStream::run(dofHandler.begin_active(),
                    dofHandler.end(),
                    std::bind(&HyperelasticSolver<dim>::assembleLocalRHS,
                              this,
                              std::placeholders::_1,
                              std::placeholders::_2,
                              std::placeholders::_3),
                    std::bind(&HyperelasticSolver<dim>::copyLocalToGlobalRHS,
                              this,
                              std::placeholders::_1),
                    scratch_data,
                    per_task_data);
    timer.leave_subsection();
  }

  template<int dim>
  void HyperelasticSolver<dim>::copyLocalToGlobalRHS(const PerTaskDataRHS &data)
  {
    for (unsigned int i = 0; i < dofsPerCell; ++i)
    {
      sysRHS(data.local_dof_indices[i]) += data.cell_rhs(i);
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::assembleLocalRHS(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataRHS &scratch, PerTaskDataRHS &data) const
  {
    data.reset();
    scratch.reset();
    scratch.feValues.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    const std::vector<std::shared_ptr<const PointHistory<dim>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == numQuadPts, ExcInternalError());

    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      const Tensor<2, dim> F_inv = lqph[q]->getFInv();
      for (unsigned int k = 0; k < dofsPerCell; ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        Assert(k_group == 0, ExcInternalError());
        scratch.symm_grad_Nx[q][k] = symmetrize(scratch.feValues[uFe].gradient(k, q)*F_inv);
      }
    }

    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      const SymmetricTensor<2, dim> tau = lqph[q]->getTau();
      const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx = scratch.symm_grad_Nx[q];
      const double JxW = scratch.feValues.JxW(q);
      for (unsigned int i = 0; i < dofsPerCell; ++i)
      {
        const unsigned int i_group = fe.system_to_base_index(i).first.first;
        Assert(i_group == 0, ExcInternalError());
        data.cell_rhs(i) -= (symm_grad_Nx[i]*tau)*JxW;
      }
    }

    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (cell->face(face)->at_boundary() == true && cell->face(face)->boundary_id() == 6)
      {
        scratch.feFaceValues.reinit(cell, face);
        for (unsigned int q = 0; q < numFaceQuadPts; ++q)
        {
          const Tensor<1, dim> &N = scratch.feFaceValues.normal_vector(q);
          static const double  p0 = -4.0/(parameters.scale*parameters.scale);
          const double time_ramp = (time.current()/time.end());
          const double pressure = p0*parameters.p_p0*time_ramp;
          const Tensor<1, dim> traction = pressure*N;
          for (unsigned int i = 0; i < dofsPerCell; ++i)
          {
            const unsigned int i_group = fe.system_to_base_index(i).first.first;
            Assert(i_group == 0, ExcInternalError());
            {
              const unsigned int component_i = fe.system_to_component_index(i).first;
              const double Ni = scratch.feFaceValues.shape_value(i, q);
              const double JxW = scratch.feFaceValues.JxW(q);
              data.cell_rhs(i) += (Ni*traction[component_i])*JxW;
            }
          }
        }
      }
    }
  }

  template<int dim>
  void HyperelasticSolver<dim>::makeConstraints(const int &it_nr)
  {
    std::cout << " CST " << std::flush;
    if (it_nr > 0)
    {
      return;
    }
    constraints.clear();

    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    {
      const int boundary_id = 0;
      VectorTools::interpolate_boundary_values(dofHandler,
                                               boundary_id,
                                               Functions::ZeroFunction<dim>(n_components),
                                               constraints,
                                               fe.component_mask(x_displacement));
    }
    {
      const int boundary_id = 2;
      VectorTools::interpolate_boundary_values(dofHandler,
                                               boundary_id,
                                               Functions::ZeroFunction<dim>(n_components),
                                               constraints,
                                               fe.component_mask(y_displacement));
    }

    if (dim==3)
    {
      const FEValuesExtractors::Scalar z_displacement(2);
      {
        const int boundary_id = 3;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                  boundary_id,
                                                  Functions::ZeroFunction<dim>(n_components),
                                                  constraints,
                                                  (fe.component_mask(x_displacement) |
                                                   fe.component_mask(z_displacement)));
      }
      {
        const int boundary_id = 4;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 boundary_id,
                                                 Functions::ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(z_displacement));
      }
      {
        const int boundary_id = 6;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 boundary_id,
                                                 Functions::ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 (fe.component_mask(x_displacement) |
                                                  fe.component_mask(z_displacement)));
      }
    }
    else
    {
      {
        const int boundary_id = 3;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 boundary_id,
                                                 Functions::ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
      }
      {
        const int boundary_id = 6;
        VectorTools::interpolate_boundary_values(dofHandler,
                                                 boundary_id,
                                                 Functions::ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
      }
    }
    constraints.close();
  }

  template<int dim>
  std::pair<unsigned int, double>
  HyperelasticSolver<dim>::solveLinearSystem(Vector<double> &newton_update)
  {
    unsigned int lin_it = 0;
    double lin_res = 0.0;

    timer.enter_subsection("Linear solver");
    std::cout << " SLV " << std::flush;
    if (parameters.type_lin == "CG")
    {
      const int solver_its = tangentMatrix.m()*parameters.max_iterations_lin;
      const double tol_sol = parameters.tol_lin*sysRHS.l2_norm();
      SolverControl solver_control(solver_its, tol_sol);
      GrowingVectorMemory<Vector<double>> GVM;
      SolverCG<Vector<double>> solver_CG(solver_control, GVM);
      PreconditionSelector<SparseMatrix<double>, Vector<double>>
      preconditioner(parameters.preconditioner_type, parameters.preconditioner_relaxation);
      preconditioner.use_matrix(tangentMatrix);
      solver_CG.solve(tangentMatrix, newton_update, sysRHS, preconditioner);
      lin_it = solver_control.last_step();
      lin_res = solver_control.last_value();
    }
    else
    {
      Assert(false, ExcMessage("Linear solver type not implemented"));
    }
    timer.leave_subsection();
    constraints.distribute(newton_update);
    return std::make_pair(lin_it, lin_res);
  }

  template <int dim>
  void HyperelasticSolver<dim>::output() const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
      DataComponentInterpretation::component_is_part_of_vector);
    std::vector<std::string> solution_name(dim, "displacement");
    data_out.attach_dof_handler(dofHandler);
    data_out.add_data_vector(solution, solution_name, DataOut<dim>::type_dof_data,
      data_component_interpretation);
    Vector<double> soln(solution.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
    {
      soln(i) = solution_n(i);
    }
    MappingQEulerian<dim> q_mapping(degree, dofHandler, soln);
    data_out.build_patches(q_mapping, degree);
    std::ostringstream filename;
    filename << "solution-" << dim << "d-" << time.get_timestep() << ".vtk";
    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
  }
}

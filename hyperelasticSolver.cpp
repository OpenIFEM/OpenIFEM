#include "hyperelasticSolver.h"

namespace IFEM
{
  using namespace dealii;
  template<int dim>
  HyperelasticSolver<dim>::HyperelasticSolver(const std::string& infile) :
    parameters(infile), vol_reference(0.), 
    time(parameters.end_time, parameters.delta_t),
    timer(std::cout, TimerOutput::summary, TimerOutput::wall_times),
    degree(parameters.poly_degree), fe(FE_Q<dim>(parameters.poly_degree, dim)),
    dofHandler(tria), dofsPerCell(fe.dofs_per_cell), quadFormula(parameters.quad_order),
    quadFaceFormula(parameters.quad_order), numQuadPts(quadFormula.size()),
    numFaceQuadPts(quadFaceFormula.size())
  {
  }

  template<int dim>
  void HyperelasticSolver<dim>::runStatics()
  {
    generateMesh();
    setup();
    time.increment();
    Vector<double> solution_delta(this->dofHandler.n_dofs());
    while (time.current() < time.end())
    {
      solution_delta = 0.0;
      solve_nonlinear_timestep(solution_delta);
      solution += solution_delta;
      output();
      time.increment();
    }
  }

  template<int dim>
  struct HyperelasticSolver<dim>::PerTaskData_K
  {
    FullMatrix<double> cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_K(const unsigned int dofsPerCell) :
      cell_matrix(dofsPerCell, dofsPerCell),
      local_dof_indices(dofsPerCell) {}
    void reset()
    {
      cell_matrix = 0.0;
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchData_K
  {
    FEValues<dim> feValues;
    std::vector<std::vector<double>> Nx;
    std::vector<std::vector<Tensor<2, dim>>> grad_Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;

    ScratchData_K(const FiniteElement<dim>& fe_cell,
      const QGauss<dim>& qf_cell, const UpdateFlag uf_cell) :
      feValues(fe_cell, qf_cell, uf_cell),
      Nx(qf_cell.size(), std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(), std::vector<Tensor<2, dim>>(fe_cell.dofs_per_cell)),
      symm_grad_Nx(qf_cell.size(),
        std::vector<SymmetricTensor<2, dim>>(fe_cell.dofs_per_cell)) {}

    void reset()
    {
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
  struct HyperelasticSolver<dim>::PerTaskData_RHS
  {
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_RHS(const unsigned int dofs_per_cell) :
      cell_rhs(dofs_per_cell), local_dof_indices(dofs_per_cell) {}
    void reset()
    {
      cell_rhs = 0.0;
    }
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchData_RHS
  {
    FEValues<dim> feValues;
    FEFaceValues<dim> feFaceValues;
    std::vector<std::vector<double>> Nx;
    std::vector<std::vector<SymmetricTensor<2, dim>>> symm_grad_Nx;

    ScratchData_RHS(const ScratchData_RHS &rhs) :
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
  struct HyperelasticSolver<dim>::PerTaskData_UQPH
  {
    void reset() {}
  };

  template<int dim>
  struct HyperelasticSolver<dim>::ScratchData_UQPH
  {
    const Vector<double> &solution;
    std::vector<Tensor<2, dim>> solution_grad_u;
    FEValues<dim> feValues;

    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
      const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
      const Vector<double> &soln) : solution(soln),
      solution_grad_u(qf_cell.size()), feValues(fe_cell, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs) :
      solution(rhs.solution), solution_grad_u(rhs.solution_grad_u),
      feValues(rhs.feValues.get_fe(), rhs.feValues.get_quadrature(),
        rhs.feValues.get_update_flags())
    {}
  };

  template<int dim>
  void HyperelasticSolver<dim>::generateMesh()
  {
    GridGenerator::subdivided_hyper_rectangle(this->tria, std::vector<unsigned int>{32, 4, 4},
      (dim == 3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
      (dim == 3 ? Point<dim>(1.0, 1.0, 1.0) : Point<dim>(1.0, 1.0)),
      true);
    this->vol_reference = GridTools::volume(this->tria);
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
  }
}

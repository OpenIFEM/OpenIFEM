#ifndef HYPERELASTIC_SOLVER
#define HYPERELASTIC_SOLVER

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>

#include "neoHookean.h"
#include "parameters.h"

namespace
{
  /** This struct will be more useful if
   * mixed formulation is used.
   */
  struct Errors
  {
    Errors() : norm(1.0) {}
    void reset() { norm = 1.0; }
    void normalize(const Errors &rhs)
    {
      if (rhs.norm != 0.0)
        {
          norm /= rhs.norm;
        }
    }
    double norm;
  };

  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0), current(0.0), end(time_end), delta(delta_t)
    {
    }
    virtual ~Time() {}
    double getCurrent() const { return current; }
    double getEnd() const { return end; }
    double getDelta() const { return delta; }
    unsigned int getTimestep() const { return timestep; }
    void increment()
    {
      current += delta;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double current;
    const double end;
    const double delta;
  };

  /*! Data to store at the quadrature points.
   *  We cache the kinematics information at quadrature points
   *  by storing a PointHistory at each cell,
   *  so that they can be conveniently accessed in the assembly
   *  or post processing. We also store a material pointer
   *  in case different material properties are used at
   *  different locations.
   */
  template <int dim>
  class PointHistory
  {
    using ST = dealii::Physics::Elasticity::StandardTensors<dim>;

  public:
    PointHistory()
      : FInv(ST::I),
        tau(dealii::SymmetricTensor<2, dim>()),
        Jc(dealii::SymmetricTensor<4, dim>()),
        dPsi_vol_dJ(0.0),
        d2Psi_vol_dJ2(0.0)
    {
    }
    virtual ~PointHistory() {}
    /** Initialize the members with the input parameters */
    void setup(const IFEM::Parameters::AllParameters &);
    /**
     * Update the state with the displacement gradient
     * in the reference configuration.
     */
    void update(const IFEM::Parameters::AllParameters &,
                const dealii::Tensor<2, dim> &);
    double getDetF() const { return material->getDetF(); }
    const dealii::Tensor<2, dim> &getFInv() const { return FInv; }
    const dealii::SymmetricTensor<2, dim> &getTau() const { return tau; }
    const dealii::SymmetricTensor<4, dim> &getJc() const { return Jc; }
    double get_dPsi_vol_dJ() const { return dPsi_vol_dJ; }
    double get_d2Psi_vol_dJ2() const { return d2Psi_vol_dJ2; }

  private:
    std::shared_ptr<IFEM::HyperelasticMaterial<dim>> material;
    dealii::Tensor<2, dim> FInv;
    dealii::SymmetricTensor<2, dim> tau;
    dealii::SymmetricTensor<4, dim> Jc;
    double dPsi_vol_dJ;
    double d2Psi_vol_dJ2;
  };
}

namespace IFEM
{
  /*! \brief Solver for hyperelastic materials
   *
   *  Unlike LinearElasticSolver, we do not store material in this solver.
   *  Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_44.html
   */
  // FIXME: It should be derived from SolverBase
  template <int dim>
  class HyperelasticSolver
  {
  public:
    HyperelasticSolver(const std::string &infile = "parameters.prm");
    HyperelasticSolver(const IFEM::Parameters::AllParameters &);
    ~HyperelasticSolver() { this->dofHandler.clear(); }
    void runStatics();

  private:
    /**
     * Forward-declare some structures for assembly using WorkStream
     * ScratchData* are used to store the inputs for the WorkStream,
     * and PerTaskData* are used to store the results.
     */
    struct PerTaskDataK;
    struct ScratchDataK;
    struct PerTaskDataRHS;
    struct ScratchDataRHS;
    struct PerTaskDataQPH; // QPH: Quadrature PointHistory
    struct ScratchDataQPH;

    void generateMesh();
    void systemSetup();

    // Tangent matrix
    void assembleGlobalK();
    void assembleLocalK(
      const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      ScratchDataK &scratch,
      PerTaskDataK &data) const;
    void copyLocalToGlobalK(const PerTaskDataK &);

    // System RHS
    void assembleGlobalRHS();
    void assembleLocalRHS(
      const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
      ScratchDataRHS &scratch,
      PerTaskDataRHS &data) const;
    void copyLocalToGlobalRHS(const PerTaskDataRHS &);

    // Apply boundary conditions
    void makeConstraints(const int &);

    // Set up the quadrature point history
    void setupQPH();
    /**
     * update the quadrature point history
     * This is also done in a WorkStream manner.
     * Although in this case we don't need to copy anything from
     * local to global - everything is local, we still need
     * write an empty function to make the WorkStream happy.
     */
    void updateGlobalQPH(const dealii::Vector<double> &);
    void updateLocalQPH(
      const typename dealii::DoFHandler<dim>::active_cell_iterator &,
      ScratchDataQPH &,
      PerTaskDataQPH &);
    void copyLocalToGlobalQPH(const PerTaskDataQPH &) {}

    // Using Newton iteration to solve for a nonlinear timestep.
    void solveNonlinearTimestep(dealii::Vector<double> &);
    /* Solve a linear equation, return the number of iterations and residual. */
    std::pair<unsigned int, double> solveLinearSystem(dealii::Vector<double> &);
    // Given the increment of the solution, return the current solution.
    dealii::Vector<double> getSolution(const dealii::Vector<double> &) const;
    void output() const;

    Parameters::AllParameters parameters;
    double vol;
    dealii::Triangulation<dim> tria;
    Time time;
    mutable dealii::TimerOutput timer; // Record the time profile of the program

    /**
     * We store a PointHistory structure at every quadrature point,
     * so that kinematics information like F as well as material properties
     * can be cached.
     */
    dealii::CellDataStorage<typename dealii::Triangulation<dim>::cell_iterator,
                            PointHistory<dim>>
      quadraturePointHistory;

    const unsigned int degree;
    const dealii::FESystem<dim> fe;
    dealii::DoFHandler<dim> dofHandler;
    const unsigned int dofsPerCell;
    const dealii::QGauss<dim> quadFormula;
    const dealii::QGauss<dim - 1> quadFaceFormula;
    const unsigned int numQuadPts;
    const unsigned int numFaceQuadPts;
    // Tells dealii to view the dofs as a vector when necessary
    const dealii::FEValuesExtractors::Vector uFe;

    dealii::ConstraintMatrix constraints;
    dealii::SparsityPattern pattern;
    dealii::SparseMatrix<double> tangentMatrix;
    dealii::Vector<double> systemRHS;
    dealii::Vector<double> solution;

    /**
     * errorResidual: norm of the residual at a Newton iteration
     * errorResidual0: norm of the residual at the first iteration
     * errorResidualNorm: errorResidual/errorResidual0
     * errorUpdate: norm of the solution increment
     * errorUpdate0: norm of the solution increment at the first iteration
     * errorUpdateNorm: errorUpdate/errorUpdate0
     */
    Errors errorResidual, errorResidual0, errorResidualNorm, errorUpdate,
      errorUpdate0, errorUpdateNorm;

    // Reture the residual in the Newton iteration
    void getErrorResidual(Errors &);
    // Compute the l2 norm of the solution increment
    void getErrorUpdate(const dealii::Vector<double> &, Errors &);

    // Return the current volume of the geometry
    double computeVolume() const;

    // Print the header and footer of the output table
    static void printConvHeader();
    void printConvFooter();
  };
}

#endif

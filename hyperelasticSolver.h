#ifndef HYPERELASTIC_SOLVER
#define HYPERELASTIC_SOLVER

#include "solverBase.h"
#include "parameters.h"
#include "utilities.h"

namespace IFEM
{
  /*! \brief Solver for hyperelastic materials
   *
   *  Unlike LinearElasticSolver, we do not store material in this solver.
   *  Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_44.html
   */
  //FIXME: It should be derived from SolverBase
  template<int dim>
  class HyperelasticSolver
  {
  public:
    HyperelasticSolver(const std::string& infile = "parameters.prm");
    ~HyperelasticSolver() {dofHandler.clear();}
    void runStatics(const std::string& fileName = "");
  private:
    // Forward-declare some structures for assembly using WorkStream
    struct PerTaskData_K;
    struct ScratchData_K;
    struct PerTaskData_RHS;
    struct ScratchData_RHS;
    struct PerTaskData_UQPH; // Updating Quadrature PointHistory
    struct ScratchData_UQPH;

    void generateMesh();
    void setup();
    void assemble_global_K();
    void assemble_local_K();
    void copy_local_to_global_K(const PerTaskData_K&);
    void assemble_global_rhs();
    void assemble_local_rhs();
    void copy_local_to_global_rhs(const PerTaskData_RHS&);

    void make_constraints(const int&);
    void setup_qph();
    void update_global_qph_incremental(const dealii::Vector<double>&);
    void update_local_qph_incremental(
      const typename dealii::DoFHandler<dim>::active_cell_iterator&,
      ScratchData_UQPH&, PerTaskData_UQPH&);
    // No need to copy
    void copy_local_to_global_UQPH(const PerTaskData_UQPH&) {}

    void solve_nonlinear_timestep(dealii::Vector<double>&);
    std::pair<unsigned int, double> solve_linear_system(dealii::Vector<double>&);
    dealii::Vector<double> get_solution(const dealii::Vector<double>&) const;
    void output() const;

    Parameters::AllParameters parameters;
    double vol_reference;
    dealii::Triangulation<dim> tria;
    Time time;
    mutable dealii::TimerOutput timer;

    dealii::CellDataStorage<typename dealii::Triangulation<dim>::cell_iterator,
      PointHistory<dim>> quadrature_point_history;
    const unsigned int degree;
    const dealii::FESystem<dim> fe;
    dealii::DoFHandler<dim> dofHandler;
    const unsigned int dofsPerCell;
    const dealii::QGauss<dim> quadFormula;
    const dealii::QGauss<dim-1> quadFaceFormula;
    const unsigned int numQuadPts;
    const unsigned int numFaceQuadPts;

    dealii::ConstraintMatrix constraints;
    dealii::SparsityPattern pattern;
    dealii::SparseMatrix<double> sysMatrix;
    dealii::Vector<double> sysRhs;
    dealii::Vector<double> solution;

    Errors error_residual, error_residual_0, error_residual_norm,
           error_update, error_update_0, error_update_norm;
    void getErrorResidual(Errors&);
    void getErrorUpdate(const dealii::Vector<double>&, Errors&);
    std::pair<double, double> getErrorDilation() const;
    double compute_vol_current() const;

    static void printConvHeader();
    static void printConvFooter();
  };
}

#endif

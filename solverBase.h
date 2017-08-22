#ifndef SOLVER_BASE
#define SOLVER_BASE

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>
#include <exception>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using namespace dealii;
using namespace std;

/*! \brief Base class for all solvers.
 *
 * It implements some general functions.
 */
template<int dim>
class SolverBase
{
public:
  SolverBase(int order = 1) : dofHandler(tria), fe(FE_Q<dim>(order), dim) {}
  ~SolverBase() {this->dofHandler.clear();}
  /** Mesh generator.
   *  deal.II can generate simple meshes, we use those generators if
   *  we only need simple geometry. Currently it is not automated,
   *  one has to change the code in this function for different geometries.
   */
  void generateMesh();
  /**
   * Abaqus reader.
   * This function reads Abaqus input file. It constructs a triangulation
   * and number the boundaries based on the input file. The details of
   * the bc and load in Abaqus will be ignored. Because of the design of
   * deal.II, all the locations where a Dirichlet or a Neumann bc is
   * applied must be associated with a surface named SS<indicator>
   * where indicator is an int.
   */
  void readMesh(const std::string&);
  /**
   * BC reader.
   * We use a json file to denote the boundary conditions. The format is:
   * [
   *   {"type" : "traction", "boundary_id" : 3, "value" : [0.0, -1e-4]},
   *   {"type" : "displacement", "boundary_id" : 4, "dof" : [1, 1], "value" : [0.0, 0.0]}
   *   {"type" : "pressure", "boundary_id" : 1, "value" : 1.0},
   *   {"type" : "gravity", "value" : [0.0, 0.0]}
   * ]
   */
  void readBC(const std::string& filename = "bc.json");
  /**
   * Set up the dofHandler, reorder the grid, sparsity pattern
   * and initialize the matrix, solution, and rhs.
   */
  void setup();
  /**
   * Apply the Dirichlet bc.
   */
  void applyBC();
  /**
   * Solve the equation.
   */
  void solve();
  /**
   * Output in vtk format. It only outputs the displacement,
   * should be overriden by specific solvers.
   */
  void output(const unsigned int) const;
protected:
  Triangulation<dim> tria;
  DoFHandler<dim> dofHandler;
  FESystem<dim> fe;
  SparsityPattern pattern;
  SparseMatrix<double> sysMatrix;
  Vector<double> solution;
  Vector<double> sysRhs;

  struct BoundaryCondition
  {
    Tensor<1, dim> gravity;
    map<unsigned int, Tensor<1, dim>> traction;
    map<unsigned int, double> pressure;
    /** A mapping between the boundary id and a pair of direction mask
     and the corresponding value. */
    map<unsigned int,
      pair<vector<bool>, vector<double>>> displacement;
  } bc;
};

#include "solverBase.tpp"

#endif

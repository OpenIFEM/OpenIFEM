#ifndef LINEAR_ELASTIC_SOLVER
#define LINEAR_ELASTIC_SOLVER

#include "solverBase.h"

/*! \brief Solver for linear elastic materials
 *
 * Currently dynamics is not implemented.
 * Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_8.html
 */
template<int dim>
class LinearElasticSolver : public SolverBase<dim>
{
public:
  LinearElasticSolver(int order = 1): SolverBase<dim>(order) {};
  /*
   * Assemble the system matrix and system rhs at the same time.
   */
  void assemble();
};

#include "linearElasticSolver.tpp"

#endif

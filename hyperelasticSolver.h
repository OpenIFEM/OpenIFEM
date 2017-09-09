#ifndef HYPERELASTIC_SOLVER
#define HYPERELASTIC_SOLVER

#include "solverBase.h"
#include "neoHookean.h"

namespace IFEM
{
  extern template class SolverBase<2>;
  extern template class SolverBase<3>;

  /*! \brief Solver for hyperelastic materials
   *
   *  Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_44.html
   */
  template<int dim>
  class HyperelasticSolver : public SolverBase<dim>
  {
  public:
    HyperelasticSolver(const std::string& infile = "parameters.prm"):
      SolverBase<dim>(infile) {}
  };
}

#endif

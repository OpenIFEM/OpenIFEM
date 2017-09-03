#ifndef LINEAR_ELASTIC_SOLVER
#define LINEAR_ELASTIC_SOLVER

#include "solverBase.h"
#include "linearMaterial.h"

namespace IFEM
{
  extern template class SolverBase<2>;
  extern template class SolverBase<3>;

  /*! \brief Solver for linear elastic materials
  *
  * Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_8.html
  */
  template<int dim>
  class LinearElasticSolver : public SolverBase<dim>
  {
  public:
    LinearElasticSolver(int order = 1): SolverBase<dim>(order) {};
    void runStatics(const std::string& fileName = "");
    void runDynamics(const std::string& fileName = "");
    /*
    * Assemble the system matrix and system rhs at the same time.
    */
    void assemble();
    void setMaterial(const LinearMaterial<dim>& mat) {this->material = mat;}
    void output(const unsigned int) const override;
  private:
    LinearMaterial<dim> material;
  };
}

#endif

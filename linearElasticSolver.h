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
  private:
    LinearMaterial<dim> material;
    /**
     * The scratch to be used in multithread assembly.
     * We could as well construct them as local variables but repeatedly constructing
     * FEValues is expensive.
     */
    struct AssemblyScratchData
    {
      AssemblyScratchData(const dealii::FiniteElement<dim>&, const dealii::QGauss<dim>&,
        const dealii::QGauss<dim-1>&);
      AssemblyScratchData(const AssemblyScratchData &scratch);

      dealii::FEValues<dim> fe_values;
      dealii::FEFaceValues<dim> fe_face_values;
    };
    /**
     * Instead of using mutex, we store the result of the local assembly in a structure
     * and then copy it into the global matrix SERIALLY.
     */
    struct AssemblyCopyData
    {
      dealii::FullMatrix<double> cell_matrix;
      dealii::Vector<double> cell_rhs;
      std::vector<unsigned int> dof_indices;
    };
    /**
     * Run the localAssemble in parallel and the scatter in serial.
     */
    void globalAssemble();
    /**
     * Assemble the local matrix and rhs at a given cell.
     */
    void localAssemble(const typename dealii::DoFHandler<dim>::active_cell_iterator&,
      AssemblyScratchData &, AssemblyCopyData &);
    /**
     * Copy the local data to global. This should run serially.
     */
    void scatter(const AssemblyCopyData &);
    void setMaterial(const LinearMaterial<dim>& mat) {this->material = mat;}
    void output(const unsigned int) const override;
  };
}

#endif

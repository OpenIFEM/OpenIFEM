#ifndef MPI_SHARED_LINEAR_ELASTICITY
#define MPI_SHARED_LINEAR_ELASTICITY

#include "linear_elastic_material.h"
#include "mpi_shared_solid_solver.h"

namespace Solid
{
  extern template class LinearElasticMaterial<2>;
  extern template class LinearElasticMaterial<3>;

  namespace MPI
  {
    using namespace dealii;

    /*! \brief A fully distributed parallel time-dependent solver for linear
     * elasticity.
     *
     * Both the triangulation and the dofs are fully distributed, the algebraic
     * operations are done using PETSc wrappers offered by deal.II.
     * The output is also parallelized: every processor writes its own output,
     * ParaView is able to group them together.
     * The mesh refinement is parallelized too.
     *
     * The solution vectors, for example displacement, are declared as
     * non-ghosted vectors. There are two reasons for this: 1. we do not need
     * the dofs not owned by the current processor in assembly since the system
     * matrix is not dependent on the dofs at all;
     * 2. we need to do matrix/vector operations on these vectors, for example
     * add and vmult, this requires the vectors to be non-ghosted.
     *
     * Algorithm-wise, this class is not different from the serial version,
     * Newmark-beta method is used for time-discretization and
     * displacement-based finite element is used for space-discretization.
     */
    template <int dim>
    class SharedLinearElasticity : public SharedSolidSolver<dim>
    {
      MPISharedSolidSolverInheritanceMacro();

    public:
      /*! \brief Constructor.
       *
       * The triangulation can either be generated using dealii functions or
       * from Abaqus input file. It is fully distributed.
       * Also we use a parameter handler to specify all the input parameters.
       */
      SharedLinearElasticity(Triangulation<dim> &,
                             const Parameters::AllParameters &);
      /*! \brief Destructor. */
      ~SharedLinearElasticity() {}

    private:
      /**
       * Assembles lhs and rhs. At time step 0, the lhs is the mass matrix;
       * at all the following steps, it is \f$ M + \beta{\Delta{t}}^2K \f$.
       */
      void assemble_system(bool is_initial);

      /**
       * Update the strain and stress, used in output_results and FSI.
       */
      virtual void update_strain_and_stress() override;

      void run_one_step(bool first_step);

      std::vector<LinearElasticMaterial<dim>> material;
    };
  } // namespace MPI
} // namespace Solid

#endif

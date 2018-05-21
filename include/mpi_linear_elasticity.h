#ifndef MPI_LINEAR_ELASTICITY
#define MPI_LINEAR_ELASTICITY

#include "linear_elastic_material.h"
#include "mpi_solid_solver.h"

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
    class LinearElasticity : public SolidSolver<dim>
    {
    public:
      /*! \brief Constructor.
       *
       * The triangulation can either be generated using dealii functions or
       * from Abaqus input file. It is fully distributed.
       * Also we use a parameter handler to specify all the input parameters.
       */
      LinearElasticity(parallel::distributed::Triangulation<dim> &,
                       const Parameters::AllParameters &);
      /*! \brief Destructor. */
      ~LinearElasticity() {}

    private:
      using SolidSolver<dim>::triangulation;
      using SolidSolver<dim>::parameters;
      using SolidSolver<dim>::dof_handler;
      using SolidSolver<dim>::dg_dof_handler;
      using SolidSolver<dim>::fe;
      using SolidSolver<dim>::dg_fe;
      using SolidSolver<dim>::volume_quad_formula;
      using SolidSolver<dim>::face_quad_formula;
      using SolidSolver<dim>::constraints;
      using SolidSolver<dim>::system_matrix;
      using SolidSolver<dim>::stiffness_matrix;
      using SolidSolver<dim>::system_rhs;
      using SolidSolver<dim>::current_acceleration;
      using SolidSolver<dim>::current_velocity;
      using SolidSolver<dim>::current_displacement;
      using SolidSolver<dim>::previous_acceleration;
      using SolidSolver<dim>::previous_velocity;
      using SolidSolver<dim>::previous_displacement;
      using SolidSolver<dim>::mpi_communicator;
      using SolidSolver<dim>::pcout;
      using SolidSolver<dim>::time;
      using SolidSolver<dim>::timer;
      using SolidSolver<dim>::locally_owned_dofs;
      using SolidSolver<dim>::locally_relevant_dofs;

      /**
       * Assembles lhs and rhs. At time step 0, the lhs is the mass matrix;
       * at all the following steps, it is \f$ M + \beta{\Delta{t}}^2K \f$.
       */
      void assemble_system(bool is_initial);

      void run_one_step(bool first_step);

      LinearElasticMaterial<dim> material;
    };
  } // namespace MPI
} // namespace Solid

#endif

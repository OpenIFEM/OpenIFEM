#ifndef PARALLEL_NAVIER_STOKES
#define PARALLEL_NAVIER_STOKES

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_parallel_block_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "parameters.h"
#include "utilities.h"

namespace Fluid
{
  using namespace dealii;

  /** \brief The incompressible Navier Stokes equation solver.
   *
   * This program is built upon dealii tutorials step-57, step-22, step-20.
   * We use fully implicit scheme for time stepping.
   * At each time step, Newton's method is used to solve for the update,
   * so we define two variables: the present solution and the update.
   * Additionally, the evaluation point is for temporarily holding
   * the updated solution.
   * We use one ConstraintMatrix for Dirichlet boundary conditions
   * at the initial step and a zero ConstraintMatrix for the rest steps.
   * Although the density does not matter in the incompressible flow, we still
   * include it in the formulation in order to be consistent with the
   * slightly compressible flow. Correspondingly the viscosity represents
   * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
   * and the pressure block in the solution is the real pressure.
   */
  template <int dim>
  class ParallelNavierStokes
  {
  public:
    /** \brief Constructor.
     *
     * We do not want to modify the solver every time we change the
     * triangulation.
     * So we pass in a Triangulation<dim> that is either generated using dealii
     * functions or by reading Abaqus input file. Also, a parameter handler is
     * required to specify all the input parameters.
     */
    ParallelNavierStokes(parallel::distributed::Triangulation<dim> &,
                         const Parameters::AllParameters &);
    /*! \brief Destructor. */
    ~ParallelNavierStokes() { dof_handler.clear(); };
    /**
     * This function implements the Newton iteration with given tolerance
     * and maximum number of iterations.
     */
    void run();

  private:
    /**
     * Currently the Dirichlet BCs in the input file can only be constant
     * values.
     * Space/time-dependent Dirichlet BCs are hard-coded in this class.
     */
    class BoundaryValues;
    /**
     * The blcok preconditioner for the whole linear system.
     * It is a private member of NavierStokes<dim>.
     */
    class BlockSchurPreconditioner;
    /**
     * This function initializes the DoFHandler and constraints.
     */
    void setup_dofs();
    /**
     * Specify the sparsity pattern and reinit matrices and vectors.
     * It is separated from setup_dofs because when we do mesh refinement
     * we need to transfer the solution from old grid to the new one.
     */
    void initialize_system();
    /**
     * This function builds the system matrix and right hand side that we
     * currently work on. The initial_step argument is used to determine
     * which set of constraints we apply (nonzero for the initial step and zero
     * for the others). The assemble_matrix flag determines whether to
     * assemble the whole system or only the right hand side vector,
     * respectively.
     */
    void assemble(const bool, const bool);
    void assemble_system(const bool);
    void assemble_rhs(const bool);
    /**
     * In this function, we use GMRES solver with the block preconditioner,
     * which is defined at the beginning of the program, to solve the linear
     * system. What we obtain at this step is the solution update.
     * For the initial step, nonzero constraints are applied in order to
     * make sure boundary conditions are satisfied.
     * In the following steps, we will solve for the Newton update so zero
     * constraints are used.
     */
    std::pair<unsigned int, double> solve(const bool);
    /**
     * After finding a good initial guess on the coarse mesh, we hope to
     * decrease the error through refining the mesh. Here we do adaptive
     * refinement based on the Kelly estimator on the velocity only.
     * We also need to transfer the current solution to the
     * next mesh using the SolutionTransfer class.
     */
    void refine_mesh(const unsigned int, const unsigned int);
    /**
     * Write a vtu file for the current solution, as well as a pvtu file to
     * organize them.
     */
    void output_results(const unsigned int) const;

    double viscosity; //!< Dynamic viscosity
    double rho;
    double gamma;
    const unsigned int degree;
    std::vector<types::global_dof_index> dofs_per_block;

    parallel::distributed::Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    ConstraintMatrix zero_constraints;
    ConstraintMatrix nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    PETScWrappers::MPI::BlockSparseMatrix system_matrix;
    PETScWrappers::MPI::BlockSparseMatrix mass_matrix;
    PETScWrappers::MPI::BlockSparseMatrix mass_schur;

    /// The latest known solution.
    PETScWrappers::MPI::BlockVector present_solution;
    /// The increment at a certain Newton iteration.
    PETScWrappers::MPI::BlockVector newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    PETScWrappers::MPI::BlockVector evaluation_point;
    PETScWrappers::MPI::BlockVector system_rhs;

    const double tolerance;
    const unsigned int max_iteration;

    Parameters::AllParameters parameters;

    MPI_Comm mpi_communicator;

    ConditionalOStream pcout;

    /// The IndexSets of owned velocity and pressure respectively.
    std::vector<IndexSet> owned_partitioning;

    /// The IndexSets of relevant velocity and pressure respectively.
    std::vector<IndexSet> relevant_partitioning;

    /// The IndexSet of all relevant dofs. This seems to be redundant but handy.
    IndexSet locally_relevant_dofs;

    /// The BlockSchurPreconditioner for the whole system:
    std::shared_ptr<BlockSchurPreconditioner> preconditioner;

    Utils::Time time;
    mutable TimerOutput timer;

    /** \brief Helper function to specify Dirchlet boundary conditions.
     *
     *  It specifies a parabolic velocity profile at the left side boundary,
     *  and all the remaining boundaries are considered as walls
     *  except for the right side one.
     */
    class BoundaryValues : public Function<dim>
    {
    public:
      BoundaryValues() : Function<dim>(dim + 1) {}
      virtual double value(const Point<dim> &p,
                           const unsigned int component) const;

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const;
    };

    /** \brief Block preconditioner for the system
     *
     * A right block preconditioner is defined here:
     * \f{eqnarray*}{
     *      P^{-1} = \begin{pmatrix} \tilde{A}^{-1} & 0\\ 0 & I\end{pmatrix}
     *               \begin{pmatrix} I & -B^T\\ 0 & I\end{pmatrix}
     *               \begin{pmatrix} I & 0\\ 0 & \tilde{S}^{-1}\end{pmatrix}
     * \f}
     *
     * \f$\tilde{A}\f$ is unsymmetric thanks to the convection term.
     * We do not have a nice way to deal with other than using a direct
     * solver. (Geometric multigrid is the right way to go if only you are
     * brave enough to implement it.)
     *
     * \f$\tilde{S}^{-1}\f$ is the inverse of the total Schur complement,
     * which consists of a reaction term, a diffusion term, a Grad-Div term
     * and a convection term.
     * In practice, the convection contribution is ignored because it is not
     * clear how to treat it. But the block preconditioner is good enough even
     * without it. Namely,
     *
     * \f[
     *   \tilde{S}^{-1} = -(\nu + \gamma)M_p^{-1} -
     * \frac{1}{\Delta{t}}{[B(diag(M_u))^{-1}B^T]}^{-1}
     * \f]
     * where \f$M_p\f$ is the pressure mass, and \f${[B(diag(M_u))^{-1}B^T]}\f$
     * is
     * an
     * approximation to the Schur complement of (velocity) mass matrix
     * \f$BM_u^{-1}B^T\f$.
     *
     * In summary, in order to form the BlockSchurPreconditioner for our system,
     * we need to compute \f$M_u^{-1}\f$, \f$M_p^{-1}\f$, \f$\tilde{A}^{-1}\f$
     * and them operate on them.
     * The first two matrices can be easily defined indirectly with CG solver,
     * and the last one is going to be solved with direct solver.
     */
    class BlockSchurPreconditioner : public Subscriptor
    {
    public:
      /// Constructor.
      BlockSchurPreconditioner(
        TimerOutput &timer,
        double gamma,
        double viscosity,
        double rho,
        double dt,
        const std::vector<IndexSet> &owned_partitioning,
        const PETScWrappers::MPI::BlockSparseMatrix &system,
        const PETScWrappers::MPI::BlockSparseMatrix &mass,
        PETScWrappers::MPI::BlockSparseMatrix &schur);

      /// The matrix-vector multiplication must be defined.
      void vmult(PETScWrappers::MPI::BlockVector &dst,
                 const PETScWrappers::MPI::BlockVector &src) const;

    private:
      /// We would like to time the BlockSchuPreconditioner in detail.
      TimerOutput &timer;
      const double gamma;
      const double viscosity;
      const double rho;
      const double dt;

      /// dealii smart pointer checks if an object is still being referenced
      /// when it is destructed therefore is safer than plain reference.
      const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
        system_matrix;
      const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
        mass_matrix;
      /**
       * As discussed, \f${[B(diag(M_u))^{-1}B^T]}\f$ and its inverse
       * need to be computed.
       * We can either explicitly compute it out as a matrix, or define it as a
       * class
       * with a vmult operation. The second approach saves some computation to
       * construct the matrix, but leads to slow convergence in CG solver
       * because
       * of the absence of preconditioner.
       * Based on my tests, the first approach is more than 10 times faster so I
       * go with this route.
       */
      const SmartPointer<PETScWrappers::MPI::BlockSparseMatrix> mass_schur;

      /// A dummy solver control object for MUMPS solver
      SolverControl dummy_sc;
      /**
       * Similar to the serial code, reuse the factorization.
       * Although SparseDirectMUMPS does not have the initialize and vmult pair
       * like UMFPACK, it reuses the factorization as long as it is not reset
       * and the matrix does not change.
       */
      mutable PETScWrappers::SparseDirectMUMPS A_inverse;
    };
  };
}

#endif

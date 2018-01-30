#ifndef NAVIER_STOKES
#define NAVIER_STOKES

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
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

// To transfer solutions between meshes, this file is included:
#include <deal.II/numerics/solution_transfer.h>

// This file includes UMFPACK: the direct solver:
#include <deal.II/lac/sparse_direct.h>

// And the one for ILU preconditioner:
#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "parameters.h"
#include "utilities.h"

template <int>
class FSI;

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
  class NavierStokes
  {
  public:
    friend FSI<dim>;

    /** \brief Constructor.
     *
     * We do not want to modify the solver every time we change the
     * triangulation.
     * So we pass in a Triangulation<dim> that is either generated using dealii
     * functions or by reading Abaqus input file. Also, a parameter handler is
     * required to specify all the input parameters.
     */
    NavierStokes(Triangulation<dim> &, const Parameters::AllParameters &);
    /**
     * This function implements the Newton iteration with given tolerance
     * and maximum number of iterations.
     */
    void run();
    /**
     * In FSI application, the workflow is controled by the FSI class,
     * as an interface, this function runs the simulation for one time step.
     */
    void run_one_step(bool);

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
    struct CellProperty;

    /**
     * This function initializes the DoFHandler and constraints.
     */
    void setup_dofs();
    /**
     * This function sets up the material property stored at each cell.
     */
    void setup_cell_property();
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
    void assemble(const bool initial_step, const bool assemble_matrix);
    void assemble_system(const bool initial_step);
    void assemble_rhs(const bool initial_step);
    /**
     * In this function, we use FGMRES together with the block preconditioner,
     * which is defined at the beginning of the program, to solve the linear
     * system. What we obtain at this step is the solution vector. If this is
     * the initial step, the solution vector gives us an initial guess for the
     * Navier Stokes equations. For the initial step, nonzero constraints are
     * applied in order to make sure boundary conditions are satisfied. In the
     * following steps, we will solve for the Newton update so zero
     * constraints are used.
     */
    std::pair<unsigned int, double> solve(const bool initial_step);
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

    Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    ConstraintMatrix zero_constraints;
    ConstraintMatrix nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> mass_matrix;
    SparsityPattern mass_schur_pattern;
    SparseMatrix<double> mass_schur;

    /// The latest known solution.
    BlockVector<double> present_solution;
    /// The increment at a certain Newton iteration.
    BlockVector<double> newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    BlockVector<double> evaluation_point;
    /**
     * The solution increment from the previous time step to the current time
     * step,
     * i.e., the summation of the newton_update at a certain time step.
     * This is redundant in fluid stand-alone simulation, but useful in FSI
     * application because we are interested in fluid acceleration.
     */
    BlockVector<double> solution_increment;
    BlockVector<double> system_rhs;

    const double tolerance;
    const unsigned int max_iteration;

    Utils::Time time;
    mutable TimerOutput timer;

    Parameters::AllParameters parameters;

    /// The BlockSchurPreconditioner for the whole system.
    std::shared_ptr<BlockSchurPreconditioner> preconditioner;

    CellDataStorage<typename Triangulation<dim>::cell_iterator, CellProperty>
      cell_property;

    std::vector<SymmetricTensor<2, dim>> fsi_stress;
    std::vector<Tensor<1, dim>> fsi_acceleration;

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
     * is an
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
      BlockSchurPreconditioner(TimerOutput &timer,
                               double gamma,
                               double viscosity,
                               double rho,
                               double dt,
                               const BlockSparseMatrix<double> &system,
                               const BlockSparseMatrix<double> &mass,
                               SparseMatrix<double> &schur);

      /// The matrix-vector multiplication must be defined.
      void vmult(BlockVector<double> &dst,
                 const BlockVector<double> &src) const;

    private:
      TimerOutput &timer;
      const double gamma;
      const double viscosity;
      const double rho;
      const double dt;

      /// dealii smart pointer checks if an object is still being referenced
      /// when it is destructed therefore is safer than plain reference.
      const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
      const SmartPointer<const BlockSparseMatrix<double>> mass_matrix;
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
      const SmartPointer<SparseMatrix<double>> mass_schur;

      /// The direct solver used for \f$\tilde{A}\f$. We declare it as a member
      /// so that it can be initialized only once for many applications of the
      /// preconditioner.
      SparseDirectUMFPACK A_inverse;
    };

    /**
     * This struct tells whether a cell contains real fluid or artificial fluid,
     * and returns the corresponding properties.
     */
    struct CellProperty
    {
      /**
       * Material indicator: 1 for solid and 0 for fluid.
       */
      int indicator;
      /**
       * The viscosity and density of fluid and solid.
       */
      double fluid_mu;
      double fluid_rho;
      double solid_mu;
      double solid_rho;
      /**
       * Return the density of the current cell.
       */
      double get_rho() const;
      /**
       * Return the viscosity of the current cell.
       */
      double get_mu() const;
    };
  };
}

#endif

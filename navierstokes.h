#ifndef NAVIER_STOKES
#define NAVIER_STOKES

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

namespace
{
  using namespace dealii;

  /** \brief Helper function to specify Dirchlet boundary conditions.
   *
   *  It specifies a parabolic velocity profile at the left side boundary,
   *  and all the remaining boundaries are considered as walls
   *  except for the right side one.
   */
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() : Function<dim>(dim + 1) {}
    virtual double value(const Point<dim> &p,
                         const unsigned int component) const;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const;
  };

  /** \brief Inverse of a symmetric matrix.
   *
   * The inverse of symmetric matrices are required for both
   * \f$\tilde{S_M}^{-1}\f$
   * and \f$M_p^{-1}\f$, so we define a class to compute it.
   * However, instead of computing it explicitly, we only define its
   * matrix-vector
   * multiplication operation through solving a linear system.
   * CG solver is used because symmetric matrices are expected.
   */
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType &m,
                  const PreconditionerType &preconditioner);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const MatrixType> matrix;
    const SmartPointer<const PreconditionerType> preconditioner;
  };

  /** \brief Approximate Schur complement of mass matrix
   *
   * The Schur complement of mass matrix is written as \f$S_M = BM^{-1}B^T\f$,
   * we use \f$B(diag(M))^{-1}B^T\f$ to approximate it.
   */
  class ApproximateMassSchur : public Subscriptor
  {
  public:
    ApproximateMassSchur(const BlockSparseMatrix<double> &M);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> mass_matrix;
    mutable Vector<double> tmp1, tmp2;
  };

  /** \brief The inverse matrix of the system Schur complement.
   *
   * The inverse of the total Schur complement can be decomposed as the sum of
   * the inverse of the diffusion plus Grad-Div Schur complement,
   * and the inverse of the mass Schur complement.
   * The inverse of the convection Schur complement is ignored because it is
   * unknown how to treat it. But the block preconditioner is good enough even
   * without it.
   * For our time-dependent problem,
   * \f[
   *   \tilde{S}^{-1} = -(\nu + \gamma)M_p^{-1} -
   * \frac{1}{\Delta{t}}{[B(diag(M))^{-1}B^T]}^{-1}
   * \f]
   * where $M_p$ is the pressure mass, and ${[B(diag(M))^{-1}B^T]}$ is
   * ApproximateMassSchur.
   */
  template <class PreconditionerSm, class PreconditionerMp>
  class SchurComplementInverse : public Subscriptor
  {
  public:
    SchurComplementInverse(
      double gamma,
      double viscosity,
      double dt,
      const InverseMatrix<ApproximateMassSchur, PreconditionerSm> &Sm_inv,
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp> &Mp_inv);
    void vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const double gamma;
    const double viscosity;
    const double dt;
    const SmartPointer<
      const InverseMatrix<ApproximateMassSchur, PreconditionerSm>>
      Sm_inverse;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerMp>>
      Mp_inverse;
  };

  /** \brief Block preconditioner for the system
   *
   * A right block preconditioner is defined here:
   * \f{eqnarray*}{
   *      P^{-1} = \begin{pmatrix} \tilde{A}^{-1} & 0\\ 0 & I\end{pmatrix}
   *               \begin{pmatrix} I & -B^T\\ 0 & I\end{pmatrix}
   *               \begin{pmatrix} I & 0\\ 0 & \tilde{S}^{-1}\end{pmatrix}
   * \f}
   * \f$\tilde{S}^{-1}\f$ is defined by SchurComplementInverse,
   * \f$\tilde{A}^{-1}\f$ is unsymmetric thanks to the convection term,
   * which we do not have a nice way to deal with other than using a direct
   * solver.
   * (Geometric multigrid is the right way to go if only you are brave enough
   * to implement it.)
   * The template arguments are the same as SchurComplementInverse.
   */
  template <class PreconditionerSm, class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      const BlockSparseMatrix<double> &system,
      const SchurComplementInverse<PreconditionerSm, PreconditionerMp> &S_inv);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const SchurComplementInverse<PreconditionerSm, PreconditionerMp>>
      S_inverse;
    SparseDirectUMFPACK A_inverse;
  };
}

namespace Fluid
{
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
   */
  template <int dim>
  class NavierStokes
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
    NavierStokes(Triangulation<dim> &, const Parameters::AllParameters &);
    /**
     * This function implements the Newton iteration with given tolerance
     * and maximum number of iterations.
     */
    void run();

  private:
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
     * NOTE: this function has not been tested.
     */
    void refine_mesh();
    /**
     * Write a vtu file for the current solution, as well as a pvtu file to
     * organize them.
     */
    void output_results(const unsigned int) const;

    double viscosity;
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

    /// The latest known solution.
    BlockVector<double> present_solution;
    /// The increment at a certain Newton iteration.
    BlockVector<double> newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    BlockVector<double> evaluation_point;
    BlockVector<double> system_rhs;

    const double tolerance;
    const unsigned int max_iteration;

    Utils::Time time;
    mutable TimerOutput timer;

    Parameters::AllParameters parameters;

    /**
     * The first building block of \f$\tilde{S}^{-1}\f$:
     * inverse of mass Schur complement.
     */
    std::shared_ptr<ApproximateMassSchur> approximate_Sm;
    std::shared_ptr<PreconditionIdentity> preconditioner_Sm;
    std::shared_ptr<InverseMatrix<ApproximateMassSchur, PreconditionIdentity>>
      Sm_inverse;

    /**
     * The second building block of the \f$\tilde{S}^{-1}\f$:
     * inverse of the pressure mass matrix, which is used for diffusion and
     * Grad-Div.
     */
    std::shared_ptr<SparseILU<double>> preconditioner_Mp;
    std::shared_ptr<InverseMatrix<SparseMatrix<double>, SparseILU<double>>>
      Mp_inverse;

    /// The SchurComplementInverse \f$\tilde{S}^{-1}\f$.
    std::shared_ptr<
      SchurComplementInverse<PreconditionIdentity, SparseILU<double>>>
      S_inverse;

    /// The BlockSchurPreconditioner for the whole system:
    std::shared_ptr<
      BlockSchurPreconditioner<PreconditionIdentity, SparseILU<double>>>
      preconditioner;
  };
}

#endif

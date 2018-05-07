#ifndef INS_IMEX
#define INS_IMEX

#include "fluidSolver.h"

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  extern template class FluidSolver<2>;
  extern template class FluidSolver<3>;

  /** \brief Incompressible Navier Stokes equation solver
   *        using implicit-explicit time scheme.
   *
   * This program is built upon dealii tutorials step-57, step-22, step-20.
   * Although the density does not matter in the incompressible flow, we
   * still include it in the formulation in order to be consistent with the
   * slightly compressible flow. Correspondingly the viscosity represents
   * dynamic visocity \f$\mu\f$ instead of kinetic visocity \f$\nu\f$,
   * and the pressure block in the solution is the non-normalized pressure.
   *
   * The system equation is written in the incremental form, and we treat
   * the convection term explicitly. Therefore the system equation is linear
   * and symmetric, which does not need to be solved with Newton's iteration.
   * The system is further stablized and preconditioned with Grad-Div method,
   * where GMRES solver is used as the outer solver.
   */
  template <int dim>
  class InsIMEX : public FluidSolver<dim>
  {
  public:
    //! FSI solver need access to the private members of this solver.
    friend FSI<dim>;

    //! Constructor
    InsIMEX(Triangulation<dim> &, const Parameters::AllParameters &);

    //! Run the simulation
    void run() override;

    //! Destructor
    ~InsIMEX() {}

  private:
    class BlockSchurPreconditioner;

    using FluidSolver<dim>::setup_dofs;
    using FluidSolver<dim>::make_constraints;
    using FluidSolver<dim>::setup_cell_property;
    using FluidSolver<dim>::initialize_system;
    using FluidSolver<dim>::refine_mesh;
    using FluidSolver<dim>::output_results;

    using FluidSolver<dim>::dofs_per_block;
    using FluidSolver<dim>::triangulation;
    using FluidSolver<dim>::fe;
    using FluidSolver<dim>::dof_handler;
    using FluidSolver<dim>::volume_quad_formula;
    using FluidSolver<dim>::face_quad_formula;
    using FluidSolver<dim>::zero_constraints;
    using FluidSolver<dim>::nonzero_constraints;
    using FluidSolver<dim>::sparsity_pattern;
    using FluidSolver<dim>::system_matrix;
    using FluidSolver<dim>::mass_matrix;
    using FluidSolver<dim>::mass_schur_pattern;
    using FluidSolver<dim>::mass_schur;
    using FluidSolver<dim>::present_solution;
    using FluidSolver<dim>::system_rhs;
    using FluidSolver<dim>::time;
    using FluidSolver<dim>::timer;
    using FluidSolver<dim>::parameters;
    using FluidSolver<dim>::cell_property;

    /// Specify the sparsity pattern and reinit matrices and vectors based on
    /// the dofs and constraints.
    void initialize_system() override;

    /*! \brief Assemble the system matrix, mass mass matrix, and the RHS.
     *
     *  It can be used to assemble the entire system or only the RHS.
     *  An additional option is added to determine whether nonzero
     *  constraints or zero constraints should be used.
     */
    void assemble(bool use_nonzero_constraints, bool assemble_system);

    /*! \brief Solve the linear system using FGMRES solver plus block
     *         preconditioner.
     *
     *  After solving the linear system, the same ConstraintMatrix as used
     *  in assembly must be used again, to set the constrained value.
     *  The second argument is used to determine
     *  whether the block preconditioner should be reset or not.
     */
    std::pair<unsigned int, double> solve(bool use_nonzero_constraints,
                                          bool assemble_system);

    /// Run the simulation for one time step.
    void run_one_step(bool dummy = true) override;

    /// The increment at a certain time step.
    BlockVector<double> solution_increment;

    /// The BlockSchurPreconditioner for the entire system.
    std::shared_ptr<BlockSchurPreconditioner> preconditioner;

    /** \brief Block preconditioner for the system
     *
     * A right block preconditioner is defined here:
     * \f{eqnarray*}{
     *      P^{-1} = \begin{pmatrix} \tilde{A}^{-1} & 0\\ 0 & I\end{pmatrix}
     *               \begin{pmatrix} I & -B^T\\ 0 & I\end{pmatrix}
     *               \begin{pmatrix} I & 0\\ 0 & \tilde{S}^{-1}\end{pmatrix}
     * \f}
     *
     * \f$\tilde{A}\f$ is symmetric since the convection term is eliminated
     * from the LHS.
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
     *   \frac{1}{\Delta{t}}{[B(diag(M_u))^{-1}B^T]}^{-1}
     * \f]
     * where \f$M_p\f$ is the pressure mass, and
     * \f${[B(diag(M_u))^{-1}B^T]}\f$
     * is an approximation to the Schur complement of (velocity) mass matrix
     * \f$BM_u^{-1}B^T\f$.
     *
     * In summary, in order to form the BlockSchurPreconditioner for our
     * system, we need to compute \f$M_u^{-1}\f$, \f$M_p^{-1}\f$,
     * \f$\tilde{A}^{-1}\f$, and then operate on them.
     * These matrices are all symmetric in IMEX scheme.
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
    };
  };
} // namespace Fluid

#endif

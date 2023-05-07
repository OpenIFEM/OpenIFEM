#ifndef MPI_INSIM
#define MPI_INSIM

#include "mpi_fluid_solver.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    /** \brief The parallel incompressible Navier Stokes equation solver.
     *
     * This program is built upon dealii tutorials step-57, step-22, step-20.
     * Although the density does not matter in the incompressible flow, we still
     * include it in the formulation in order to be consistent with the
     * slightly compressible flow. Correspondingly the viscosity represents
     * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
     * and the pressure block in the solution is the non-normalized pressure.
     *
     * Fully implicit scheme is used for time stepping. Newton's method is
     * applied to solve the nonlinear system, thus the actual dofs being solved
     * is the velocity and pressure increment.
     *
     * The final linear system to be solved is nonsymmetric. GMRES solver with
     * Grad-Div right preconditioner is applied, which does modify the linear
     * system
     * a little bit, and requires the velocity shape functions to be one order
     * higher
     * than that of the pressure.
     */
    template <int dim>
    class InsIM : public FluidSolver<dim>
    {
      MPIFluidSolverInheritanceMacro();

    public:
      //! Constructor.
      InsIM(parallel::distributed::Triangulation<dim> &,
            const Parameters::AllParameters &);
      ~InsIM(){};
      //! Run the simulation.
      void run();

    private:
      class BlockSchurPreconditioner;

      /// Specify the sparsity pattern and reinit matrices and vectors based on
      /// the dofs and constraints.
      void initialize_system() override;

      /*! \brief Assemble the system matrix, mass mass matrix, and the RHS.
       *
       *  Since backward Euler method is used, the linear system must be
       * reassembled
       *  at every Newton iteration. The Dirichlet BCs are applied at the same
       * time
       *  as the cell matrix and rhs are distributed to the global matrix and
       * rhs, which is optimal according to the deal.II documentation. The
       * boolean argument is used to determine whether nonzero constraints or
       * zero constraints should be used.
       */
      void assemble(const bool use_nonzero_constraints);

      /*! \brief Solve the linear system using FGMRES solver plus block
       * preconditioner.
       *
       *  After solving the linear system, the same AffineConstraints<double> as
       * used in assembly must be used again, to set the solution to the right
       * value at the constrained dofs.
       */
      std::pair<unsigned int, double> solve(const bool use_nonzero_constraints);

      /*! \brief Run the simulation for one time step.
       *
       *  If the Dirichlet BC is time-dependent, nonzero constraints must be
       * applied
       *  at every first Newton iteration in every time step. If it is not, only
       *  apply nonzero constraints at the first iteration in the first time
       * step. A boolean argument controls whether nonzero constraints should be
       *  applied in a certain time step.
       */
      void run_one_step(bool apply_nonzero_constraints,
                        bool assemble_system = true) override;

      /// The increment at a certain Newton iteration.
      PETScWrappers::MPI::BlockVector newton_update;

      /**
       * The latest know solution plus the cumulation of all newton_updates
       * in the current time step, which approaches to the new present_solution.
       */
      PETScWrappers::MPI::BlockVector evaluation_point;

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
       * where \f$M_p\f$ is the pressure mass, and
       * \f${[B(diag(M_u))^{-1}B^T]}\f$ is an approximation to the Schur
       * complement of (velocity) mass matrix \f$BM_u^{-1}B^T\f$.
       *
       * In summary, in order to form the BlockSchurPreconditioner for our
       * system, we need to compute \f$M_u^{-1}\f$, \f$M_p^{-1}\f$,
       * \f$\tilde{A}^{-1}\f$ and them operate on them. The first two matrices
       * can be easily defined indirectly with CG solver, and the last one is
       * going to be solved with direct solver.
       */
      class BlockSchurPreconditioner : public Subscriptor
      {
      public:
        /// Constructor.
        BlockSchurPreconditioner(
          TimerOutput &timer2,
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
        TimerOutput &timer2;
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
         * We can either explicitly compute it out as a matrix, or define it as
         * a class with a vmult operation. The second approach saves some
         * computation to construct the matrix, but leads to slow convergence in
         * CG solver because of the absence of preconditioner. Based on my
         * tests, the first approach is more than 10 times faster so I go with
         * this route.
         */
        const SmartPointer<PETScWrappers::MPI::BlockSparseMatrix> mass_schur;

        /// A dummy solver control object for MUMPS solver
        SolverControl dummy_sc;
        /**
         * Similar to the serial code, reuse the factorization.
         * Although SparseDirectMUMPS does not have the initialize and vmult
         * pair like UMFPACK, it reuses the factorization as long as it is not
         * reset and the matrix does not change.
         */
        mutable PETScWrappers::SparseDirectMUMPS A_inverse;
      };
    };
  } // namespace MPI
} // namespace Fluid

#endif

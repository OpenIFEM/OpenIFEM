#ifndef SABLE
#define SABLE

#include "fluid_solver.h"
#include <mpi.h>

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  extern template class FluidSolver<2>;
  extern template class FluidSolver<3>;

  /*! \brief Incompressible Navier Stokes equation solver
   *        using implicit time scheme.
   *
   * This program is built upon dealii tutorials step-57, step-22, step-20.
   * Although the density does not matter in the incompressible flow, we still
   * include it in the formulation in order to be consistent with the
   * slightly compressible flow. Correspondingly the viscosity represents
   * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
   * and the pressure block in the solution is the non-normalized pressure.
   *
   * Fully implicit scheme is used for time stepping. Newton's method is applied
   * to solve the nonlinear system, thus the actual dofs being solved is the
   * velocity and pressure increment.
   *
   * The final linear system to be solved is nonsymmetric. GMRES solver with
   * Grad-Div right preconditioner is applied, which does modify the linear
   * system
   * a little bit, and requires the velocity shape functions to be one order
   * higher
   * than that of the pressure.
   */
  template <int dim>
  class SableWrap : public FluidSolver<dim>
  {
  public:
    friend FSI<dim>;

    SableWrap(Triangulation<dim> &,
          const Parameters::AllParameters &,
          std::vector<int> &,
          std::shared_ptr<Function<dim>> bc =
            std::make_shared<Functions::ZeroFunction<dim>>(
              Functions::ZeroFunction<dim>(dim + 1)));
    ~SableWrap(){};
    void run() override;

  private:
    class BlockSchurPreconditioner;

    using FluidSolver<dim>::setup_dofs;
    using FluidSolver<dim>::make_constraints;
    using FluidSolver<dim>::setup_cell_property;
    using FluidSolver<dim>::initialize_system;
    using FluidSolver<dim>::refine_mesh;
    using FluidSolver<dim>::output_results;
    using FluidSolver<dim>::update_stress;

    using FluidSolver<dim>::dofs_per_block;
    using FluidSolver<dim>::triangulation;
    using FluidSolver<dim>::fe;
    using FluidSolver<dim>::scalar_fe;
    using FluidSolver<dim>::dof_handler;
    using FluidSolver<dim>::scalar_dof_handler;
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
    using FluidSolver<dim>::solution_increment;
    using FluidSolver<dim>::system_rhs;
    using FluidSolver<dim>::time;
    using FluidSolver<dim>::timer;
    using FluidSolver<dim>::parameters;
    using FluidSolver<dim>::cell_property;
    using FluidSolver<dim>::boundary_values;
    using FluidSolver<dim>::stress;

    /// Specify the sparsity pattern and reinit matrices and vectors based on
    /// the dofs and constraints.
    void initialize_system() override;

    /*! \brief Assemble the system matrix, mass mass matrix, and the RHS.
     *
     *  Since backward Euler method is used, the linear system must be
     * reassembled
     *  at every Newton iteration. The Dirichlet BCs are applied at the same
     * time
     *  as the cell matrix and rhs are distributed to the global matrix and rhs,
     *  which is optimal according to the deal.II documentation.
     *  The boolean argument is used to determine whether nonzero constraints
     *  or zero constraints should be used.
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
     *  apply nonzero constraints at the first iteration in the first time step.
     *  A boolean argument controls whether nonzero constraints should be
     *  applied in a certain time step.
     */
    void run_one_step(bool apply_nonzero_constraints,
                      bool assemble_system = true) override;

    /// The increment at a certain Newton iteration.
    BlockVector<double> newton_update;
    /**
     * The latest know solution plus the cumulation of all newton_updates
     * in the current time step, which approaches to the new present_solution.
     */
    BlockVector<double> evaluation_point;

    /// The BlockSchurPreconditioner for the entire system.
    std::shared_ptr<BlockSchurPreconditioner> preconditioner;

    // Vector which stores Sable processor ids
    std::vector<int> sable_ids;

    // Recieve solution from Sable
    void rec_data(double ** rec_buffer, const std::vector <int> & cmapp, const std::vector <int> & cmapp_sizes,
	int data_size);
    
    void rec_velocity(const int& sable_n_nodes, const int& sable_n_nodes_one_dir);

    void rec_stress(const int& sable_n_elements);

    bool All(bool my_b); 

    void Max(int &send_buffer);

    /*! \brief Block preconditioner for the system
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
  };
} // namespace Fluid

#endif

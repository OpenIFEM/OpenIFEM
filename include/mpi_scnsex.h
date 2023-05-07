#ifndef MPI_SCNSEX
#define MPI_SCNSEX

#include "mpi_fluid_solver.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    /*! \brief the parallel Slightly Compresisble Navier Stokes equation solver
     *
     * Although the density does not matter in the incompressible flow, we still
     * include it in the formulation in order to be consistent with the
     * slightly compressible flow. Correspondingly the viscosity represents
     * the dynamic visocity \f$\mu\f$ instead of the kinetic visocity \f$\nu\f$,
     * and the pressure block in the solution is the non-normalized pressure.
     *
     * Explicit time scheme is used for time stepping. The velocity and pressure
     * dofs are decoupled and solved separately. A inner iteration is
     * incoporated to make sure the solution converges. However, the explicit
     * solver has a strict restriction on time step size. For pure acoustic wave
     * propagation, the time step must be smaller than 1e-7s for convergence.
     *
     * CG solver is used for both pressure and velocity matrices as they only
     * consist of the mass and viscosity components.
     */
    template <int dim>
    class SCnsEX : public FluidSolver<dim>
    {
      MPIFluidSolverInheritanceMacro();

    public:
      //! Constructor.
      SCnsEX(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &);
      ~SCnsEX(){};
      //! Run the simulation.
      void run();

      /*! \brief Set up the time limit for specified hard coded boundary
       * condition. This function is problematic for now, do not use it just
       * yet.
       */
      void set_hard_coded_boundary_condition_time(const unsigned int,
                                                  const double);

    private:
      /// Specify the sparsity pattern and reinit matrices and vectors based on
      /// the dofs and constraints.
      virtual void initialize_system() override;

      /*! \brief Assemble the system matrix, mass mass matrix, and the RHS.
       *
       * The Dirichlet BCs are applied at the sametime as the cell matrix and
       * rhs are distributed to the global matrix and rhs, which is optimal
       * according to the deal.II documentation. The 2 boolean arguments are
       * used to determine whether assemble velocity or pressure, and whether to
       * assemble the matrix or only the rhs.
       */
      void assemble(const bool assemble_system, const bool assemble_velocity);

      /*! \brief Solve the linear system using CG
       *
       *  After solving the linear system, the same AffineConstraints<double> as
       * used in assembly must be used again, to set the solution to the right
       * value at the constrained dofs.
       */
      std::pair<unsigned int, double> solve(const bool solver_for_velocity);

      /*! \brief Run the simulation for one time step.
       *
       *  Unlike SCnsIM, InsIM or InsIMEX, SCnsEX solves for the absolute
       * velocity and pressure other than the increments. Therefore, the
       * Dirichlet BCs in SCnsEX are always applied as non-zero constraints.
       * Zero constraints are not in use here.
       */
      void run_one_step(bool apply_nonzero_constraints,
                        bool assemble_system) override;

      /*! The intermiediate solution within every time step, generated from each
       * iteration.
       */
      PETScWrappers::MPI::BlockVector intermediate_solution;

      /**
       * Same as intermediate solution, we need this because intermediate
       * solution non-ghosted, and a ghosted vector is needed for interpolation
       * to the quadrature points when do the assembly.
       */
      PETScWrappers::MPI::BlockVector evaluation_point;

      /** \breif local_matrices
       * The local matrices is stored in each cell such that the program does
       * not need repeatly assemble the system matrix. Usually, one global
       * matrix suffices for such purpose. However, for time-dependent BC, we
       * need to update the inhomogenius constraints which needs the information
       * from the local system matrices.
       */
      CellDataStorage<
        typename parallel::distributed::Triangulation<dim>::cell_iterator,
        FullMatrix<double>>
        local_matrices;

      /** \breif boundary_condition_time_limits
       * This map stores the time limit for each time dependent hard coded
       * boundary conditions. In the explicit solver, the system matrix does not
       * have to re-assemble at each time step unless a time dependent boundary
       * condition is applied. Using this time limit helps accelerate the solver
       * when the hard coded condition (for example, a pulse) becomes zero after
       * some time and does not require to re-calculate the value anymore.
       */
      std::map<unsigned int, double> boundary_condition_time_limits;
    };
  } // namespace MPI
} // namespace Fluid

#endif

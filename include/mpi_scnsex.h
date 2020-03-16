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
     * SUPG incomplete Schur complement right preconditioner is applied, which
     * does modify the linear system a little bit, and requires the velocity
     * shape functions to be one order higher than that of the pressure.
     */
    template <int dim>
    class SCnsEX : public FluidSolver<dim>
    {
    public:
      //! Constructor.
      SCnsEX(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &,
             std::shared_ptr<Function<dim>> bc =
               std::make_shared<Functions::ZeroFunction<dim>>(
                 Functions::ZeroFunction<dim>(dim + 1)),
             std::shared_ptr<Function<dim>> pml =
               std::make_shared<Functions::ZeroFunction<dim>>(
                 Functions::ZeroFunction<dim>(dim + 1)),
             std::shared_ptr<TensorFunction<1, dim>> bf =
               std::make_shared<ZeroTensorFunction<1, dim>>(
                 ZeroTensorFunction<1, dim>()));
      ~SCnsEX(){};
      //! Run the simulation.
      void run();

    private:
      using FluidSolver<dim>::setup_dofs;
      using FluidSolver<dim>::make_constraints;
      using FluidSolver<dim>::setup_cell_property;
      using FluidSolver<dim>::refine_mesh;
      using FluidSolver<dim>::output_results;
      using FluidSolver<dim>::save_checkpoint;
      using FluidSolver<dim>::load_checkpoint;
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
      using FluidSolver<dim>::present_solution;
      using FluidSolver<dim>::solution_increment;
      using FluidSolver<dim>::system_rhs;
      using FluidSolver<dim>::fsi_acceleration;
      using FluidSolver<dim>::stress;
      using FluidSolver<dim>::parameters;
      using FluidSolver<dim>::mpi_communicator;
      using FluidSolver<dim>::pcout;
      using FluidSolver<dim>::owned_partitioning;
      using FluidSolver<dim>::relevant_partitioning;
      using FluidSolver<dim>::locally_owned_scalar_dofs;
      using FluidSolver<dim>::locally_relevant_dofs;
      using FluidSolver<dim>::locally_relevant_scalar_dofs;
      using FluidSolver<dim>::times_and_names;
      using FluidSolver<dim>::time;
      using FluidSolver<dim>::timer;
      using FluidSolver<dim>::timer2;
      using FluidSolver<dim>::cell_property;
      using FluidSolver<dim>::boundary_values;

      /// Specify the sparsity pattern and reinit matrices and vectors based on
      /// the dofs and constraints.
      virtual void initialize_system() override;

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
      void assemble(const bool assemble_velocity);

      /*! \brief Solve the linear system using FGMRES solver plus block
       * preconditioner.
       *
       *  After solving the linear system, the same AffineConstraints<double> as
       * used in assembly must be used again, to set the solution to the right
       * value at the constrained dofs.
       */
      std::pair<unsigned int, double> solve(const bool solver_for_velocity);

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

      /*! \brief Apply the initial condition
       *
       * This is a hard-coded function that is only used for VF cases where an
       * initial condition is applied for fast convergence. For general cases,
       * do not include it.
       */
      void apply_initial_condition();

      /// The increment at a certain Newton iteration.
      PETScWrappers::MPI::BlockVector intermediate_solution;

      /**
       * The latest know solution plus the cumulation of all newton_updates
       * in the current time step, which approaches to the new present_solution.
       */
      PETScWrappers::MPI::BlockVector evaluation_point;

      /** \brief sigma_pml_field
       * the sigma_pml_field is predefined outside the class. It specifies
       * the sigma PML field to determine where and how sigma pml is
       * distributed. With strong sigma PML it absorbs faster waves/vortices
       * but reflects more slow waves/vortices.
       */
      std::shared_ptr<Function<dim>> sigma_pml_field;

      /// Hard-coded body force. It will be added onto gravity.
      std::shared_ptr<TensorFunction<1, dim>> body_force;
    };
  } // namespace MPI
} // namespace Fluid

#endif

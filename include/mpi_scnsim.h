#ifndef MPI_SCNSIM
#define MPI_SCNSIM

#include "mpi_fluid_solver.h"
#include "preconditioner_pilut.h"

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
    class SCnsIM : public FluidSolver<dim>
    {
    public:
      //! Constructor.
      SCnsIM(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &,
             std::shared_ptr<Function<dim>> bc =
               std::make_shared<Functions::ZeroFunction<dim>>(
                 Functions::ZeroFunction<dim>(dim + 1)),
             std::shared_ptr<Function<dim>> pml =
               std::make_shared<Functions::ZeroFunction<dim>>(
                 Functions::ZeroFunction<dim>(dim + 1)));
      ~SCnsIM(){};
      //! Run the simulation.
      void run();

    private:
      class BlockIncompSchurPreconditioner;

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
      using FluidSolver<dim>::dof_handler;
      using FluidSolver<dim>::volume_quad_formula;
      using FluidSolver<dim>::face_quad_formula;
      using FluidSolver<dim>::zero_constraints;
      using FluidSolver<dim>::nonzero_constraints;
      using FluidSolver<dim>::sparsity_pattern;
      using FluidSolver<dim>::system_matrix;
      using FluidSolver<dim>::present_solution;
      using FluidSolver<dim>::solution_increment;
      using FluidSolver<dim>::system_rhs;
      using FluidSolver<dim>::stress;
      using FluidSolver<dim>::parameters;
      using FluidSolver<dim>::mpi_communicator;
      using FluidSolver<dim>::pcout;
      using FluidSolver<dim>::owned_partitioning;
      using FluidSolver<dim>::relevant_partitioning;
      using FluidSolver<dim>::locally_relevant_dofs;
      using FluidSolver<dim>::times_and_names;
      using FluidSolver<dim>::time;
      using FluidSolver<dim>::timer;
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

      PETScWrappers::MPI::SparseMatrix Abs_A_matrix;
      PETScWrappers::MPI::SparseMatrix schur_matrix;
      PETScWrappers::MPI::SparseMatrix B2pp_matrix;
      /// The increment at a certain Newton iteration.
      PETScWrappers::MPI::BlockVector newton_update;

      /**
       * The latest know solution plus the cumulation of all newton_updates
       * in the current time step, which approaches to the new present_solution.
       */
      PETScWrappers::MPI::BlockVector evaluation_point;

      /// The BlockIncompSchurPreconditioner for the entire system.
      std::shared_ptr<BlockIncompSchurPreconditioner> preconditioner;

      /** \brief sigma_pml_field
       * the sigma_pml_field is predefined outside the class. It specifies
       * the sigma PML field to determine where and how sigma pml is
       * distributed. With strong sigma PML it absorbs faster waves/vortices
       * but reflects more slow waves/vortices.
       */
      std::shared_ptr<Function<dim>> sigma_pml_field;

      /** \brief Incomplete Schur Complement Block Preconditioner
       * The format of this preconditioner is as follow:
       *
       * |Pvv^-1  -Pvv^-1*Avp*Tpp^-1|*|I            0|
       * |                          | |              |
       * |0            Tpp^-1       | |-Apv*Pvv^-1  I|
       * With Pvv the ILU(0) of Avv,
       * and Tpp the incomplete Schur complement.
       * The evaluation for Tpp is in SchurComplementTpp class,
       * and its inverse is solved by performing some GMRES iterations
       * By using B2pp = ILU(0) of (App - Apv*(rowsum|Avv|)^-1*Avp
       * as preconditioner.
       * This preconditioner is proposed in:
       * T. Washio et al., A robust preconditioner for fluid–structure
       * interaction problems, Comput. Methods Appl. Mech. Engrg.
       * 194 (2005) 4027–4047
       */
      class BlockIncompSchurPreconditioner : public Subscriptor
      {
      public:
        /// Constructor.
        BlockIncompSchurPreconditioner(
          TimerOutput &timer,
          const std::vector<IndexSet> &owned_partitioning,
          const PETScWrappers::MPI::BlockSparseMatrix &system,
          PETScWrappers::MPI::SparseMatrix &absA,
          PETScWrappers::MPI::SparseMatrix &schur,
          PETScWrappers::MPI::SparseMatrix &B2pp);

        /// The matrix-vector multiplication must be defined.
        void vmult(PETScWrappers::MPI::BlockVector &dst,
                   const PETScWrappers::MPI::BlockVector &src) const;
        /// Accessors for the blocks of the system matrix for clearer
        /// representation
        const PETScWrappers::MPI::SparseMatrix &Avv() const
        {
          return system_matrix->block(0, 0);
        }
        const PETScWrappers::MPI::SparseMatrix &Avp() const
        {
          return system_matrix->block(0, 1);
        }
        const PETScWrappers::MPI::SparseMatrix &Apv() const
        {
          return system_matrix->block(1, 0);
        }
        const PETScWrappers::MPI::SparseMatrix &App() const
        {
          return system_matrix->block(1, 1);
        }
        int get_Tpp_itr_count() const { return Tpp_itr; }
        void Erase_Tpp_count() { Tpp_itr = 0; }

      private:
        class SchurComplementTpp;

        /// We would like to time the BlockSchuPreconditioner in detail.
        TimerOutput &timer;

        /// dealii smart pointer checks if an object is still being referenced
        /// when it is destructed therefore is safer than plain reference.
        const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
          system_matrix;
        const SmartPointer<PETScWrappers::MPI::SparseMatrix> Abs_A_matrix;
        const SmartPointer<PETScWrappers::MPI::SparseMatrix> schur_matrix;
        const SmartPointer<PETScWrappers::MPI::SparseMatrix> B2pp_matrix;

        PreconditionEuclid Pvv_inverse;
        PreconditionEuclid B2pp_inverse;

        std::shared_ptr<SchurComplementTpp> Tpp;
        // iteration counter for solving Tpp
        mutable int Tpp_itr;
        class SchurComplementTpp : public Subscriptor
        {
        public:
          SchurComplementTpp(
            TimerOutput &timer,
            const std::vector<IndexSet> &owned_partitioning,
            const PETScWrappers::MPI::BlockSparseMatrix &system,
            const PETScWrappers::PreconditionerBase &Pvvinv);
          void vmult(PETScWrappers::MPI::Vector &dst,
                     const PETScWrappers::MPI::Vector &src) const;

        private:
          TimerOutput &timer;
          const SmartPointer<const PETScWrappers::MPI::BlockSparseMatrix>
            system_matrix;
          const PETScWrappers::PreconditionerBase *Pvv_inverse;
          PETScWrappers::MPI::BlockVector dumb_vector;
        };
      };
    };
  } // namespace MPI
} // namespace Fluid

#endif

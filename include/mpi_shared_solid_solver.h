#ifndef MPI_SOLID_SOLVER_SHARED
#define MPI_SOLID_SOLVER_SHARED

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>

#include "parameters.h"
#include "utilities.h"

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    /// Base class for all parallel solid solvers.
    template <int dim>
    class SharedSolidSolver
    {
    public:
      SharedSolidSolver(Triangulation<dim> &,
                        const Parameters::AllParameters &);
      ~SharedSolidSolver();
      void run();
      PETScWrappers::MPI::Vector get_current_solution() const;

    protected:
      /**
       * Set up the DofHandler, reorder the grid, sparsity pattern.
       */
      virtual void setup_dofs();

      /**
       * Initialize the matrix, solution, and rhs. This is separated from
       * setup_dofs because if we may want to transfer solution from one grid
       * to another in the refine_mesh.
       */
      virtual void initialize_system();

      /**
       * Assemble both the system matrices and rhs.
       */
      virtual void assemble_system(bool) = 0;

      /**
       * Run one time step.
       */
      virtual void run_one_step(bool) = 0;

      /**
       * Solve the linear system. Returns the number of
       * CG iterations and the final residual.
       */
      std::pair<unsigned int, double>
      solve(const PETScWrappers::MPI::SparseMatrix &,
            PETScWrappers::MPI::Vector &,
            const PETScWrappers::MPI::Vector &);

      /**
       * Output the time-dependent solution in vtu format.
       */
      void output_results(const unsigned int) const;

      /**
       * Refine mesh and transfer solution.
       */
      void refine_mesh(const unsigned int, const unsigned int);

      Triangulation<dim> &triangulation;
      Parameters::AllParameters parameters;
      DoFHandler<dim> dof_handler;
      DoFHandler<dim> dg_dof_handler; //!< Dof handler for dg_fe, which has one
                                      //!< dof per vertex.
      FESystem<dim> fe;
      FE_DGQ<dim>
        dg_fe; //!< Discontinous Glerkin FE for the nodal strain/stress
      const QGauss<dim>
        volume_quad_formula; //!< Quadrature formula for volume integration.
      const QGauss<dim - 1>
        face_quad_formula; //!< Quadrature formula for face integration.

      /**
       * Constraints to handle both hanging nodes and Dirichlet boundary
       * conditions.
       */
      ConstraintMatrix constraints;

      PETScWrappers::MPI::SparseMatrix
        system_matrix; //!< \f$ M + \beta{\Delta{t}}^2K \f$.
      PETScWrappers::MPI::SparseMatrix
        mass_matrix; //!< Required by hyperelastic solver.
      PETScWrappers::MPI::SparseMatrix
        stiffness_matrix; //!< The stiffness is used in the rhs.
      PETScWrappers::MPI::Vector system_rhs;

      /**
       * In the Newmark-beta method, acceleration is the variable to solve at
       * every
       * timestep. But displacement and velocity also contribute to the rhs of
       * the equation. For the sake of clarity, we explicitly store two sets of
       * accleration, velocity and displacement.
       */
      PETScWrappers::MPI::Vector current_acceleration;
      PETScWrappers::MPI::Vector current_velocity;
      PETScWrappers::MPI::Vector current_displacement;
      PETScWrappers::MPI::Vector previous_acceleration;
      PETScWrappers::MPI::Vector previous_velocity;
      PETScWrappers::MPI::Vector previous_displacement;

      MPI_Comm mpi_communicator;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;
      ConditionalOStream pcout;
      Utils::Time time;
      mutable TimerOutput timer;
      IndexSet locally_owned_dofs;
      IndexSet locally_relevant_dofs;
    };
  } // namespace MPI
} // namespace Solid

#endif

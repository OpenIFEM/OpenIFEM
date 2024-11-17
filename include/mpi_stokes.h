#ifndef MPI_STOKES
#define MPI_STOKES

#include "mpi_fluid_solver.h"
#include <cmath>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/vector.h>

// i dont think LA in necessary, try using petscwrappers for now
/*
 namespace LA
  {
  #if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
  #  define USE_PETSC_LA
  #elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
  #else
  #  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
  #endif
  } // namespace LA
  */

namespace Fluid

{
  namespace MPI

  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;
    /*
    template <int dim>
    struct InnerPreconditioner;

    // In 2D, we are going to use a sparse direct solver as preconditioner:
    template <>
    struct InnerPreconditioner<2>
    {
      using type = SparseDirectUMFPACK;
    };

    // And the ILU preconditioning in 3D, called by SparseILU:
    template <>
    struct InnerPreconditioner<3>
    {
      using type = SparseILU<double>;

    };
    */

    template <int dim>
    class Stokes : public FluidSolver<dim>
    {

    public:
      Stokes(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &);

      ~Stokes(){};
      void run();

    private:
      using FluidSolver<dim>::setup_dofs;
      using FluidSolver<dim>::make_constraints;
      using FluidSolver<dim>::setup_cell_property;
      using FluidSolver<dim>::initialize_system;
      using FluidSolver<dim>::refine_mesh;
      using FluidSolver<dim>::update_stress;
      using FluidSolver<dim>::dofs_per_block;
      using FluidSolver<dim>::triangulation;
      using FluidSolver<dim>::fe;
      using FluidSolver<dim>::scalar_fe;
      using FluidSolver<dim>::dof_handler;
      using FluidSolver<dim>::scalar_dof_handler;
      using FluidSolver<dim>::volume_quad_formula;
      using FluidSolver<dim>::face_quad_formula;
      using FluidSolver<dim>::sparsity_pattern;
      using FluidSolver<dim>::system_matrix;
      using FluidSolver<dim>::mass_matrix;
      // using FluidSolver<dim>::mass_schur_pattern;
      using FluidSolver<dim>::mass_schur;
      using FluidSolver<dim>::present_solution;
      using FluidSolver<dim>::system_rhs;
      using FluidSolver<dim>::time;
      using FluidSolver<dim>::timer;
      using FluidSolver<dim>::parameters;
      using FluidSolver<dim>::cell_property;
      // using FluidSolver<dim>::boundary_values;
      using FluidSolver<dim>::stress;
      using FluidSolver<dim>::mpi_communicator;
      using FluidSolver<dim>::pcout;
      using FluidSolver<dim>::owned_partitioning;
      using FluidSolver<dim>::relevant_partitioning;
      using FluidSolver<dim>::locally_owned_scalar_dofs;
      using FluidSolver<dim>::locally_relevant_dofs;
      using FluidSolver<dim>::locally_relevant_scalar_dofs;
      using FluidSolver<dim>::pvd_writer;

      void initialize_system() override;

      void output_results(const unsigned int) const;

      void set_up_boundary_values();

      void assemble();

      std::pair<unsigned int, double> solve();

      void run_one_step(bool apply_nonzero_constraints,
                        bool assemble_system = true) override;

      void run_one_step();

      PETScWrappers::MPI::BlockVector solution;

      BlockSparsityPattern preconditioner_sparsity_pattern;

      PETScWrappers::MPI::BlockSparseMatrix preconditioner_matrix;

      AffineConstraints<double> constraints;

      // std::shared_ptr<typename InnerPreconditioner<dim>::type>
      // A_preconditioner;
    };

  } // namespace MPI

} // namespace Fluid

#endif // MPI_STOKES
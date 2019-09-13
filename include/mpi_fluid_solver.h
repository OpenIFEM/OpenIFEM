#ifndef MPI_FLUID_SOLVER
#define MPI_FLUID_SOLVER

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <deal.II/physics/elasticity/standard_tensors.h>

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

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "parameters.h"
#include "utilities.h"

namespace fs = std::experimental::filesystem;

namespace MPI
{
  template <int dim>
  class FSI;
  class ShellFSI;
}

namespace Fluid
{
  using namespace dealii;

  namespace MPI
  {
    /// Base class for all mpi fluid solvers.
    template <int dim>
    class FluidSolver
    {
    public:
      //! FSI solver need access to the private members of this solver.
      friend ::MPI::FSI<dim>;
      friend ::MPI::ShellFSI;

      //! Constructor.
      FluidSolver(parallel::distributed::Triangulation<dim> &,
                  const Parameters::AllParameters &,
                  std::shared_ptr<Function<dim>> bc =
                    std::make_shared<Functions::ZeroFunction<dim>>(
                      Functions::ZeroFunction<dim>(dim + 1)));

      //! Run the simulation.
      virtual void run() = 0;

      //! Destructor
      ~FluidSolver();

      //! Return the solution for testing.
      PETScWrappers::MPI::BlockVector get_current_solution() const;

    protected:
      class BoundaryValues;
      struct CellProperty;

      //! Pure abstract function to run simulation for one step
      virtual void run_one_step(bool apply_nonzero_constraints,
                                bool assemble_system = true) = 0;

      //! Set up the dofs based on the finite element and renumber them.
      void setup_dofs();

      //! Set up the nonzero and zero constraints.
      void make_constraints();

      //! Initialize the cell properties, which only matters in FSI
      //! applications.
      void setup_cell_property();

      /// Specify the sparsity pattern and reinit matrices and vectors based on
      /// the dofs and constraints.
      virtual void initialize_system();

      /// Mesh adaption.
      void refine_mesh(const unsigned int, const unsigned int);

      /// Output in vtu format.
      void output_results(const unsigned int) const;

      /// Update stress to output
      virtual void update_stress();

      /// Save checkpoint for restart.
      void save_checkpoint(const int);

      /// Load from checkpoint to restart.
      bool load_checkpoint();

      std::vector<types::global_dof_index> dofs_per_block;

      parallel::distributed::Triangulation<dim> &triangulation;
      FESystem<dim> fe;
      FE_Q<dim> scalar_fe;
      DoFHandler<dim> dof_handler;
      DoFHandler<dim> scalar_dof_handler;
      QGauss<dim> volume_quad_formula;
      QGauss<dim - 1> face_quad_formula;

      AffineConstraints<double> zero_constraints;
      AffineConstraints<double> nonzero_constraints;

      BlockSparsityPattern sparsity_pattern;
      PETScWrappers::MPI::BlockSparseMatrix system_matrix;
      PETScWrappers::MPI::BlockSparseMatrix mass_matrix;
      PETScWrappers::MPI::BlockSparseMatrix mass_schur;

      /// The latest known solution.
      PETScWrappers::MPI::BlockVector present_solution;
      PETScWrappers::MPI::BlockVector solution_increment;
      PETScWrappers::MPI::BlockVector system_rhs;

      /**
       * Nodal strain and stress obtained by taking the average of surrounding
       * cell-averaged strains and stresses. Their sizes are
       * [dim, dim, scalar_dof_handler.n_dofs()], i.e., stress[i][j][k]
       * denotes sigma_{ij} at vertex k.
       */
      mutable std::vector<std::vector<PETScWrappers::MPI::Vector>> stress;

      Parameters::AllParameters parameters;

      MPI_Comm mpi_communicator;

      ConditionalOStream pcout;

      /// The IndexSets of owned velocity and pressure respectively.
      std::vector<IndexSet> owned_partitioning;

      /// The IndexSets of relevant velocity and pressure respectively.
      std::vector<IndexSet> relevant_partitioning;

      /// The IndexSets of owned and relevant sclalar dofs for stress
      /// computation.
      IndexSet locally_owned_scalar_dofs;
      IndexSet locally_relevant_scalar_dofs;

      /// The IndexSet of all relevant dofs. This seems to be redundant but
      /// handy.
      IndexSet locally_relevant_dofs;

      /// The vector to store vtu filenames that will be written into pvd file.
      mutable std::vector<std::pair<double, std::string>> times_and_names;

      Utils::Time time;
      mutable TimerOutput timer;
      mutable TimerOutput timer2;

      CellDataStorage<
        typename parallel::distributed::Triangulation<dim>::cell_iterator,
        CellProperty>
        cell_property;

      /// Hard-coded boundary values, only used when told so in the input
      /// parameters.
      std::shared_ptr<Function<dim>> boundary_values;

      /// A data structure that caches the real/artificial fluid indicator,
      /// FSI stress, and FSI acceleration terms at quadrature points, that
      /// will only be used in FSI simulations.
      struct CellProperty
      {
        int indicator; //!< Domain indicator: 1 for artificial fluid 0 for real
                       //! fluid.
        Tensor<1, dim>
          fsi_acceleration; //!< The acceleration term in FSI force.
        SymmetricTensor<2, dim> fsi_stress; //!< The stress term in FSI force.
        int material_id; //!< The material id of the surrounding solid cell.
      };
    };
  } // namespace MPI
} // namespace Fluid

#endif

#ifndef FLUID_SOLVER
#define FLUID_SOLVER

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
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
#include <deal.II/lac/sparse_ilu.h>

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

#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "parameters.h"
#include "utilities.h"

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  /// Base class for all fluid solvers.
  template <int dim>
  class FluidSolver
  {
  public:
    //! FSI solver need access to the private members of this solver.
    friend FSI<dim>;

    //! Constructor
    FluidSolver(Triangulation<dim> &,
                const Parameters::AllParameters &,
                std::shared_ptr<Function<dim>> bc =
                  std::make_shared<Functions::ZeroFunction<dim>>(
                    Functions::ZeroFunction<dim>(dim + 1)));

    //! Run the simulation
    virtual void run() = 0;

    //! Destructor
    ~FluidSolver() { timer.print_summary(); }

    //! Return the solution for testing.
    BlockVector<double> get_current_solution() const;

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

    //! Initialize the cell properties, which only matters in FSI applications.
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

    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> &triangulation;
    FESystem<dim> fe;
    FE_Q<dim> scalar_fe;
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> scalar_dof_handler;
    QGauss<dim> volume_quad_formula;
    QGauss<dim - 1> face_quad_formula;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockSparseMatrix<double> mass_matrix;
    SparsityPattern mass_schur_pattern;
    SparseMatrix<double> mass_schur;

    /// The latest known solution.
    BlockVector<double> present_solution;
    BlockVector<double> solution_increment;
    BlockVector<double> system_rhs;

    /**
     * Nodal strain and stress obtained by taking the average of surrounding
     * cell-averaged strains and stresses. Their sizes are
     * [dim, dim, scalar_dof_handler.n_dofs()], i.e., stress[i][j][k]
     * denotes sigma_{ij} at vertex k.
     */
    mutable std::vector<std::vector<Vector<double>>> stress;

    Utils::Time time;
    mutable TimerOutput timer;

    Parameters::AllParameters parameters;

    CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
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
      Tensor<1, dim> fsi_acceleration; //!< The acceleration term in FSI force.
      SymmetricTensor<2, dim> fsi_stress; //!< The stress term in FSI force.
    };
  };
} // namespace Fluid

#endif

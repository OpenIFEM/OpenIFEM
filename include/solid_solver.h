#ifndef SOLID_SOLVER
#define SOLID_SOLVER

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "parameters.h"
#include "utilities.h"

template <int>
class FSI;

namespace Solid
{
  using namespace dealii;

  /// Base class for all solid solvers.
  template <int dim, int spacedim = dim>
  class SolidSolver
  {
  public:
    friend FSI<spacedim>;

    SolidSolver(Triangulation<dim, spacedim> &,
                const Parameters::AllParameters &);
    ~SolidSolver();
    void run();
    Vector<double> get_current_solution() const;

  protected:
    struct CellProperty;
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
     * Update the cached strain and stress to output.
     */
    virtual void update_strain_and_stress() = 0;

    /**
     * Run one time step.
     */
    virtual void run_one_step(bool) = 0;

    /**
     * Solve the linear system. Returns the number of
     * CG iterations and the final residual.
     */
    std::pair<unsigned int, double> solve(const SparseMatrix<double> &,
                                          Vector<double> &,
                                          const Vector<double> &);

    /**
     * Output the time-dependent solution in vtu format.
     */
    void output_results(const unsigned int);

    /**
     * Refine mesh and transfer solution.
     */
    void refine_mesh(const unsigned int, const unsigned int);

    Triangulation<dim, spacedim> &triangulation;
    Parameters::AllParameters parameters;
    DoFHandler<dim, spacedim> dof_handler;
    DoFHandler<dim, spacedim> scalar_dof_handler; //!< Scalar-valued DoFHandler.
    FESystem<dim, spacedim> fe;
    FE_Q<dim, spacedim> scalar_fe; //!< Scalar FE for nodal strain/stress.
    const QGauss<dim>
      volume_quad_formula; //!< Quadrature formula for volume integration.
    const QGauss<dim - 1>
      face_quad_formula; //!< Quadrature formula for face integration.

    /**
     * Constraints to handle both hanging nodes and Dirichlet boundary
     * conditions.
     */
    AffineConstraints<double> constraints;

    SparsityPattern pattern;
    SparseMatrix<double> system_matrix; //!< \f$ M + \beta{\Delta{t}}^2K \f$.
    SparseMatrix<double> mass_matrix;   //!< Required by hyperelastic solver.
    SparseMatrix<double>
      stiffness_matrix; //!< The stiffness is used in the rhs.
    Vector<double> system_rhs;

    /**
     * In the Newmark-beta method, acceleration is the variable to solve at
     * every
     * timestep. But displacement and velocity also contribute to the rhs of the
     * equation.
     * For the sake of clarity, we explicitly store two sets of accleration,
     * velocity
     * and displacement.
     */
    Vector<double> current_acceleration;
    Vector<double> current_velocity;
    Vector<double> current_displacement;
    Vector<double> previous_acceleration;
    Vector<double> previous_velocity;
    Vector<double> previous_displacement;

    /**
     * Nodal strain and stress obtained by taking the average of surrounding
     * cell-averaged strains and stresses. Their sizes are
     * [dim, dim, scalar_dof_handler.n_dofs()], i.e., stress[i][j][k]
     * denotes sigma_{ij} at vertex k.
     */
    mutable std::vector<std::vector<Vector<double>>> strain, stress;
    mutable std::vector<Vector<double>> cellwise_stress;

    Utils::Time time;
    mutable TimerOutput timer;

    CellDataStorage<typename Triangulation<dim, spacedim>::cell_iterator,
                    CellProperty>
      cell_property;

    /**
     * The fluid traction in FSI simulation, which should be set by the FSI.
     */
    struct CellProperty
    {
      Tensor<1, spacedim> fsi_traction;
    };
  };
} // namespace Solid

#endif

#ifndef HYPERELASTIC_SOLVER
#define HYPERELASTIC_SOLVER

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>

#include "neoHookean.h"
#include "parameters.h"
#include "utilities.h"

namespace
{
  /** \brief A simple struct to normalize errors.
   *
   * This struct will be more useful if
   * mixed formulation is used.
   */
  struct Errors
  {
    Errors() : norm(1.0) {}
    void reset() { norm = 1.0; }
    void normalize(const Errors &rhs)
    {
      if (rhs.norm != 0.0)
        {
          norm /= rhs.norm;
        }
    }
    double norm;
  };

  /** \brief Data to store at the quadrature points.
   *
   * We cache the kinematics information at quadrature points
   * by storing a PointHistory at each cell,
   * so that they can be conveniently accessed in the assembly
   * or post processing. We also store a material pointer
   * in case different material properties are used at
   * different locations.
   */
  template <int dim>
  class PointHistory
  {
    using ST = dealii::Physics::Elasticity::StandardTensors<dim>;

  public:
    PointHistory()
      : F_inv(ST::I),
        tau(dealii::SymmetricTensor<2, dim>()),
        Jc(dealii::SymmetricTensor<4, dim>()),
        dPsi_vol_dJ(0.0),
        d2Psi_vol_dJ2(0.0)
    {
    }
    virtual ~PointHistory() {}
    /** Initialize the members with the input parameters */
    void setup(const Parameters::AllParameters &);
    /**
     * Update the state with the displacement gradient
     * in the reference configuration.
     */
    void update(const Parameters::AllParameters &,
                const dealii::Tensor<2, dim> &);
    double get_det_F() const { return material->get_det_F(); }
    const dealii::Tensor<2, dim> &get_F_inv() const { return F_inv; }
    const dealii::SymmetricTensor<2, dim> &get_tau() const { return tau; }
    const dealii::SymmetricTensor<4, dim> &get_Jc() const { return Jc; }
    double get_dPsi_vol_dJ() const { return dPsi_vol_dJ; }
    double get_d2Psi_vol_dJ2() const { return d2Psi_vol_dJ2; }

  private:
    /** The specific hyperelastic material to use. */
    std::shared_ptr<Solid::HyperelasticMaterial<dim>> material;

    dealii::Tensor<2, dim> F_inv;
    dealii::SymmetricTensor<2, dim> tau;
    dealii::SymmetricTensor<4, dim> Jc;
    double dPsi_vol_dJ;
    double d2Psi_vol_dJ2;
  };
}

namespace Solid
{
  using namespace dealii;

  extern template class HyperelasticMaterial<2>;
  extern template class HyperelasticMaterial<3>;

  /** \brief Solver for hyperelastic materials
   *
   * Unlike LinearElasticSolver, we do not store material in this solver.
   * Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_44.html
   */
  template <int dim>
  class HyperelasticSolver
  {
  public:
    HyperelasticSolver(Triangulation<dim> &, const Parameters::AllParameters &);
    ~HyperelasticSolver() { this->dof_handler.clear(); }
    void run();

  private:
    void setup_dofs();

    /**
     * Assemble the lhs and rhs at the same time.
     */
    void assemble(bool);

    // Set up the quadrature point history
    void setup_qph();

    /**
     * Initialize matrices and vectors, this process
     * is separate from setup_dofs because if we refine
     * mesh in time-dependent simulation, we must
     * transfer solution after setup_dofs, and then
     * initialize.
     */
    void initialize_system();

    /** update the quadrature point history
     * The displacement is incremented at every iteration, so we have to
     * update the strain, stress etc. stored at quadrature points.
     */
    void update_qph(const dealii::Vector<double> &);

    // Using Newton iteration to solve for a nonlinear timestep.
    void solve_nonlinear_step(dealii::Vector<double> &);

    /* Solve a linear equation, return the number of iterations and residual. */
    std::pair<unsigned int, double>
    solve_linear_system(dealii::Vector<double> &);

    // Given the increment of the solution, return the current solution.
    dealii::Vector<double>
    get_total_solution(const dealii::Vector<double> &) const;

    void output_results(const unsigned int) const;

    Parameters::AllParameters parameters;
    double vol;
    Utils::Time time;
    mutable dealii::TimerOutput timer; // Record the time profile of the program

    /**
     * We store a PointHistory structure at every quadrature point,
     * so that kinematics information like F as well as material properties
     * can be cached.
     */
    dealii::CellDataStorage<typename dealii::Triangulation<dim>::cell_iterator,
                            PointHistory<dim>>
      quad_point_history;

    const unsigned int degree;
    const dealii::FESystem<dim> fe;
    dealii::Triangulation<dim> &triangulation;
    dealii::DoFHandler<dim> dof_handler;
    const unsigned int dofs_per_cell;
    const dealii::QGauss<dim> volume_quad_formula;
    const dealii::QGauss<dim - 1> face_quad_formula;
    const unsigned int n_q_points;
    const unsigned int n_f_q_points;
    // Tells dealii to view the dofs as a vector when necessary
    const dealii::FEValuesExtractors::Vector displacement;

    dealii::ConstraintMatrix constraints;
    dealii::SparsityPattern pattern;
    dealii::SparseMatrix<double> system_matrix;
    dealii::Vector<double> system_rhs;
    dealii::Vector<double> solution;

    /**
     * errorResidual: norm of the residual at a Newton iteration
     * errorResidual0: norm of the residual at the first iteration
     * errorResidualNorm: errorResidual/errorResidual0
     * errorUpdate: norm of the solution increment
     * errorUpdate0: norm of the solution increment at the first iteration
     * errorUpdateNorm: errorUpdate/errorUpdate0
     */
    Errors errorResidual, errorResidual0, errorResidualNorm, errorUpdate,
      errorUpdate0, errorUpdateNorm;

    // Reture the residual in the Newton iteration
    void get_error_residual(Errors &);
    // Compute the l2 norm of the solution increment
    void get_error_update(const dealii::Vector<double> &, Errors &);

    // Return the current volume of the geometry
    double compute_volume() const;

    // Print the header and footer of the output table
    void print_conv_header();
    void print_conv_footer();
  };
}

#endif

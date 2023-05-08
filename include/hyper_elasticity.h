#ifndef HYPER_ELASTICITY
#define HYPER_ELASTICITY

#include "kirchhoff_elastic_material.h"
#include "neo_hookean.h"
#include "solid_solver.h"
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

template <int>
class FSI;

namespace Internal
{
  using namespace dealii;

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
    using ST = Physics::Elasticity::StandardTensors<dim>;

  public:
    PointHistory()
      : F_inv(ST::I),
        tau(SymmetricTensor<2, dim>()),
        Jc(SymmetricTensor<4, dim>()),
        dPsi_vol_dJ(0.0),
        d2Psi_vol_dJ2(0.0)
    {
    }
    virtual ~PointHistory() {}
    /** Initialize the members with the input parameters */
    void setup(const Parameters::AllParameters &, const unsigned int &);
    /**
     * Update the state with the displacement gradient
     * in the reference configuration.
     */
    void update(const Parameters::AllParameters &, const Tensor<2, dim> &);
    double get_det_F() const { return material->get_det_F(); }
    const Tensor<2, dim> &get_F_inv() const { return F_inv; }
    const SymmetricTensor<2, dim> &get_tau() const { return tau; }
    const SymmetricTensor<4, dim> &get_Jc() const { return Jc; }
    double get_density() const { return material->get_density(); }
    double get_dPsi_vol_dJ() const { return dPsi_vol_dJ; }
    double get_d2Psi_vol_dJ2() const { return d2Psi_vol_dJ2; }

  private:
    /** The specific hyperelastic material to use. */
    std::shared_ptr<Solid::HyperElasticMaterial<dim>> material;
    Tensor<2, dim> F_inv;
    SymmetricTensor<2, dim> tau;
    SymmetricTensor<4, dim> Jc;
    double dPsi_vol_dJ;
    double d2Psi_vol_dJ2;
  };
} // namespace Internal

namespace Solid
{
  using namespace dealii;

  extern template class SolidSolver<2>;
  extern template class SolidSolver<3>;
  extern template class HyperElasticMaterial<2>;
  extern template class HyperElasticMaterial<3>;

  /** \brief Solver for hyperelastic materials
   *
   * The solver sets up a PointHistory object at every quadrature point,
   * where the material properties, deformation, and even stress state
   * are cached. Therefore the PointHistory has to be updated whenever
   * the deformation changes.
   *
   * Based on dealii tutorial [step-44]
   * (http://www.dealii.org/8.5.0/doxygen/deal.II/step_44.html)
   */
  template <int dim>
  class HyperElasticity : public SolidSolver<dim>
  {
  public:
    friend FSI<dim>;

    HyperElasticity(Triangulation<dim> &, const Parameters::AllParameters &);
    ~HyperElasticity() {}

  private:
    /**
     * Members in its template-base class.
     * Annoying C++ feature: the compiler does not know how to access
     * template-derived members unless you tell it explicitly by using
     * declarations or this->Foo.
     */
    using SolidSolver<dim>::triangulation;
    using SolidSolver<dim>::parameters;
    using SolidSolver<dim>::dof_handler;
    using SolidSolver<dim>::scalar_dof_handler;
    using SolidSolver<dim>::fe;
    using SolidSolver<dim>::scalar_fe;
    using SolidSolver<dim>::volume_quad_formula;
    using SolidSolver<dim>::face_quad_formula;
    using SolidSolver<dim>::constraints;
    using SolidSolver<dim>::pattern;
    using SolidSolver<dim>::system_matrix;
    using SolidSolver<dim>::mass_matrix;
    using SolidSolver<dim>::system_rhs;
    using SolidSolver<dim>::current_acceleration;
    using SolidSolver<dim>::current_velocity;
    using SolidSolver<dim>::current_displacement;
    using SolidSolver<dim>::previous_acceleration;
    using SolidSolver<dim>::previous_velocity;
    using SolidSolver<dim>::previous_displacement;
    using SolidSolver<dim>::strain;
    using SolidSolver<dim>::stress;
    using SolidSolver<dim>::time;
    using SolidSolver<dim>::timer;
    using SolidSolver<dim>::cell_property;
    using SolidSolver<dim>::nodal_mass;
    using SolidSolver<dim>::calculate_KE;
    using SolidSolver<dim>::nodal_forces_traction;
    using SolidSolver<dim>::added_mass_effect;
    using SolidSolver<dim>::system_matrix_updated;
    using SolidSolver<dim>::mass_matrix_updated;

    void initialize_system();

    virtual void update_strain_and_stress() override;

    /** Assemble the lhs and rhs at the same time. */
    void assemble_system(bool);

    /** Set up the quadrature point history. */
    void setup_qph();

    /**
     * \brief Update the quadrature point history.
     *
     * The displacement is incremented at every iteration, so we have to
     * update the strain, stress etc. stored at quadrature points.
     */
    void update_qph(const Vector<double> &);

    /// Run one time step.
    void run_one_step(bool);

    /**
     * We store a PointHistory structure at every quadrature point,
     * so that kinematics information like F as well as material properties
     * can be cached.
     */
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    Internal::PointHistory<dim>>
      quad_point_history;

    double error_residual; //!< Norm of the residual at a Newton iteration.
    double
      initial_error_residual; //!< Norm of the residual at the first iteration.
    double
      normalized_error_residual;    //!< error_residual / initial_error_residual
    double error_update;            //!< Norm of the solution increment.
    double initial_error_update;    //!< Norm of the solution increment at the
                                    //! first iteration.
    double normalized_error_update; //!< error_update / initial_error_update

    // Return the residual in the Newton iteration
    void get_error_residual(double &);
    // Compute the l2 norm of the solution increment
    void get_error_update(const Vector<double> &, double &);

    // Return the current volume of the geometry
    double compute_volume() const;
  };
} // namespace Solid

#endif

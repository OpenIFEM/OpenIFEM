#ifndef HYPERELASTIC_MATERIAL
#define HYPERELASTIC_MATERIAL

#include <deal.II/base/tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <memory>

#include "material.h"

namespace Solid
{
  /*! \brief An abstract class for hyperelastic materials.
   *
   *  It is supposed to be stored at every integration points so it contains
   *  simulation specific data as well, as well as support different material
   *  properties in different parts.
   *
   *  The methods to compute the stress and elasticity are very brutal:
   *  simply follow the formulas in Holzapfel's book because dealii offers
   *  all the necessary tensors that we need - we choose not to manually
   * simplify the derivation.
   *
   *  This class is written for diaplacement-based formulation,
   *  assuming the volumetric part of the energy potential is 1/2*kappa*(J-1)^2.
   */
  template <int dim>
  class HyperelasticMaterial : public Material<dim>
  {
    using ST = typename dealii::Physics::Elasticity::StandardTensors<dim>;

  public:
    HyperelasticMaterial()
      : Material<dim>(), kappa(0.0), det_F(1.0), b_bar(ST::I)
    {
    }

    HyperelasticMaterial(double k, double rho = 0.0)
      : Material<dim>(rho), kappa(k), det_F(1.0), b_bar(ST::I)
    {
    }

    virtual ~HyperelasticMaterial() {}

    /** Update the material model with deformation data. */
    virtual void update_data(const dealii::Tensor<2, dim> &);

    /** Return the Kirchhoff stress */
    virtual dealii::SymmetricTensor<2, dim> get_tau() const
    {
      return get_tau_iso() + get_tau_vol();
    }

    /** Return the spatial elasticity tensor multiplied with J */
    virtual dealii::SymmetricTensor<4, dim> get_Jc() const
    {
      return get_Jc_iso() + get_Jc_vol();
    }

    /** Return the J. */
    double get_det_F() { return det_F; }

    /* Return the derivative of the volumetric part of the energy potential
     * w.r.t the J. */
    virtual double get_dPsi_vol_dJ() const { return kappa * (det_F - 1); }

    /* Return the second order derivative of the volumetric part of the energy
     * potential w.r.t the J. */
    virtual double get_d2Psi_vol_dJ2() const { return kappa; }

  protected:
    const double kappa; //!< every hyperelastic material should have kappa

    // b_bar and det_F are not material properties,
    // they are stored just for convenience.

    double det_F; //!< determinant of the Jacobian matrix
    dealii::SymmetricTensor<2, dim>
      b_bar; //!< modified left Cauchy-Green tensor

    /** Return the isochoric part of the Kirchhoff stress. */
    virtual dealii::SymmetricTensor<2, dim> get_tau_iso() const
    {
      return ST::dev_P * get_tau_bar();
    }

    /** Return the volumetric part of the Kirchhoff stress. tau_{vol} = pI */
    virtual dealii::SymmetricTensor<2, dim> get_tau_vol() const
    {
      return det_F * get_dPsi_vol_dJ() * ST::I;
    }

    /** Return the volumetric part of the spatial elasticity tensor multiplied
     * with J */
    virtual dealii::SymmetricTensor<4, dim> get_Jc_vol() const;

    /** Return the isochoric part of the spatial elasticity tensor multiplied
     * with J */
    virtual dealii::SymmetricTensor<4, dim> get_Jc_iso() const;

    /** Return the fictitious Kirchhoff stress. Model-dependent. */
    virtual dealii::SymmetricTensor<2, dim> get_tau_bar() const = 0;

    /** Return the fictitious spatial elasticity tensor. Model-dependent. */
    virtual dealii::SymmetricTensor<4, dim> get_cc_bar() const = 0;
  };
} // namespace Solid

#endif

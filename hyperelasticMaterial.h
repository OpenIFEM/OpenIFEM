#ifndef HYPERELASTIC_MATERIAL
#define HYPERELASTIC_MATERIAL

#include <deal.II/base/tensor.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <memory>

#include "material.h"

namespace IFEM
{
  /*! \brief An abstract class for hyperelastic materials.
   *
   *  It is supposed to be stored at every integration points so it contains
   *  simulation specific data as well, as well as support different material
   *  properties in different parts.
   *
   *  The methods to compute the stress and elasticity are very brutal:
   *  simply follow the formulas in Holzapfel's book because dealii offers
   *  all the necessary tensors that we need - we choose not to manually simplify
   *  the derivation.
   *
   *  This class is written for diaplacement-based formulation,
   *  assuming the volumetric part of the energy potential is 1/2*kappa*(J-1)^2.
   */
  template<int dim>
  class HyperelasticMaterial : public Material<dim>
  {
  using ST = typename dealii::Physics::Elasticity::StandardTensors<dim>;
  public:
    HyperelasticMaterial() : Material<dim>(), kappa(0.0), detF(1.0), bbar(ST::I) {}
    HyperelasticMaterial(double rho = 0.0) : Material<dim>(rho),
      kappa(0.0), detF(1.0), bbar(ST::I) {}
    virtual ~HyperelasticMaterial() {}
    /** Update the material model with deformation data. */
    virtual void updateData(const dealii::Tensor<2, dim>&);
    /** Return the Kirchhoff stress */
    virtual dealii::SymmetricTensor<2, dim> getTau() const {return getTauIso() + getTauVol();}
    /** Return the spatial elasticity tensor multiplied with J */
    virtual dealii::SymmetricTensor<4, dim> getJc() const {return getJcIso() + getJcVol();}
    /** Return the J. */
    double getDetF() {return detF;}

    /* Return the derivative of the volumetric part of the energy potential
     * w.r.t the J. */
    virtual double get_dPsi_vol_dJ() const {return this->kappa*(this->detF - 1);}
    /* Return the second order derivative of the volumetric part of the energy potential
     * w.r.t the J. */
    virtual double get_d2Psi_vol_dJ2() const {return this->kappa;}

    /** Return the isochoric part of the Kirchhoff stress. */
    virtual dealii::SymmetricTensor<2, dim> getTauIso() const
      {return ST::dev_P*getTauBar();}
    /** Return the volumetric part of the Kirchhoff stress. tau_{vol} = pI */
    virtual dealii::SymmetricTensor<2, dim> getTauVol() const
      {return get_dPsi_vol_dJ()*ST::I;}
    /** Return the volumetric part of the spatial elasticity tensor multiplied with J */
    virtual dealii::SymmetricTensor<4, dim> getJcVol() const;
    /** Return the isochoric part of the spatial elasticity tensor multiplied with J */
    virtual dealii::SymmetricTensor<4, dim> getJcIso() const;
    
    /** Return the fictitious Kirchhoff stress. Model-dependent. */
    virtual dealii::SymmetricTensor<2, dim> getTauBar() const = 0;
    /** Return the fictitious spatial elasticity tensor. Model-dependent. */
    virtual dealii::SymmetricTensor<4, dim> getCcBar() const = 0;
  protected:
    const double kappa; // every hyperelastic material should have kappa
    /** bbar and detF are not material properties, they are stored just for convenience. */
    double detF; // determinant of the Jacobian matrix
    dealii::SymmetricTensor<2, dim> bbar;
  };
}

#endif

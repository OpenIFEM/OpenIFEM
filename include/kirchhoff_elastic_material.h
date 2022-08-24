#ifndef KIRCHHOFF_ELASTIC_MATERIAL
#define KIRCHHOFF_ELASTIC_MATERIAL

#include "hyper_elastic_material.h"
#include <deal.II/base/symmetric_tensor.h>

namespace Solid
{

  extern template class HyperElasticMaterial<2>;
  extern template class HyperElasticMaterial<3>;

  /*! \breif Kirchhoff elastic material.
   */
  template <int dim>
  class KirchhoffElasticMaterial : public HyperElasticMaterial<dim>
  {

    using ST = typename dealii::Physics::Elasticity::StandardTensors<dim>;

  public:
    KirchhoffElasticMaterial()
      : HyperElasticMaterial<dim>(), E(0.0), nu(0.0), lambda(0.0), mu(0.0)
    {
    }

    KirchhoffElasticMaterial(double param1, double param2, double rho = 0.0)
      : HyperElasticMaterial<dim>(param2, rho), E(param1), nu(param2)
    {
      this->lambda = this->E * this->nu / ((1 + this->nu) * (1 - 2 * this->nu));
      this->mu = this->E / (2 * (1 + this->nu));
    }

    /** Compute PK2 stress */
    dealii::SymmetricTensor<2, dim>
    get_pk2_stress(const dealii::Tensor<2, dim> &F) const
    {
      const dealii::SymmetricTensor<2, dim> E_tensor =
        dealii::Physics::Elasticity::Kinematics::E(F);
      dealii::SymmetricTensor<2, dim> pk2_stress =
        this->lambda * dealii::trace(E_tensor) * ST::I +
        2 * this->mu * E_tensor;
      return pk2_stress;
    }

    /** Compute the tangent elasticity tensor*/

    virtual dealii::SymmetricTensor<4, dim> get_Jc() const override
    {
      return (this->lambda) * ST::IxI + 2 * (this->mu) * ST::S;
    }

    virtual dealii::SymmetricTensor<2, dim> get_tau_bar() const override
    {
      return dealii::SymmetricTensor<2, dim>();
    }

    virtual dealii::SymmetricTensor<4, dim> get_cc_bar() const override

    {
      return dealii::SymmetricTensor<4, dim>();
    }

  private:
    double E;      //!< Young's modulus
    double nu;     //!< Poisson's ratio
    double lambda; //!< First lame parameter
    double mu;     //!< Second lame parameter
  };

} // namespace Solid

#endif

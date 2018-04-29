#ifndef NEO_HOOKEAN
#define NEO_HOOKEAN

#include "hyperelasticMaterial.h"

namespace Solid
{
  extern template class HyperelasticMaterial<2>;
  extern template class HyperelasticMaterial<3>;

  /*! \brief Neo-Hookean material.
   *
   * The isotropic part of the strain energy in the Neo-Hookean
   * model is written as \f$ C_1(\bar{I}_1 -3) \f$.
   */
  template <int dim>
  class NeoHookean : public HyperelasticMaterial<dim>
  {
  public:
    NeoHookean() : HyperelasticMaterial<dim>(), c1(0.0) {}
    NeoHookean(double param1, double param2, double rho = 0.0)
      : HyperelasticMaterial<dim>(param2, rho), c1(param1)
    {
    }

    virtual dealii::SymmetricTensor<2, dim> get_tau_bar() const override
    {
      return 2 * this->c1 * this->b_bar;
    }

    virtual dealii::SymmetricTensor<4, dim> get_cc_bar() const override
    {
      return dealii::SymmetricTensor<4, dim>();
    }

  private:
    double c1;
  };
} // namespace Solid

#endif

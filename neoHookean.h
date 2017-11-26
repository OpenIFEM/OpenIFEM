#ifndef NEO_HOOKEAN
#define NEO_HOOKEAN

#include "hyperelasticMaterial.h"

namespace Solid
{
  extern template class HyperelasticMaterial<2>;
  extern template class HyperelasticMaterial<3>;

  /*! \brief Neo-Hookean material.
   *  This is written for Displacement-based formulation
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

    virtual dealii::SymmetricTensor<2, dim> getTauBar() const override
    {
      return 2 * this->c1 * this->bbar;
    }

    virtual dealii::SymmetricTensor<4, dim> getCcBar() const override
    {
      return dealii::SymmetricTensor<4, dim>();
    }

  private:
    double c1;
  };
}

#endif

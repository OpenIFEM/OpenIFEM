#include "hyperelasticMaterial.h"

namespace Solid
{
  using namespace dealii;

  template <int dim>
  void HyperelasticMaterial<dim>::update_data(const dealii::Tensor<2, dim> &F)
  {
    det_F = dealii::determinant(F);
    const dealii::Tensor<2, dim> FBar =
      Physics::Elasticity::Kinematics::F_iso(F);
    b_bar = dealii::Physics::Elasticity::Kinematics::b(FBar);
    Assert(det_F > 0, ExcInternalError());
  }

  template <int dim>
  dealii::SymmetricTensor<4, dim> HyperelasticMaterial<dim>::get_Jc_vol() const
  {
    double p = get_dPsi_vol_dJ();
    double p_tilde = p + det_F * get_d2Psi_vol_dJ2();
    return det_F * (p_tilde * ST::IxI - 2 * p * ST::S);
  }

  template <int dim>
  dealii::SymmetricTensor<4, dim> HyperelasticMaterial<dim>::get_Jc_iso() const
  {
    const dealii::SymmetricTensor<2, dim> tau_bar = get_tau_bar();
    const dealii::SymmetricTensor<2, dim> tau_iso = get_tau_iso();
    const dealii::SymmetricTensor<4, dim> tau_iso_x_I // tau_iso * I
      = dealii::outer_product(tau_iso, ST::I);
    const dealii::SymmetricTensor<4, dim> I_x_tau_iso // I * tau_iso
      = dealii::outer_product(ST::I, tau_iso);
    const dealii::SymmetricTensor<4, dim> ccBar =
      get_cc_bar(); // fictitious elasticity tensor
    return (2.0 / dim) * dealii::trace(tau_bar) * ST::dev_P -
           (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
           ST::dev_P * ccBar * ST::dev_P;
  }

  template class HyperelasticMaterial<2>;
  template class HyperelasticMaterial<3>;
} // namespace Solid

#include "hyperelasticMaterial.h"

namespace IFEM
{
  using namespace dealii;

  template <int dim>
  void HyperelasticMaterial<dim>::updateData(const dealii::Tensor<2, dim> &F)
  {
    detF = dealii::determinant(F);
    const dealii::Tensor<2, dim> FBar =
      Physics::Elasticity::Kinematics::F_iso(F);
    bbar = dealii::Physics::Elasticity::Kinematics::b(FBar);
    Assert(detF > 0, ExcInternalError());
  }

  template <int dim>
  dealii::SymmetricTensor<4, dim> HyperelasticMaterial<dim>::getJcVol() const
  {
    double p = get_dPsi_vol_dJ();
    double p_tilde = p + detF * get_d2Psi_vol_dJ2();
    return detF * (p_tilde * ST::IxI - 2 * p * ST::S);
  }

  template <int dim>
  dealii::SymmetricTensor<4, dim> HyperelasticMaterial<dim>::getJcIso() const
  {
    const dealii::SymmetricTensor<2, dim> tau_bar = getTauBar();
    const dealii::SymmetricTensor<2, dim> tau_iso = getTauIso();
    const dealii::SymmetricTensor<4, dim> tau_iso_x_I // tau_iso * I
      = dealii::outer_product(tau_iso, ST::I);
    const dealii::SymmetricTensor<4, dim> I_x_tau_iso // I * tau_iso
      = dealii::outer_product(ST::I, tau_iso);
    const dealii::SymmetricTensor<4, dim> ccBar =
      getCcBar(); // fictitious elasticity tensor
    return (2.0 / dim) * dealii::trace(tau_bar) * ST::dev_P -
           (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
           ST::dev_P * ccBar * ST::dev_P;
  }

  template class HyperelasticMaterial<2>;
  template class HyperelasticMaterial<3>;
}

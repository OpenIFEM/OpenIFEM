#include "pointHistory.h"

namespace IFEM
{
  template<int dim>
  void PointHistory<dim>::init(const Parameters::AllParameters& parameters)
  {
    if (parameters.type == "NeoHookean")
    {
      auto nh = std::dynamic_pointer_cast<NeoHookean<dim>>(this->material);
      Assert(nh, dealii::ExcInternalError());
      Assert(!parameters.C.empty(), dealii::ExcInternalError());
      nh.reset(new NeoHookean<dim>(parameters.C[0], parameters.rho));
      //FIXME: Is this necessary?
      nh->updateData(dealii::Tensor<2, dim>());
    }
    else
    {
      Assert(false, dealii::ExcNotImplemented());
    }
  }

  template<int dim>
  void PointHistory<dim>::update(const dealii::Tensor<2, dim>& Grad_u)
  {
    const dealii::Tensor<2, dim> F =
      dealii::Physics::Elasticity::Kinematics::F(Grad_u);
    this->material->updateData(F);
    this->FInv = dealii::invert(F);
    //FIXME: getTau and getJc are calling model specific functions
    // here we don't know the type of model, this is definitely bad.
    {
      auto nh = std::dynamic_pointer_cast<NeoHookean<dim>>(this->material);
      Assert(nh, dealii::ExcInternalError());
      this->tau = nh->getTau();
      this->Jc = nh->getJc();
    }
    this->dPsi_vol_dJ = this->material->get_dPsi_vol_dJ();
    this->d2Psi_vol_dJ2 = this->material->get_d2Psi_vol_dJ2();
  }

  template class PointHistory<2>;
  template class PointHistory<3>;
}

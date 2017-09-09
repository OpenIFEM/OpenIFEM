  template<int dim>
  PointHistory<dim>::(const Parameters::AllParameters &parameters) :
    FInv(I), tau(SymmetricTensor<2, dim>()), dPsi_vol_dJ(0.0), d2Psi_vol_dJ2(0.0),
    Jc(SymmetricTensor<4, dim>()) {}
  void PointHistory<dim>::init(const Parameters::AllParameters& parameters)
  {
    if (parameters.type == "NeoHookean")
    {
      auto nh = std::dynamic_pointer_cast<NeoHookean<dim>>(this->material);
      Assert(nh, ExcInternalError());
      Assert(!param.C.empty(), ExcInternalError());
      nh->reset(new NeoHookean<dim>(parameters.C[0], parameters.rho));
      updateData(Tensor<2, dim>());
    }
    else
    {
      Assert(false, ExcNotImplemented());
    }
  }


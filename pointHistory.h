#ifndef POINT_HISTORY
#define POINT_HISTORY

namespace IFEM
{
  template<int dim>
  class PointHistory
  {
  public:
    using dealii::Physics::Elasticity::StandardTensors<dim>::I;
    PointHistory() : FInv(I), tau(dealii::SymmetricTensor<2, dim>(),
      dPsi_vol_dJ(0.0), d2Psi_vol_dJ2(0.0), Jc(dealii::SymmetricTensor<4, dim>() {}
    virtual ~PointHistory() {}
    /** Initialize the members with the input parameters */
    void init(const Parameters::AllParameters&);
  private:
    std::shared_ptr<HyperelasticMaterial<dim>> material;
    dealii::Tensor<2, dim> FInv;
    dealii::SymmetricTensor<2, dim> tau;
    dealii::SymmetricTensor<4, dim> Jc;
    double dPsi_vol_dJ;
    double d2Psi_vol_dJ2;
  };
}

#endif

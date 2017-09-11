#ifndef POINT_HISTORY
#define POINT_HISTORY

#include <memory>
#include "hyperelasticMaterial.h"
#include "parameters.h"
#include "neoHookean.h"

namespace IFEM
{
  template<int dim>
  class PointHistory
  {
  using ST = dealii::Physics::Elasticity::StandardTensors<dim>;
  public:
    PointHistory() : FInv(ST::I), tau(dealii::SymmetricTensor<2, dim>()),
      Jc(dealii::SymmetricTensor<4, dim>()), dPsi_vol_dJ(0.0), d2Psi_vol_dJ2(0.0) {}
    virtual ~PointHistory() {}
    /** Initialize the members with the input parameters */
    void init(const Parameters::AllParameters&);
    /** 
     * Update the state with the displacement gradient
     * in the reference configuration. 
     */
    void update(const dealii::Tensor<2, dim>&);
    double getDetF() const {return this->material->getDetF();}
    const dealii::Tensor<2, dim>& getFInv() const {return this->FInv;}
    const dealii::SymmetricTensor<2, dim>& getTau() const {return this->tau;}
    const dealii::SymmetricTensor<4, dim>& getJc() const {return this->Jc;}
    double get_dPsi_vol_dJ() const {return this->dPsi_vol_dJ;}
    double get_d2Psi_vol_dJ2() const {return this->d2Psi_vol_dJ2;}
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

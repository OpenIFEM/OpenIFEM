#ifndef LINEAR_ELASTIC_MATERIAL
#define LINEAR_ELASTIC_MATERIAL

#include <deal.II/base/symmetric_tensor.h>

#include "material.h"

namespace IFEM
{
  /*! \breif Linear elastic material.
   *  It inherits density from the base class.
   */
  template<int dim>
  class LinearElasticMaterial : public Material<dim>
  {
  public:
    LinearElasticMaterial() : Material<dim>(),
      E(0.0), nu(0.0), lambda(0.0), mu(0.0) {}
    // Give density a default value since we don't always need it.
    LinearElasticMaterial(double, double, double rho = 0.0);
    dealii::SymmetricTensor<4, dim> getElasticityTensor() const;
  protected:
    double E; // Young's modulus
    double nu; // Poisson's ratio
    double lambda; // First lame parameter
    double mu; // Second lame parameter
  };
}

#endif

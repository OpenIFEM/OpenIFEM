#ifndef LINEAR_ELASTIC_MATERIAL
#define LINEAR_ELASTIC_MATERIAL

#include <deal.II/base/symmetric_tensor.h>

#include "material.h"

namespace Solid
{
  /*! \breif Linear elastic material.
   */
  template <int dim>
  class LinearElasticMaterial : public Material<dim>
  {
  public:
    LinearElasticMaterial()
      : Material<dim>(), E(0.0), nu(0.0), lambda(0.0), mu(0.0), eta(0.0)
    {
    }
    /**
     * Constructor using Young's modulus and Poisson's ratio.
     */
    LinearElasticMaterial(double, double, double, double);
    dealii::SymmetricTensor<4, dim> get_elasticity() const;
    dealii::SymmetricTensor<4, dim> get_viscosity() const;
    double get_lambda() const { return lambda; }
    double get_mu() const { return mu; }

  protected:
    double E;      //!< Young's modulus
    double nu;     //!< Poisson's ratio
    double lambda; //!< First lame parameter
    double mu;     //!< Second lame parameter
    double eta;    //!< Viscosity
  };
} // namespace Solid

#endif

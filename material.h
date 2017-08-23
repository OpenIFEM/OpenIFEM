#ifndef MATERIAL
#define MATERIAL

#include <stdexcept>

namespace IFEM
{
  using namespace dealii;
  using namespace std;

  template<int dim>
  class Material
  {
  public:
    /** Default constructor needed because there exists another constructor.*/
    Material() {}
    Material(double lame1, double lame2, double rho = 1.0) :
      lambda(lame1), mu(lame2), density(rho), initialized(true) {}
    virtual ~Material() {}

    double getLameFirst() const;
    double getLameSecond() const;
    double getShearModulus() const;
    double getYoungsModulus() const;
    double getBulkModulus() const;
    double getPoissonsRatio() const;
    double getDensity() const;

    virtual void print() const;
  protected:
    double lambda = 0.0; // Lame's first parameter
    double mu = 0.0; // Shear modulus
    double density = 0.0;
    bool initialized = false;
  };

#include "material.tpp"
}


#endif

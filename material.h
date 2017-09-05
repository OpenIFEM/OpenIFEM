#ifndef MATERIAL
#define MATERIAL

namespace IFEM
{
  /*! \brief The base class for all materials. */
  template<int dim>
  class Material
  {
  public:
    /** Default constructor needed because there exists another constructor.*/
    Material() : lambda(0.0), mu(0.0), density(0.0), initialized(false) {}
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
    double lambda; // Lame's first parameter
    double mu; // Shear modulus
    double density;
    bool initialized;
  };
}


#endif

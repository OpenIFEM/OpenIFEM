#ifndef MATERIAL
#define MATERIAL

namespace IFEM
{
  /*! \brief The base class for all materials.
   *  Apparently it is homogeneous for now.
   */
  template<int dim>
  class Material
  {
  public:
    /** Default constructor needed because there exists another constructor.
     *  Maybe should disable the default constructor instead.
     */
    Material() : lambda(0.0), mu(0.0), density(0.0) {}
    Material(double lame1, double lame2, double rho = 1.0) :
      lambda(lame1), mu(lame2), density(rho) {}
    virtual ~Material() {}

    double getLameFirst() const {return lambda;}
    double getLameSecond() const {return mu;}
    double getShearModulus() const {return getLameSecond();}
    double getYoungsModulus() const
    {
      return mu*(3*this->lambda + 2*this->mu)/(this->lambda + this->mu);
    }
    double getBulkModulus() const {return lambda + 2*mu/3;}
    double getPoissonsRatio() const {return lambda/(2*(lambda + mu));}
    double getDensity() const {return density;};
  protected:
    double lambda; // Lame's first parameter
    double mu; // Shear modulus
    double density;
  };

  template class Material<2>;
  template class Material<3>;
}


#endif

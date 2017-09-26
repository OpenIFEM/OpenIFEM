#ifndef MATERIAL
#define MATERIAL

namespace IFEM
{
  /*! \brief An abstract class for all materials.
   *  It has density as the only member.
   */
  template <int dim>
  class Material
  {
  public:
    Material() : density(0.0) {}
    Material(double rho) : density(rho) {}
    virtual ~Material() {}
    double getDensity() { return this->density; }

  protected:
    double density;
  };
}

#endif

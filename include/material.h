#ifndef MATERIAL
#define MATERIAL

namespace Solid
{
  /*! \brief An abstract class for all solid materials.
   */
  template <int dim>
  class Material
  {
  public:
    Material() : density(0.0) {}
    Material(double rho) : density(rho) {}
    virtual ~Material() {}
    double get_density() { return density; }

  protected:
    double density;
  };
} // namespace Solid

#endif

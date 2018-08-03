/**
 * 2D leaflet case with serial incompressible fluid solver and hyperelastic
 * solver.
 */
#include "fsi.h"
#include "hyper_elasticity.h"
#include "insim.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;
extern template class Solid::HyperElasticity<2>;
extern template class Solid::HyperElasticity<3>;
extern template class FSI<2>;
extern template class FSI<3>;

const double L = 4, H = 1, a = 0.1, b = 0.4, h = 0.05, U = 1.5;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues() : Function<dim>(dim + 1) {}
  virtual double value(const Point<dim> &p, const unsigned int component) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const
{
  if (component == 0 && std::abs(p[0]) < 1e-10 &&
      std::abs(p[1]) > 1e-10 && std::abs(p[1] - H) > 1e-10)
    {
      return U;
    }
  return 0;
}

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues::value(p, c);
}

int main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);

      if (params.dimension == 2)
        {
          Triangulation<2> fluid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(L / h),
             static_cast<unsigned int>(H / h)},
            Point<2>(0, 0),
            Point<2>(L, H),
            true);
          auto ptr = std::make_shared<BoundaryValues<2>>(BoundaryValues<2>());
          Fluid::InsIM<2> fluid(fluid_tria, params, ptr);

          Triangulation<2> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {static_cast<unsigned int>(a / h),
             static_cast<unsigned int>(b / h)},
            Point<2>(L / 4, 0),
            Point<2>(a + L / 4, b),
            true);
          Solid::HyperElasticity<2> solid(solid_tria, params);

          FSI<2> fsi(fluid, solid, params);
          fsi.run();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}

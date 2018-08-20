/**
 * This program tests serial NavierStokes solver with a 2D flow around cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * To save time, the global mesh refinement level is set to 1.
 * For real application, 2 should be used.
 * This test takes about 240s.
 */
#include "fsi.h"
#include "hyper_elasticity.h"
#include "insim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;
extern template class Solid::HyperElasticity<2>;
extern template class Solid::HyperElasticity<3>;
extern template class FSI<2>;
extern template class FSI<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

using namespace dealii;

double L = 1.0, R = 0.1, X = 0.6, Y = 0.5;

int main(int argc, char *argv[])
{
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
          GridGenerator::hyper_cube(fluid_tria, 0, L, true);
          Fluid::InsIM<2> fluid(fluid_tria, params);

          Triangulation<2> solid_tria;
          Point<2> center(X, Y);
          Utils::GridCreator<2>::sphere(solid_tria, center, R);
          Solid::HyperElasticity<2> solid(solid_tria, params);

          FSI<2> fsi(fluid, solid, params);
          fsi.run();
        }
      else
        {
          AssertThrow(false, ExcMessage("This test should be run in 2D!"));
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

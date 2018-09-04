/**
 * This program tests serial NavierStokes solver with a 2D lid-driven cavity
 * case at Re = 100.
 * This test takes about 476s.
 */
#include "insim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

using namespace dealii;

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
          Triangulation<2> tria;
          GridGenerator::hyper_cube(tria, 0, 1, true);
          Fluid::InsIM<2> flow(tria, params);
          flow.run();
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

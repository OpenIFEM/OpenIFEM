/**
 * This program tests serial LinearElasticSolver with a 2D ball dropping case.
 * In a free falling case, we know the velocity and displacement at a certain
 * time.
 */
#include "linearElasticSolver.h"
#include "parameters.h"
#include "utilities.h"

extern template class Solid::LinearElasticSolver<2>;
extern template class Solid::LinearElasticSolver<3>;

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
          Triangulation<2> solid_tria;
          Point<2> center(2, 1);
          double radius = 0.25;
          Utils::GridCreator::sphere(solid_tria, center, radius);
          Solid::LinearElasticSolver<2> solid(solid_tria, params);
          solid.run();

          auto u = solid.get_current_solution();
          double umin = *std::min_element(u.begin(), u.end());
          double uerror = std::abs(umin + 5.0) / 5.0;
          AssertThrow(uerror < 1e-3, ExcMessage("Incorrect min velocity!"));
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

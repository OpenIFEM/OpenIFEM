/**
 * This program tests serial linear elastic solver with a 2D bending beam case.
 * Constant traction is applied to the upper surface.
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

      double L = 8.0, H = 1.0;

      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {32, 4}, Point<2>(0, 0), Point<2>(L, H), true);
          Solid::LinearElasticSolver<2> solid(tria, params);
          solid.run();
          auto u = solid.get_current_solution();
          double umin = *std::min_element(u.begin(), u.end());
          double uerror = std::abs(umin + 0.1337) / 0.1337;
          AssertThrow(uerror < 1e-3,
                      ExcMessage("Minimum displacement is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {32, 4, 4}, Point<3>(0, 0, 0), Point<3>(L, H, H), true);
          Solid::LinearElasticSolver<3> solid(tria, params);
          solid.run();
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

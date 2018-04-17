/**
 * This program tests beam bending problem with Mooney-Rivlin model.
 * Constant traction is applied to the upper surface.
 * This test takes 4 sec for 2D and 200 sec for 3D.
 */
#include "hyperelasticSolver.h"
#include "parameters.h"
#include "utilities.h"

extern template class Solid::HyperelasticSolver<2>;
extern template class Solid::HyperelasticSolver<3>;

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
      double L = 10.0, H = 1.0;
      Vector<double> u;
      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{40, 4},
            Point<2>(0, 0),
            Point<2>(L, H),
            true);
          Solid::HyperelasticSolver<2> solid(tria, params);
          solid.run();
          u = solid.get_current_solution();
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{40, 4, 4},
            Point<3>(0, 0, 0),
            Point<3>(L, H, H),
            true);
          Solid::HyperelasticSolver<3> solid(tria, params);
          solid.run();
          u = solid.get_current_solution();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
      double umin = *std::min_element(u.begin(), u.end());
      double umax = *std::max_element(u.begin(), u.end());
      double umin_expected = (params.dimension == 2 ? -0.0616287 : -0.0618016);
      double umax_expected = (params.dimension == 2 ? 0.00867069 : 0.00765623);
      double uerror = std::abs((umin - umin_expected) / umin_expected);
      AssertThrow(uerror < 1e-3,
                  ExcMessage("Minimum displacemet is incorrect"));
      uerror = std::abs((umax - umax_expected) / umax_expected);
      AssertThrow(uerror < 1e-3,
                  ExcMessage("Maximum displacemet is incorrect"));
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

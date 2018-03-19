/**
 * This program tests serial NavierStokes solver with a 2D pipe flow case.
 * There is no velocity input. With a constant gravity, we expect to see
 * a pressure difference.
 * 2D test takes about 7s.
 */
#include "navierstokes.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::NavierStokes<2>;
extern template class Fluid::NavierStokes<3>;

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

      double L = 2.0, D = 0.2;

      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {100, 5}, Point<2>(0, 0), Point<2>(L, D / 2), true);
          Fluid::NavierStokes<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // The pressure difference between the max and min should be rho * g *
          // L
          auto p = solution.block(1);
          double pmax = *std::max_element(p.begin(), p.end());
          double pmin = *std::min_element(p.begin(), p.end());
          double pdiff = pmax - pmin;
          double perror = std::abs(pdiff - 20) / 20;
          AssertThrow(perror < 1e-3,
                      ExcMessage("Pressure difference is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          dealii::GridGenerator::cylinder(tria, D / 2, L / 2);
          static const CylindricalManifold<3> cylinder;
          tria.set_manifold(0, cylinder);
          Fluid::NavierStokes<3> flow(tria, params);
          flow.run();
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

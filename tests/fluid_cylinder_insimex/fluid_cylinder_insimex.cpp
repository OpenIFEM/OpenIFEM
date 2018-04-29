/**
 * This program tests serial InsIMEX solver with a 2D flow around cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * To save time, the global mesh refinement level is set to 1.
 * For real application, 2 should be used.
 * This test takes about 72.
 */
#include "insimex.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::InsIMEX<2>;
extern template class Fluid::InsIMEX<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

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
          Triangulation<2> tria;
          Utils::GridCreator<2>::flow_around_cylinder(tria);
          Fluid::InsIMEX<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // Check the max values of velocity and pressure
          auto v = solution.block(0), p = solution.block(1);
          double vmax = *std::max_element(v.begin(), v.end());
          double pmax = *std::max_element(p.begin(), p.end());
          double verror = std::abs(vmax - 0.402021) / 0.402021;
          double perror = std::abs(pmax - 0.147245) / 0.147245;
          AssertThrow(verror < 1e-3 && perror < 1e-3,
                      ExcMessage("Maximum velocity or pressure is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          Utils::GridCreator<3>::flow_around_cylinder(tria);
          Fluid::InsIMEX<3> flow(tria, params);
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

/**
 * This program tests serial NavierStokes solver with a 2D pressure-driven pipe
 * flow case.
 * The final axial velocity profile should be parabolic, we should check
 * the maximum velocity.
 * 2D test takes about 26.9s.
 */
#include "insim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;

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
            tria, {100, 10}, Point<2>(0, 0), Point<2>(L, D), true);
          Fluid::InsIM<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // Assuming the mass is conserved and final velocity profile is
          // parabolic,
          // vmax should equal 1.5 times inlet velocity.
          auto v = solution.block(0);
          double vmax = *std::max_element(v.begin(), v.end());
          double verror = std::abs(vmax - 2.5e-2) / 2.5e-2;
          AssertThrow(verror < 1e-3,
                      ExcMessage("Maximum velocity is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          dealii::GridGenerator::cylinder(tria, D / 2, L / 2);
          static const CylindricalManifold<3> cylinder;
          tria.set_manifold(0, cylinder);
          Fluid::InsIM<3> flow(tria, params);
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

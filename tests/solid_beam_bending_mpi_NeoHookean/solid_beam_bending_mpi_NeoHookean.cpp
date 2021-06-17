/**
 * This program tests beam bending problem with Neo-Hookean model.
 * Constant traction is applied to the upper surface.
 * This test takes 2 sec for 2D and 100 sec for 3D.
 */
#include "mpi_hyper_elasticity.h"
#include "parameters.h"
#include "utilities.h"

extern template class Solid::MPI::HyperElasticity<2>;
extern template class Solid::MPI::HyperElasticity<3>;

int main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);
      double L = 10.0, H = 1.0;
      PETScWrappers::MPI::Vector u;
      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{40, 4},
            Point<2>(0, 0),
            Point<2>(L, H),
            true);
          Solid::MPI::HyperElasticity<2> solid(tria, params);
          solid.run();
          u = solid.get_current_solution();
        }
      else if (params.dimension == 3)
        {
          parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{40, 4, 4},
            Point<3>(0, 0, 0),
            Point<3>(L, H, H),
            true);
          Solid::MPI::HyperElasticity<3> solid(tria, params);
          solid.run();
          u = solid.get_current_solution();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
      double umin = Utils::PETScVectorMin(u);
      double umax = Utils::PETScVectorMax(u);
      double umin_expected = (params.dimension == 2 ? -0.0616287 : -0.0617214);
      double umax_expected = (params.dimension == 2 ? 0.00867069 : 0.00867507);
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

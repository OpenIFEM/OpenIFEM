#include "mpi_fsi.h"
#include "mpi_scnsim.h"
#include "mpi_shared_linear_elasticity.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Solid::MPI::SharedLinearElasticity<2>;
extern template class MPI::FSI<2>;

using namespace dealii;

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

      if (params.dimension == 2)
        {
          // Create fluid mesh
          parallel::distributed::Triangulation<2> tria_fluid(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(
            tria_fluid, {50, 25}, Point<2>(0, 0), Point<2>(2.0, 1.0), true);

          // Create solid mesh
          Triangulation<2> tria_solid;
          GridGenerator::subdivided_hyper_rectangle(
            tria_solid, {10, 11}, Point<2>(0, 0), Point<2>(1.0, 1.02), true);

          // Translate solid mesh
          Tensor<1, 2> offset({0.25, 0});
          GridTools::shift(offset, tria_solid);

          Fluid::MPI::SCnsIM<2> fluid(tria_fluid, params);
          Solid::MPI::SharedLinearElasticity<2> solid(tria_solid, params);

          auto penetration_criterion = [](const Point<2> &p) -> double {
            double wall_height = 1.0;
            return (p[1] - wall_height);
          };

          MPI::FSI<2> fsi(fluid, solid, params);
          fsi.set_penetration_criterion(penetration_criterion,
                                        Tensor<1, 2>({0, -1}));
          fsi.run();
          Vector<double> u(solid.get_current_solution());
          double umin = *std::min_element(u.begin(), u.end());
          double uerror = std::abs(umin + 0.01999) / 0.01999;
          AssertThrow(uerror < 1e-3,
                      ExcMessage("Minimum displacement is incorrect!"));
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

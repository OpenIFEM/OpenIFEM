/**
 * 2D leaflet case with serial incompressible fluid solver and hyperelastic
 * solver.
 */
#include "mpi_fsi.h"
#include "mpi_scnsim.h"
#include "mpi_shared_hyper_elasticity.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Fluid::MPI::SCnsIM<3>;
extern template class Solid::MPI::SharedHyperElasticity<2>;
extern template class Solid::MPI::SharedHyperElasticity<3>;
extern template class MPI::FSI<2>;
extern template class MPI::FSI<3>;

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

      double L = 4, H = 1, a = 0.1, b = 0.4, h = 0.05;

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> fluid_tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(L / h),
             static_cast<unsigned int>(H / h)},
            Point<2>(0, 0),
            Point<2>(L, H),
            true);
          Fluid::MPI::SCnsIM<2> fluid(fluid_tria, params);

          Triangulation<2> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {static_cast<unsigned int>(a / h),
             static_cast<unsigned int>(b / h)},
            Point<2>(L / 4, 0),
            Point<2>(a + L / 4, b),
            true);
          Solid::MPI::SharedHyperElasticity<2> solid(solid_tria, params);

          MPI::FSI<2> fsi(fluid, solid, params);
          fsi.run();
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

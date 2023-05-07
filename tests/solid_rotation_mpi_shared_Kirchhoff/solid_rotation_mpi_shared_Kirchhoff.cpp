// This program verifies that Kirchhoff solid solver can handle large solid rotation.
// This test should be ran for all future nonlinear solid solvers.
#include "mpi_shared_hyper_elasticity.h"
#include "parameters.h"
#include "utilities.h"

extern template class Solid::MPI::SharedHyperElasticity<2>;

using namespace dealii;

int main(int argc, char *argv[]) {
  using namespace dealii;
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::string infile("parameters.prm");
    if (argc > 1) {
      infile = argv[1];
    }
    Parameters::AllParameters params(infile);

    if (params.dimension == 2) {
      Triangulation<2> solid_tria;
      GridGenerator::subdivided_hyper_rectangle(
          solid_tria, {2, 2}, Point<2>(0.0, 0.0), Point<2>(1, 1), true);

      Solid::MPI::SharedHyperElasticity<2> solid(solid_tria, params);
      solid.run();
    } else {
      AssertThrow(false, ExcMessage("This test should be run in 2D!"));
    }
  } catch (std::exception &exc) {
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
  } catch (...) {
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

/**
 * This program tests parallel NavierStokes solver with a 2D flow around
 * cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * Only one step is run, and the test takes about 33s.
 */
#include "mpi_scnsim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Fluid::MPI::SCnsIM<3>;

using namespace dealii;

int main(int argc, char *argv[])
{
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
          auto initial_condition = [](const Point<2> &point,
                                      const unsigned int component) -> double {
            double pressure = 1e4;
            if (component == 2)
              {
                if (point[0] > 4.0 && point[0] < 5.0)
                  {
                    return pressure * (point[0] - 4.0);
                  }
                else if (point[0] >= 5.0 && point[0] < 12.0)
                  {
                    return pressure;
                  }
              }
            return 0.0;
          };
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(
            tria, {150, 20}, Point<2>(0, 0), Point<2>(15, 2), true);
          Fluid::MPI::SCnsIM<2> flow(tria, params);
          flow.set_initial_condition(initial_condition);
          flow.run();
          // Check the max values of pressure
          auto solution = flow.get_current_solution();
          auto p = solution.block(1);
          double pmax = p.max();
          double perror = std::abs(pmax - 1e4) / 1e4;
          AssertThrow(perror < 1e-8,
                      ExcMessage("Maximum pressure is incorrect!"));
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

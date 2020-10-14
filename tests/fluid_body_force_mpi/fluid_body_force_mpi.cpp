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
          auto body_force = [](const Point<2> &point,
                               const unsigned int component) -> double {
            double rho = 1.3e-3;
            double bf = 1.0e3 / rho;
            if (point[0] > 3.5 - 5e-4 && point[0] < 4.5 + 5e-4 &&
                component == 0)
              {
                return bf;
              }
            return 0.0;
          };

          auto sigma_pml = [](const Point<2> &point,
                              const unsigned int component) -> double {
            (void)component;
            double sigmaMax = 340000;
            double pmlLength = 3.0;
            double sigmaPML = 0.0;
            std::vector<double> boundary = {0.0, 8.0};
            std::vector<unsigned int> boundary_dir = {0, 0};
            for (unsigned int i = 0; i < boundary.size(); ++i)
              {
                if (std::abs(point[boundary_dir[i]] - boundary[i]) < pmlLength)
                  {
                    sigmaPML =
                      sigmaMax *
                      pow((pmlLength -
                           std::abs(point[boundary_dir[i]] - boundary[i])) /
                            pmlLength,
                          4);
                  }
              }
            return sigmaPML;
          };

          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          GridGenerator::subdivided_hyper_rectangle(
            tria, {160, 30}, Point<2>(0, 0), Point<2>(8, 2), true);
          Fluid::MPI::SCnsIM<2> flow(tria, params);
          flow.set_body_force(body_force);
          flow.set_sigma_pml_field(sigma_pml);
          flow.run();
          // Check the max values of pressure
          auto solution = flow.get_current_solution();
          auto p = solution.block(1);
          double pmax = p.max();
          double pmin = p.min();
          double perror = std::abs((pmax - pmin) - 1e3) / 1e3;
          AssertThrow(perror < 1e-3,
                      ExcMessage("Pressure difference is incorrect!"));
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

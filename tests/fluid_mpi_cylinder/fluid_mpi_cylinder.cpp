/**
 * This program tests parallel NavierStokes solver with a 2D flow around cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * To save time, the global mesh refinement level is set to 1.
 * For real application, 2 should be used.
 * This test takes about 240s.
 */
#include "mpi_navierstokes.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::ParallelNavierStokes<2>;
extern template class Fluid::ParallelNavierStokes<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

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
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          Utils::GridCreator<2>::flow_around_cylinder(tria);
          Fluid::ParallelNavierStokes<2> flow(tria, params);
          flow.run();
          // Check the max values of velocity and pressure
          /*
          auto solution = flow.get_current_solution();
          auto v = solution.block(0), p = solution.block(1);
          double vmax = *std::max_element(v.begin(), v.end());
          double pmax = *std::max_element(p.begin(), p.end());
          double verror = std::abs(vmax - 0.4008695) / 0.4008695;
          double perror = std::abs(pmax - 0.1473135) / 0.1473135;
          AssertThrow(verror < 1e-3 && perror < 1e-3,
                      ExcMessage("Maximum velocity or pressure is incorrect!"));
                      */
        }
      else if (params.dimension == 3)
        {
          parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
          Utils::GridCreator<3>::flow_around_cylinder(tria);
          Fluid::ParallelNavierStokes<3> flow(tria, params);
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

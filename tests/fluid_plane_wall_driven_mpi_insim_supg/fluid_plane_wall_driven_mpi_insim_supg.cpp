/*
This program tests the icompressible instance of the MPI SUPG fluid solver; it
uses the assembly portion of the solver as implemented in mpi_insim_supg.cpp
file and the other components of the solver as implemented in
mpi_supg_solver.cpp file. It is a pure fluid 2D test case at Re=200
The final horizontal velocity profile is linear with a positive gradient.
The test runs for about 1.86s.
*/
#include "mpi_insim_supg.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SUPGInsIM<2>;
extern template class Fluid::MPI::SUPGInsIM<3>;

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

      double L = 2, D = 0.4, d = 0.025, l = 0.1;
      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria,
            {static_cast<unsigned int>(L / l),
             static_cast<unsigned int>(D / d)},
            Point<2>(0, 0),
            Point<2>(L, D),
            true);
          Fluid::MPI::SUPGInsIM<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // Check the l2_norm of the velocity.
          auto v = solution.block(0);
          double l2_norm = v.l2_norm();
          double verror = std::abs(l2_norm - 4.7112) / 4.7112;
          AssertThrow(verror < 1e-3,
                      ExcMessage("The l2 norm of velocity is incorrect!"));
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

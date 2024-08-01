/*
This program tests the icompressible instance of the MPI SUPG fluid solver; it
uses the assembly portion of the solver as implemented in mpi_insim_supg.cpp
file and the other components of the solver as implemented in
mpi_supg_solver.cpp file. It is a pure fluid 2D test case at Re=10
The final horizontal velocity profile is parabolic.
The test runs for 3.21s.
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

      double L = 2, D = 0.2;
      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {100, 10}, Point<2>(0, 0), Point<2>(L, D), true);
          Fluid::MPI::SUPGInsIM<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          // Maximum velocity is given by P * D^2 / (16 \mu L)
          auto v = solution.block(0);
          Vector<double> serialized_v(v);
          // Boundary velocities are affected by the numerical stabilization so
          // they are larger than analytic solution. Take them out by using the
          // 40th largest value.
          std::sort(
            serialized_v.begin(), serialized_v.end(), std::greater<double>());
          double vmax_30th = serialized_v[29];
          double vmax = serialized_v[0];
          // Check twice. The total max is not exceeding 2% and 30th max is not
          // exceeding 0.01%.
          double verror_max = std::abs(vmax - 2.5e-2) / 2.5e-2;
          AssertThrow(verror_max < 2e-2,
                      ExcMessage("Maximum velocity is incorrect!"));
          double verror_30thmax = std::abs(vmax_30th - 2.5e-2) / 2.5e-2;
          AssertThrow(verror_30thmax < 1e-3,
                      ExcMessage("Maximum velocity is incorrect!"));
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

/**
 * This program tests serial Slightly Compressible solver with an
 * acoustic wave in 2D duct case.
 * A Gaussian pulse is used as the time dependent BC with max velocity
 * equal to 6cm/s.
 * This test takes about 770s.
 */
#include "mpi_scnsex.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SCnsEX<2>;
extern template class Fluid::MPI::SCnsEX<3>;

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

      double L = 4, H = 1;

      auto gaussian_pulse = [dt =
                               params.time_step](const Point<2> &p,
                                                 const unsigned int component,
                                                 const double time) -> double {
        auto time_value = [](double t) {
          return 6.0 * exp(-0.5 * pow((t - 0.5e-4) / 0.15e-4, 2));
        };

        if (component == 0 && std::abs(p[0]) < 1e-10)
          return time_value(time);

        return 0;
      };

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {8, 2}, Point<2>(0, 0), Point<2>(L, H), true);
          Fluid::MPI::SCnsEX<2> flow(tria, params);
          flow.add_hard_coded_boundary_condition(0, gaussian_pulse);
          flow.set_hard_coded_boundary_condition_time(0, 1.1e-4);
          flow.run();
          // After the computation the max velocity should be ~
          // the peak of the Gaussian pulse (with dispersion).
          auto solution = flow.get_current_solution();
          auto v = solution.block(0);
          double vmax = v.max();
          double verror = std::abs(vmax - 5.97) / 5.97;
          AssertThrow(verror < 1e-3,
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

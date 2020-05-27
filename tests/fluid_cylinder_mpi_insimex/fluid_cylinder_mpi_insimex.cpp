/**
 * This program tests InsIMEX solver with a 2D flow around
 * cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * Only one step is run, and the test takes about 33s.
 */
#include "mpi_insimex.h"

extern template class Fluid::MPI::InsIMEX<2>;
extern template class Fluid::MPI::InsIMEX<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

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

      auto inflow_bc = [dim = params.dimension](const Point<2> &p,
                                                const unsigned int component,
                                                const double time) -> double {
        (void)time;
        double left_boundary = (dim == 2 ? 0.0 : -0.3);
        unsigned int flow_component = (dim == 2 ? 0 : 2);
        if (component == flow_component &&
            std::abs(p[flow_component] - left_boundary) < 1e-10)
          {
            // For a parabolic velocity profile, Uavg = 2/3 * Umax in
            // 2D, and 4/9 * Umax in 3D. If nu = 0.001, D = 0.1, then Re
            // = 100 * Uavg
            double Uavg = 0.2;
            double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
            double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
            if (dim == 3)
              {
                value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
              }
            return value;
          }
        return 0.0;
      };

      auto inflow_bc_3d = [dim =
                             params.dimension](const Point<3> &p,
                                               const unsigned int component,
                                               const double time) -> double {
        (void)time;
        double left_boundary = (dim == 2 ? 0.0 : -0.3);
        unsigned int flow_component = (dim == 2 ? 0 : 2);
        if (component == flow_component &&
            std::abs(p[flow_component] - left_boundary) < 1e-10)
          {
            // For a parabolic velocity profile, Uavg = 2/3 * Umax in
            // 2D, and 4/9 * Umax in 3D. If nu = 0.001, D = 0.1, then Re
            // = 100 * Uavg
            double Uavg = 0.2;
            double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
            double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
            if (dim == 3)
              {
                value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
              }
            return value;
          }
        return 0.0;
      };

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          Utils::GridCreator<2>::flow_around_cylinder(tria);
          Fluid::MPI::InsIMEX<2> flow(tria, params);
          flow.add_hard_coded_boundary_condition(0, inflow_bc);
          flow.run();
          // Check the max values of velocity and pressure
          auto solution = flow.get_current_solution();
          auto v = solution.block(0), p = solution.block(1);
          double vmax = v.max();
          double pmax = p.max();
          double verror = std::abs(vmax - 0.374062) / 0.374062;
          double perror = std::abs(pmax - 46.5308) / 46.5308;
          AssertThrow(verror < 1e-3 && perror < 1e-3,
                      ExcMessage("Maximum velocity or pressure is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
          Utils::GridCreator<3>::flow_around_cylinder(tria);
          Fluid::MPI::InsIMEX<3> flow(tria, params);
          flow.add_hard_coded_boundary_condition(4, inflow_bc_3d);
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

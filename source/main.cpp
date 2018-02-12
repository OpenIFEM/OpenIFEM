#include "fsi.h"
#include "hyperelasticSolver.h"
#include "linearElasticSolver.h"
#include "mpi_linearelasticity.h"
#include "mpi_navierstokes.h"
#include "navierstokes.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::NavierStokes<2>;
extern template class Fluid::NavierStokes<3>;
extern template class Solid::LinearElasticSolver<2>;
extern template class Solid::LinearElasticSolver<3>;
extern template class Solid::HyperelasticSolver<2>;
extern template class Solid::HyperelasticSolver<3>;

extern template class Solid::ParallelLinearElasticity<2>;
extern template class Solid::ParallelLinearElasticity<3>;
extern template class Fluid::ParallelNavierStokes<2>;
extern template class Fluid::ParallelNavierStokes<3>;

extern template class FSI<2>;
extern template class FSI<3>;

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
          Triangulation<2> fluid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria, {32, 8}, Point<2>(0, 0), Point<2>(8e-2, 2e-2), true);
          Fluid::NavierStokes<2> fluid(fluid_tria, params);

          Triangulation<2> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {4, 8},
            // Point<2>(3.65e-2, 0),
            // Point<2>(4.15e-2, 1e-2),
            Point<2>(3.75e-2, 0),
            Point<2>(4.25e-2, 1e-2),
            true);
          Solid::LinearElasticSolver<2> solid(solid_tria, params);

          FSI<2> fsi(fluid, solid, params);
          fsi.run();
        }
      else if (params.dimension == 3)
        {
          parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
          Utils::GridCreator::flow_around_cylinder(tria);
          Fluid::ParallelNavierStokes<3> flow(tria, params);
          flow.run();
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

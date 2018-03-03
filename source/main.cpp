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
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          Utils::GridCreator::flow_around_cylinder(tria);
          Fluid::ParallelNavierStokes<2> flow(tria, params);
          flow.run();
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

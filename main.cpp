#include "hyperelasticSolver.h"
#include "linearElasticSolver.h"
#include "mpi_linearelasticity.h"
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

int main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      Parameters::AllParameters params("parameters.prm");
      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          Utils::GridCreator::flow_around_cylinder(tria);
          Fluid::NavierStokes<2> flow(tria, params);
          flow.run();
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          Utils::GridCreator::flow_around_cylinder(tria);
          Fluid::NavierStokes<3> flow(tria, params);
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

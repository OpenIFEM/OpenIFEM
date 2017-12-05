#include "hyperelasticSolver.h"
#include "linearElasticSolver.h"
#include "navierstokes.h"
#include "parameters.h"
#include "utilities.h"
#include "mpi_linearelasticity.h"

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
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD,
            typename Triangulation<2>::MeshSmoothing
              (Triangulation<2>::smoothing_on_refinement |
               Triangulation<2>::smoothing_on_coarsening));
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{16, 2},
            Point<2>(0, 0),
            Point<2>(8, 1),
            true);
          Solid::ParallelLinearElasticity<2> solid(tria, params);
          solid.run();
        }
      else if (params.dimension == 3)
        {
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

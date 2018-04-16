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
       std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);
      
      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          GridGenerator::subdivided_hyper_rectangle(
            tria,
            std::vector<unsigned int>{20, 4},
            Point<2>(0, 0),
            Point<2>(5, 1),
            true);
          Solid::HyperelasticSolver<2> solid(tria, params);
          solid.run();
        }
      else if (params.dimension == 3)
        {
	  Triangulation<3> tria;
          GridGenerator::subdivided_hyper_rectangle(
          tria,
          std::vector<unsigned int>{40, 4, 4},
          Point<3>(0, 0, 0),
          Point<3>(10, 1, 1),
          true);
          Solid::HyperelasticSolver<3> solid(tria, params);
          solid.run();
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

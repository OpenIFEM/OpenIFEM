#include <iostream>

#include "hyperelasticSolver.h"
#include "linearElasticSolver.h"
#include "navierstokes.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::NavierStokes<2>;
extern template class Fluid::NavierStokes<3>;
extern template class Solid::LinearElasticSolver<2>;
extern template class Solid::LinearElasticSolver<3>;
extern template class Solid::HyperelasticSolver<2>;
extern template class Solid::HyperelasticSolver<3>;

int main()
{
  using namespace dealii;

  try
    {
      Parameters::AllParameters params("parameters.prm");
      if (params.dimension == 2)
        {
          Triangulation<2> triangulation;
          /*
          GridGenerator::subdivided_hyper_rectangle(
            triangulation,
            std::vector<unsigned int>{16, 2},
            Point<2>(0, 0),
            Point<2>(8, 1),
            true);
          Solid::LinearElasticSolver<2> solid(triangulation, params);
          // Solid::HyperelasticSolver<2> solid(triangulation, params);
          solid.run();
          */

          // Utils::GridCreator::flow_around_cylinder(triangulation);
          GridGenerator::subdivided_hyper_rectangle(
            triangulation,
            std::vector<unsigned int>({16U, 4U}),
            Point<2>(),
            Point<2>(8.0, 2.0),
            true);
          Fluid::NavierStokes<2> flow(triangulation, params);
          flow.run();
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> triangulation;
          GridGenerator::subdivided_hyper_rectangle(
            triangulation,
            std::vector<unsigned int>{16, 2, 2},
            Point<3>(0, 0, 0),
            Point<3>(8, 1, 1),
            true);
          Solid::LinearElasticSolver<3> solid(triangulation, params);
          // Solid::HyperelasticSolver<3> solid(triangulation, params);
          solid.run();

          /*
          Utils::GridCreator::flow_around_cylinder(triangulation);
          Fluid::NavierStokes<3> flow(triangulation, params);
          flow.run();
          */
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

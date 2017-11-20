#include <iostream>

#include "parameters.h"
#include "utilities.h"
#include "navierstokes.h"

extern template class Fluid::NavierStokes<2>;
extern template class Fluid::NavierStokes<3>;

int main()
{
  try
    {
      Parameters::AllParameters params("parameters.prm");
      if (params.dimension == 2)
        {
          dealii::Triangulation<2> triangulation;
          Utils::GridCreator::flow_around_cylinder(triangulation);
          Fluid::NavierStokes<2> flow(triangulation, params);
          flow.run();
        }
      else if (params.dimension == 3)
        {
          dealii::Triangulation<3> triangulation;
          Utils::GridCreator::flow_around_cylinder(triangulation);
          Fluid::NavierStokes<3> flow(triangulation, params);
          flow.run();
        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
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

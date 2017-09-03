#include "linearElasticSolver.h"
#include "linearMaterial.h"

#include <iostream>

int main(int argc, char* argv[])
{
  try
  {
    using namespace dealii;
    using namespace IFEM;

    LinearElasticSolver<2> solver;
    if (argc > 1)
    {
      //solver.runStatics(string(argv[1]));
      solver.runDynamics(string(argv[1]));
    }
    else
    {
      solver.runStatics();
    }
  }
  catch (std::exception& exc)
  {
    std::cerr << std::endl << std::endl
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
    std::cerr << std::endl << std::endl
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

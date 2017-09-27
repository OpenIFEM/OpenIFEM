#include <iostream>

#include "hyperelasticSolver.h"
#include "linearElasticSolver.h"

extern template class IFEM::LinearElasticSolver<2>;
extern template class IFEM::LinearElasticSolver<3>;
extern template class IFEM::HyperelasticSolver<2>;
extern template class IFEM::HyperelasticSolver<3>;

int main()
{
  try
    {
      IFEM::Parameters::AllParameters params("parameters.prm");
      if (params.dimension == 2)
        {
          IFEM::HyperelasticSolver<2> solver(params);
          solver.runStatics();
        }
      else if (params.dimension == 3)
        {
          IFEM::HyperelasticSolver<3> solver(params);
          solver.runStatics();
        }
      else
        {
          Assert(false, dealii::ExcNotImplemented());
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

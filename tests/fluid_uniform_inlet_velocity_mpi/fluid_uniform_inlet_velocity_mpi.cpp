/*
This program tests the incompressible solver with stabilization. It is a pure fluid 2D test case 
with a rectangular channel of height 0.4cm and length 2.0cm as the domain. The purpose of this program is to do
a mesh convergence study. To achieve this, four different meshes were used: 500, 2000,8000 and 32000 elements. 
With the exact solution known, the L2-norm and L-infinity errors for pressure , velocity in x-direction and velocity in 
y-direction were computed and convergence plots generated from the error estimates. Taking into consideration entrance effect,
the error estimation was done for the poertion of the domain starting from the centre (i.e x= 1cm) to the exit of the domain 
(i.e. x=2.0).
It takes approximately 16s of runtime to get steady state solution. 

All physical quatities are in the 'cgs' unit system. The density of the fluid was chosen as 1 g/cm^3 and the viscosity
was chosen as 0.001 dyne-s/cm^2
*/




#include "mpi_scnsim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Fluid::MPI::SCnsIM<3>;
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
        
        double L=2.0 ,D= 0.4 ;

        if (params.dimension == 2)


          {

            parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
            dealii::GridGenerator::subdivided_hyper_rectangle(
             tria, {50,10}, Point<2>(0, 0), Point<2>(L, D), true);
             Fluid::MPI::SCnsIM<2> flow(tria, params);
            flow.run();
            auto solution = flow.get_current_solution();
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



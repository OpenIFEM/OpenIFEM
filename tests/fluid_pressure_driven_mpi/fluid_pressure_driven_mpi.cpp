/*
This program tests the incompressible solve with stabilization. It is a pure flue 2D test case 
with a rectangular channel of height 0.1cm and length 2.0cm as the domain. The program was tested at 3 different Reynold's 
number (i.e. 10,50 and 100). It takes approximately 16s of runtime to get steady state solution; however, with an initial 
condition steady state solution can be obtained by 8s. 

An initial conditon can be applied as done in  'tests/fluid_initial_condition_mpi.cpp'. More information on how to apply
initial condition can be obtained from mpi_fluid_solver.h:121.

The horizontal velocity at the exit of the fluid domain was compared with the analytical solution 
for all three differnt Reynold's number(Re); the maximum percentage error between the two solutions came out as 5.4% and 
that was for the Re=100 case.

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
        
        double L=2.0 ,D= 0.1 ;

        if (params.dimension == 2)
          {
             
          


            parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
            dealii::GridGenerator::subdivided_hyper_rectangle(
              tria, {160,8}, Point<2>(0, 0), Point<2>(L, D), true);
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




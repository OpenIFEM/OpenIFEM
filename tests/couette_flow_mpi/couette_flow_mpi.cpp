/*
This program tests the incompressible solver with stabilization. It is a pure fluid 2D test case 
with a rectangular channel of height 0.4cm and length 2.2cm as the domain. The program was tested at 3 different Reynold's 
number (i.e. 40,100 and 200).

It takes approximately 40s of runtime to get steady state solution. The horizontal velocity at the exit of the fluid domain 
was compared with the analytical solution for all three differnt Reynold's number(Re); the maximum percentage error between
the two solutions came out as 3.3% and that was for the Re=200 case.

All physical quatities are in the 'cgs' unit system. The density of the fluid was chosen as 1 g/cm^3 and the viscosity
was chosen as 0.002 dyne-s/cm^2
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

      double L = 2.2, D = 0.4 , d=0.005 , l=0.02 ;
    if (params.dimension == 2)
        {
          
          
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria,
             {static_cast<unsigned int> (L/l),
                static_cast<unsigned int> (D/d)},
            Point<2>(0, 0),
            Point<2>(L, D ),
            true);
          Fluid::MPI::SCnsIM<2> flow(tria, params);
          flow.run();
          auto solution = flow.get_current_solution();
          
        }
      
   else
        {
          AssertThrow(false, ExcMessage("This test should be run in 2D!"));
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

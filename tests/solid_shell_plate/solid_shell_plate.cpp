#include "parameters.h"
#include "shell_solid_solver.h"
#include "utilities.h"
#include <deal.II/grid/grid_in.h>

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
      std::string meshfile("solid.gmsh");
      if (argc > 2)
        {
          meshfile = argv[2];
        }
      std::string forcefile("solid_f");
      if (argc > 3)
        {
          forcefile = argv[3];
        }

      Parameters::AllParameters params(infile);
      if (params.dimension == 3)
        {
          libMesh::LibMeshInit libmesh_init(argc, argv);
          Triangulation<2, 3> tria;
          GridIn<2, 3> gridin;
          gridin.attach_triangulation(tria);
          std::ifstream input_solid(meshfile);
          gridin.read_msh(input_solid);
          Solid::ShellSolidSolver solid(tria, params, &libmesh_init);
          std::cout << "Reading forcefile: " << forcefile << std::endl;
          solid.get_forcing_file(forcefile);
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
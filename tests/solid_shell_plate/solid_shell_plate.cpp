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

  // ShellSolid::shellparam param;
  // // read command-line arguments and initialize global variables
  // if (param.read_parameters(argc, argv))
  //   {
  //     std::cout << "Read command-line arguments.......OK" << std::endl;
  //   }
  // else
  //   {
  //     std::cout << "Read command-line arguments.......FAILED" << std::endl;
  //     return -1;
  //   }
  // // Hard coded parameters for test E
  // param.nu = 0.3;
  // param.em = 10000;
  // param.thickness = 0.25;
  // param.debug = false;

  // Initialize libMesh and any dependent library
  // libMesh::LibMeshInit init(argc, argv);

  // // Initialize the mesh
  // // Create a 2D mesh distributed across the default MPI communicator.
  // libMesh::Mesh mesh(init.comm(), 2);
  // // prevent libMesh from renumber nodes on its own
  // mesh.allow_renumbering(false);
  // mesh.read(param.in_filename);

  // // Construct BC map for displacement
  // std::map<libMesh::boundary_id_type, unsigned int> dirichlet_bcs;
  // // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
  // dirichlet_bcs.insert(
  //   std::pair<libMesh::boundary_id_type, unsigned int>(0, 7));
  // ShellSolid::shellsolid shell(mesh, param);
  // shell.make_constraints(dirichlet_bcs);
  // shell.run();

  return 0;
}
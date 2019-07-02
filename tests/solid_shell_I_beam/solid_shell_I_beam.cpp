#include "fem-shell.h"

int main(int argc, char *argv[])
{
  ShellSolid::shellparam param;
  // read command-line arguments and initialize global variables
  if (param.read_parameters(argc, argv))
    {
      std::cout << "Read command-line arguments.......OK" << std::endl;
    }
  else
    {
      std::cout << "Read command-line arguments.......FAILED" << std::endl;
      return -1;
    }
  // Hard coded parameters for test E
  param.nu = 0.3;
  param.em = 10000;
  param.thickness = 0.25;
  param.debug = false;

  // Initialize libMesh and any dependent library
  libMesh::LibMeshInit init(argc, argv);

  // Initialize the mesh
  // Create a 2D mesh distributed across the default MPI communicator.
  libMesh::Mesh mesh(init.comm(), 2);
  // prevent libMesh from renumber nodes on its own
  mesh.allow_renumbering(false);
  mesh.read(param.in_filename);

  // Construct BC map for displacement
  std::map<libMesh::boundary_id_type, unsigned int> dirichlet_bcs;
  // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
  dirichlet_bcs.insert(
    std::pair<libMesh::boundary_id_type, unsigned int>(0, 7));
  ShellSolid::shellsolid shell(mesh, param);
  shell.make_constraints(dirichlet_bcs);
  shell.run();
  return 0;
}
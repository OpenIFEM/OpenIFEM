#include "shell_solid_solver.h"
#include <deal.II/grid/grid_out.h>
#include <fstream>
#include <libmesh/ucd_io.h>

namespace Solid
{
  using namespace dealii;

  ShellSolidSolver::ShellSolidSolver(Triangulation<2, 3> &tria,
                                     const Parameters::AllParameters &params)
    : SolidSolver<2, 3>(tria, params), libmesh_init(0, nullptr)
  {
  }

  void ShellSolidSolver::initialize_system()
  {
    SolidSolver<2, 3>::initialize_system();
    // Transfer the mesh into the wrapped class ShellSolid::shellSslid.
    std::fstream tria_buffer("tmp_mesh.ucd");
    GridOut grid_out;
    // Set flags that writes boundary ids.
    grid_out.set_flags(GridOutFlags::Ucd(false, true, true));
    grid_out.write(triangulation, tria_buffer);
    // Constructor a libMesh mesh object
    libMesh::Mesh mesh(libmesh_init.comm(), 2);
    mesh.allow_renumbering(false);
    libMesh::UCDIO libmesh_input(mesh);
    libmesh_input.read("tmp_mesh");
    // Create the params file for shellsolid
    ShellSolid::shellparam shell_params;
    shell_params.debug = false;
    shell_params.nu = parameters.nu[0];
    shell_params.em = parameters.E[0];
    shell_params.thickness = 0.1;
    shell_params.isOutfileSet = false;
    // Create internal shell solid solver
    this->m_shell =
      std::make_unique<ShellSolid::shellsolid>(mesh, shell_params);
  }

  void ShellSolidSolver::update_strain_and_stress() {}

  void ShellSolidSolver::assemble_system(bool initial_step)
  {
    // Do nothing. We don't assemble in the wrapper
    (void)initial_step;
  }

  void ShellSolidSolver::run_one_step(bool first_step) { (void)first_step; }

  void ShellSolidSolver::synchronize() {}

} // namespace Solid
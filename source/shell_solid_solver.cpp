#include "shell_solid_solver.h"

namespace Solid
{
  using namespace dealii;

  ShellSolidSolver::ShellSolidSolver(Triangulation<2, 3> &tria,
                                     const Parameters::AllParameters &params)
    : SolidSolver<2, 3>(tria, params)
  {
  }

  void ShellSolidSolver::initialize_system()
  {
    SolidSolver<2, 3>::initialize_system();
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
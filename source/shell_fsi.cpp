#include "shell_fsi.h"

ShellFSI::ShellFSI(Fluid::FluidSolver<3> &f,
                   Solid::SolidSolver<2, 3> &s,
                   const Parameters::AllParameters &p,
                   bool use_dirichlet_bc)
  : FSI<3, 2>(f, s, p, use_dirichlet_bc)
{
  Assert(use_dirichlet_bc == false,
         ExcMessage("ShellFSI cannot use dirichlet BC for fluid!"));
}

void ShellFSI::run() {}

void ShellFSI::update_solid_box() {}
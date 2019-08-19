#ifndef SHELL_FSI_H
#define SHELL_FSI_H

#include "fsi.h"

using namespace dealii;

class ShellFSI : public FSI<3, 2>
{
public:
  ShellFSI(Fluid::FluidSolver<3> &,
           Solid::SolidSolver<2, 3> &,
           const Parameters::AllParameters &,
           bool use_dirichlet_bc = false);
  virtual void run() override;
  ~ShellFSI();

protected:
  using FSI<3, 2>::point_in_solid;
  using FSI<3, 2>::update_indicator;
  using FSI<3, 2>::move_solid_mesh;
  using FSI<3, 2>::find_solid_bc;
  using FSI<3, 2>::update_solid_displacement;
  using FSI<3, 2>::find_fluid_bc;
  using FSI<3, 2>::refine_mesh;
  using FSI<3, 2>::fluid_solver;
  using FSI<3, 2>::solid_solver;
  using FSI<3, 2>::parameters;
  using FSI<3, 2>::time;
  using FSI<3, 2>::timer;
  using FSI<3, 2>::solid_box;
  using FSI<3, 2>::use_dirichlet_bc;

  virtual void update_solid_box();
};
#endif
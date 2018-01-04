#ifndef FSI_H
#define FSI_H

#include "linearElasticSolver.h"
#include "navierstokes.h"

using namespace dealii;

template <int dim>
class FSI
{
public:
  FSI(Fluid::NavierStokes<dim> &,
      Solid::LinearElasticSolver<dim> &,
      const Parameters::AllParameters &);
  /**
   * Update the indicator field of the fluid solver.
   * This is done with brute force: compute the center of every fluid cell and
   * test if it is in any solid cell.
   */
  void update_indicator();
  void initialize_system();
  void move_solid_mesh(bool);
  void run();

private:
  Fluid::NavierStokes<dim> &fluid_solver;
  Solid::LinearElasticSolver<dim> &solid_solver;
  Parameters::AllParameters parameters;
  Utils::Time time;
};

#endif

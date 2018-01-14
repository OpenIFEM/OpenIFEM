#ifndef FSI_H
#define FSI_H

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

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
  /**
   * Set up the initialize both fluid and solid solvers.
   */
  void initialize_system();
  /**
   * Move solid triangulation either forward or backward using displacements,
   */
  void move_solid_mesh(bool);
  /**
   * Solid Neumann boundary conditions are obtained from the fluid.
   * Currently we only consider the fluid pressure instead of the full traction.
   * The implementation is again straight-forward: loop over the faces on the
   * solid Neumann boundary, compute the center point, find which fluid cell it
   * is in, compute the pressure at that location by interpolation from the
   * support
   * points. The pressure is then applied as the boundary condition on the solid
   * face.
   */
  void find_solid_bc(std::vector<Tensor<1, dim>> &);
  void run();

private:
  Fluid::NavierStokes<dim> &fluid_solver;
  Solid::LinearElasticSolver<dim> &solid_solver;
  Parameters::AllParameters parameters;
  Utils::Time time;
};

#endif

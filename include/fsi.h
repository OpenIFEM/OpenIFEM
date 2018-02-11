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
  void run();

private:
  /**
   * Helper function to determine if a point is inside the mesh associated to a
   * DoFHandler.
   * If it is not inside, the whole triangulation will be iterated.
   */
  bool point_in_mesh(const DoFHandler<dim> &, const Point<dim> &);
  /**
   * Update the indicator field of the fluid solver.
   * If all of the volume quadrature points of a fluid cell is immersed in the
   * solid
   * triangulation, then it is identified as an artificial fluid cell.
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
  /**
   * The solid acts on the fluid through the FSI force defined as:
   * \f$ F^{\text{FSI}} = (\sigma^s_{ij,j} - \sigma^f_{ij,j}) -
   * \rho^s (\frac{Dv^s_i}{Dt} - \frac{Dv^f_i}{Dt})\f$.
   * We need to evaluate \f$ \int \delta v_i F^{\text{FSI}} d\Omega =
   * \int \delta v_{i,j}(\sigma^f_{ij} - \sigma^s_{ij}) d\Omega -
   * \int \delta v_i \rho^s (\frac{Dv^s_i}{Dt} - \frac{Dv^f_i}{Dt}) d\Omega\f$
   * in the artificial fluid domain.
   *
   * The solid solver has the acceleration and nodal stress computed, this
   * function
   * iterate through all fluid quadrature points and interpolate the solid
   * acceleration
   * and stress at them. In addition, it computes the fluid stress and
   * acceleration
   * term. Combing these terms altogether forms a body force, which is used by
   * the
   * fluid solver.
   *
   * Note the solid acceleration provided by the solid solver is already the
   * material
   * acceleration because it uses Lagrangian formulation. The fluid
   * acceleration, however,
   * must take convection into account.
   */
  void find_fluid_fsi(std::vector<SymmetricTensor<2, dim>> &,
                      std::vector<Tensor<1, dim>> &);

  Fluid::NavierStokes<dim> &fluid_solver;
  Solid::LinearElasticSolver<dim> &solid_solver;
  Parameters::AllParameters parameters;
  Utils::Time time;
};

#endif

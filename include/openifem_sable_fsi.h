#ifndef OpenIFEM_Sable_FSI_H
#define OpenIFEM_Sable_FSI_H

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "sable_wrapper.h"
#include "fsi.h"
#include "utilities.h"

using namespace dealii;

extern template class Fluid::FluidSolver<2>;
extern template class Fluid::FluidSolver<3>;
extern template class Solid::SolidSolver<2>;
extern template class Solid::SolidSolver<3>;
extern template class Utils::GridInterpolator<2, Vector<double>>;
extern template class Utils::GridInterpolator<3, Vector<double>>;
extern template class Utils::GridInterpolator<2, BlockVector<double>>;
extern template class Utils::GridInterpolator<3, BlockVector<double>>;
extern template class Utils::SPHInterpolator<2, Vector<double>>;
extern template class Utils::SPHInterpolator<3, Vector<double>>;

template <int dim>
class OpenIFEM_Sable_FSI : public FSI<dim> 
{
public:
  OpenIFEM_Sable_FSI(Fluid::SableWrap<dim> &,
      Solid::SolidSolver<dim> &,
      const Parameters::AllParameters &,
      bool use_dirichlet_bc = false);
  virtual void run();
  ~OpenIFEM_Sable_FSI();

private:
    using FSI<dim>::update_solid_box;
    using FSI<dim>::point_in_solid;
    using FSI<dim>::move_solid_mesh;
    using FSI<dim>::find_solid_bc;
    using FSI<dim>::update_solid_displacement;
    using FSI<dim>::refine_mesh;
    using FSI<dim>::fluid_solver;
    using FSI<dim>::solid_solver;
    using FSI<dim>::parameters;
    using FSI<dim>::time;
    using FSI<dim>::timer;
    using FSI<dim>::solid_box;
    using FSI<dim>::use_dirichlet_bc;
  
  /*! \brief Update the indicator field of the fluid solver.
   *
   *  Although the indicator field is defined at quadrature points in order to
   *  cache the fsi force, the quadrature points in the same cell are updated
   *  as a whole: they are either all 1 or all 0. The criteria is that whether
   *  all of the vertices are in solid mesh (because later on Dirichlet BCs
   *  obtained from the solid will be applied).
   */
   void update_indicator();

  /*! \brief Compute the Dirichlet BCs on the artificial fluid using solid
   * velocity,
   *         as well as the fsi stress and acceleration terms at the artificial
   * fluid
   *         quadrature points.
   *
   *  The Dirichlet BCs are obtained by interpolating solid velocity to the
   * fluid
   *  vertices and the FSI force is defined as:
   *  \f$F^{\text{FSI}} = \frac{Dv^f_i}{Dt}) - \sigma^f_{ij,j})\f$.
   *  In practice, we avoid directly evaluating stress divergence, so the stress
   *  itself and the acceleration are separately cached onto the fluid
   * quadrature
   *  points to be used by the fluid solver.
   */
  void find_fluid_bc();

  Fluid::SableWrap<dim> &sable_solver;
};

#endif

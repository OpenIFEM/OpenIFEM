#ifndef OpenIFEM_Sable_FSI_H
#define OpenIFEM_Sable_FSI_H

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "fsi.h"
#include "sable_wrapper.h"
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
  using FSI<dim>::update_solid_displacement;
  using FSI<dim>::refine_mesh;
  using FSI<dim>::fluid_solver;
  using FSI<dim>::solid_solver;
  using FSI<dim>::parameters;
  using FSI<dim>::time;
  using FSI<dim>::timer;
  using FSI<dim>::solid_box;
  using FSI<dim>::use_dirichlet_bc;

  /// Setup the hints for searching for each fluid cell.
  void setup_cell_hints();

  /*! \brief Update the indicator field of the fluid solver.
   *
   *  Although the indicator field is defined at quadrature points in order to
   *  cache the fsi force, the quadrature points in the same cell are updated
   *  as a whole: they are either all 1 or all 0. The criteria is that whether
   *  all of the vertices are in solid mesh (because later on Dirichlet BCs
   *  obtained from the solid will be applied).
   */
  void update_indicator();

  /*  Calculate indicator field based on quadrature points. Indicator value is
   * between 0 and 1 based on the number of quadrature points inside the solid*/
  void update_indicator_qpoints();

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

  /* Calculate FSI force at quadrature points instead of nodes */
  void find_fluid_bc_qpoints();

  /*! \brief Compute the fluid traction on solid boundaries.
   *
   *  The implementation is straight-forward: loop over the faces on the
   *  solid Neumann boundary, find the quadrature points and normals,
   *  then interpolate the fluid pressure and symmetric gradient of velocity at
   *  those points, based on which the fluid traction is calculated.
   */
  void find_solid_bc();

  /*! More efficient implementation of find_fluid_bc
   */
  void find_fluid_bc_new();

  /*! More efficient implementation of find_fluid_bc_qpoints
   */
  void find_fluid_bc_qpoints_new();

  /*! \brief Compute the fluid cell index given the solid quad point coordiantes
   * and its unit normal vector Currently, this algorithm only works for a 2-D
   * structured Eulerian mesh without refinements.
   */
  int compute_fluid_cell_index(Point<dim> &, const Tensor<1, dim> &);

  /*! \brief Compute the added mass effect on the Lagrangian boundary
   */
  void compute_added_mass();

  /*! output l2 norm of the velocity difference calculated at both Eulerian and
   * Lagrangian mesh */

  void output_vel_diff(bool first_step);

  /*! checks if the point is inside Lagrangian solid, if inside then returns the
   * iteratore of corresponding cell */
  std::pair<bool, const typename DoFHandler<dim>::active_cell_iterator>
  point_in_solid_new(const DoFHandler<dim> &df, const Point<dim> &point);

  /*! chekcs if the point is in the given cell. For 2D it is same as deal.ii
     cell->point_inside() but for 3D it is modified to include a tolerence
     value. Note: for faster results, check if the point is inside the
     cell's bounding box before calling this function */
  bool point_in_cell(const typename DoFHandler<dim>::active_cell_iterator &,
                     const Point<dim> &);

  /*! if an Eulerian vertex is inside the solid, stores vertex id and the
   * corresponding solid cell iterator*/
  std::unordered_map<int, const typename DoFHandler<dim>::active_cell_iterator>
    vertex_indicator_data;

  /* value is set to true if the cell is partially inside solid
     value is false if the cell is completely inside or outside solid
  */
  std::vector<bool> cell_partially_inside_solid;

  /*map key: id of the cell which is partially inside the solid
    map objects: vectors store local node ids which are inside and outside the
    solid
  */
  std::map<int, std::vector<int>> cell_nodes_inside_solid;
  std::map<int, std::vector<int>> cell_nodes_outside_solid;

  // Cell storage that stores hints of cell searching from last time step.
  CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
                  typename DoFHandler<dim>::active_cell_iterator>
    cell_hints;

  Fluid::SableWrap<dim> &sable_solver;
};

#endif

#ifndef MPI_OpenIFEM_Sable_FSI_H
#define MPI_OpenIFEM_Sable_FSI_H

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "mpi_fsi.h"
#include "mpi_sable_wrapper.h"
#include "utilities.h"

using namespace dealii;

extern template class Fluid::MPI::FluidSolver<2>;
extern template class Fluid::MPI::FluidSolver<3>;
extern template class Solid::MPI::SharedSolidSolver<2>;
extern template class Solid::MPI::SharedSolidSolver<3>;
extern template class Utils::GridInterpolator<2, Vector<double>>;
extern template class Utils::GridInterpolator<3, Vector<double>>;
extern template class Utils::GridInterpolator<2, PETScWrappers::MPI::Vector>;
extern template class Utils::GridInterpolator<3, PETScWrappers::MPI::Vector>;
extern template class Utils::GridInterpolator<2,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::GridInterpolator<3,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::SPHInterpolator<2, Vector<double>>;
extern template class Utils::SPHInterpolator<3, Vector<double>>;
extern template class Utils::SPHInterpolator<2,
                                             PETScWrappers::MPI::BlockVector>;
extern template class Utils::SPHInterpolator<3,
                                             PETScWrappers::MPI::BlockVector>;
extern template class Utils::CellLocator<2, DoFHandler<2, 2>>;
extern template class Utils::CellLocator<3, DoFHandler<3, 3>>;

namespace MPI
{
  template <int dim>
  class OpenIFEM_Sable_FSI : public FSI<dim>
  {
  public:
    OpenIFEM_Sable_FSI(Fluid::MPI::SableWrap<dim> &,
                       Solid::MPI::SharedSolidSolver<dim> &,
                       const Parameters::AllParameters &,
                       bool use_dirichlet_bc = false);

    virtual void run();
    ~OpenIFEM_Sable_FSI();

  private:
    using FSI<dim>::fluid_solver;
    using FSI<dim>::solid_solver;
    using FSI<dim>::parameters;
    using FSI<dim>::time;
    using FSI<dim>::timer;
    using FSI<dim>::mpi_communicator;
    using FSI<dim>::pcout;
    using FSI<dim>::update_solid_box;
    using FSI<dim>::move_solid_mesh;
    using FSI<dim>::solid_box;
    using FSI<dim>::point_in_solid;

    Fluid::MPI::SableWrap<dim> &sable_solver;

    /*  Calculate indicator field based on quadrature points. Indicator value is
     * between 0 and 1 based on the number of quadrature points inside the
     * solid*/
    void update_indicator_qpoints();

    /*! chekcs if the point is in the given cell. For 2D it is same as deal.ii
       cell->point_inside() but for 3D it is modified to include a tolerence
       value. Note: for faster results, check if the point is inside the
       cell's bounding box before calling this function */
    bool point_in_cell(const typename DoFHandler<dim>::active_cell_iterator &,
                       const Point<dim> &);

    /*! checks if the point is inside Lagrangian solid, if inside then returns
     * the iteratore of corresponding cell */
    std::pair<bool, const typename DoFHandler<dim>::active_cell_iterator>
    point_in_solid_new(const DoFHandler<dim> &df, const Point<dim> &point);

    /*! \brief Compute the Dirichlet BCs on the artificial fluid using solid
     * velocity,
     *         as well as the fsi stress and acceleration terms at the
     * artificial fluid quadrature points.
     *
     *  The Dirichlet BCs are obtained by interpolating solid velocity to the
     * fluid
     *  vertices and the FSI force is defined as:
     *  \f$F^{\text{FSI}} = \frac{Dv^f_i}{Dt}) - \sigma^f_{ij,j})\f$.
     *  In practice, we avoid directly evaluating stress divergence, so the
     * stress itself and the acceleration are separately cached onto the fluid
     * quadrature
     *  points to be used by the fluid solver.
     */
    /* Calculate FSI force at quadrature points instead of nodes */
    void find_fluid_bc_qpoints();

    /*! if an Eulerian vertex is inside the solid, stores vertex id and the
     * corresponding solid cell iterator*/
    std::unordered_map<int,
                       const typename DoFHandler<dim>::active_cell_iterator>
      vertex_indicator_data;

    /*map key: id of the cell which is partially inside the solid
     map objects: vectors store local node ids which are inside and outside the
     solid
   */
    std::unordered_map<int, std::vector<int>> cell_nodes_inside_solid;
    std::unordered_map<int, std::vector<int>> cell_nodes_outside_solid;

    /*! \brief Compute the fluid cell index given the solid quad point
     * coordiantes and its unit normal vector Currently, this algorithm only
     * works structured Eulerian mesh without refinements.
     */
    int compute_fluid_cell_index_qpoint(Point<dim> &, const Tensor<1, dim> &);

    /*! \brief Compute the fluid cell index given the solid vertex
     * coordiantes, this algorithm only
     * works structured Eulerian mesh without refinements.
     */
    int compute_fluid_cell_index_vertex(Point<dim> &);

    /*! \brief Compute the fluid traction on solid boundaries.
     *
     *  The implementation is straight-forward: loop over the faces on the
     *  solid Neumann boundary, find the quadrature points and normals,
     *  then interpolate the fluid pressure and symmetric gradient of velocity
     * at those points, based on which the fluid traction is calculated.
     */
    void find_solid_bc();

    /*! \brief Compute the added mass effect on the Lagrangian boundary
     */
    void compute_added_mass();

    /*! calculate velocity difference on Lagrangian mesh for penalty application
     * and output l2 norm of the velocity difference calculated at both Eulerian
     * and Lagrangian mesh */
    void compute_lag_penalty(bool first_step);
  };
} // namespace MPI

#endif

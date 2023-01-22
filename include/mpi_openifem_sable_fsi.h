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
  };
} // namespace MPI

#endif

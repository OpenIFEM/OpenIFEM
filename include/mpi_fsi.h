#ifndef MPI_FSI
#define MPI_FSI

#include <deal.II/base/table_indices.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include "mpi_fluid_solver.h"
#include "mpi_shared_solid_solver.h"

using namespace dealii;

extern template class Fluid::MPI::FluidSolver<2>;
extern template class Fluid::MPI::FluidSolver<3>;
extern template class Solid::MPI::SharedSolidSolver<2>;
extern template class Solid::MPI::SharedSolidSolver<3>;
extern template class Utils::GridInterpolator<2, Vector<double>>;
extern template class Utils::GridInterpolator<3, Vector<double>>;
extern template class Utils::GridInterpolator<2,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::GridInterpolator<3,
                                              PETScWrappers::MPI::BlockVector>;
extern template class Utils::DiracDeltaInterpolator<2, Vector<double>>;
extern template class Utils::DiracDeltaInterpolator<3, Vector<double>>;

namespace MPI
{
  template <int dim>
  class FSI
  {
  public:
    FSI(Fluid::MPI::FluidSolver<dim> &,
        Solid::MPI::SharedSolidSolver<dim> &,
        const Parameters::AllParameters &);
    void run();

    //! Destructor
    ~FSI();

  private:
    /// Define a smallest rectangle (or hex in 3d) that contains the solid.
    void update_solid_box();

    /// Check if a point is inside a mesh.
    bool point_in_solid(const DoFHandler<dim> &, const Point<dim> &);

    /*! \brief Update the indicator field of the fluid solver.
     *
     *  Although the indicator field is defined at quadrature points in order
     * to cache the fsi force, the quadrature points in the same cell are
     * updated as a whole: they are either all 1 or all 0. The criteria is
     * that whether all of the vertices are in solid mesh (because later on
     * Dirichlet BCs obtained from the solid will be applied).
     */
    void update_indicator();

    /// Move solid triangulation either forward or backward using
    /// displacements,
    void move_solid_mesh(bool);

    /*! \brief Compute the fluid traction on solid boundaries.
     *
     *  The implementation is straight-forward: loop over the faces on the
     *  solid Neumann boundary, find the quadrature points and normals,
     *  then interpolate the fluid pressure and symmetric gradient of velocity
     * at those points, based on which the fluid traction is calculated.
     */
    void find_solid_bc();

    /*! \brief Interpolate the fluid velocity to solid vertices.
     *
     *  This is IFEM, not mIFEM.
     */
    void update_solid_displacement();

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
    void find_fluid_bc();

    // For MPI FSI, the solid solver uses shared trianulation. i.e.,
    // each process has the entire graph, for the ease of looping.
    Fluid::MPI::FluidSolver<dim> &fluid_solver;
    Solid::MPI::SharedSolidSolver<dim> &solid_solver;
    Parameters::AllParameters parameters;
    MPI_Comm mpi_communicator;
    ConditionalOStream pcout;
    Utils::Time time;
    mutable TimerOutput timer;

    // This vector represents the smallest box that contains the solid.
    // The point stored is in the order of:
    // (x_min, x_max, y_min, y_max, z_min, z_max)
    Vector<double> solid_box;
  };
} // namespace MPI

#endif

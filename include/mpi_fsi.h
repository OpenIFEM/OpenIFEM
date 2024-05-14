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
  class FSI
  {
  public:
    FSI(Fluid::MPI::FluidSolver<dim> &,
        Solid::MPI::SharedSolidSolver<dim> &,
        const Parameters::AllParameters &,
        bool use_dirichlet_bc = false);
    void run();

    void
    set_penetration_criterion(const std::function<double(const Point<dim> &)> &,
                              Tensor<1, dim>);

    //! Destructor
    ~FSI();

  protected:
    using SolidVerticesStorage = std::map<
      typename Triangulation<dim>::active_vertex_iterator,
      std::pair<
        std::list<std::pair<typename Triangulation<dim>::active_cell_iterator,
                            unsigned int>>,
        int>>;

    /// Collect all the boundary lines in solid triangulation.
    void collect_solid_boundaries();

    /// Setup the hints for searching for each fluid cell.
    void setup_cell_hints();

    /// Define a smallest rectangle (or hex in 3d) that contains the solid.
    void update_solid_box();

    /// Find the vertices that are owned by the local process.
    void update_vertices_mask();

    /*! checks if the point is inside Lagrangian solid, if inside then returns
     * the iteratore of corresponding cell */
    std::pair<bool, const typename DoFHandler<dim>::active_cell_iterator>
    point_in_solid_new(const DoFHandler<dim> &, const Point<dim> &);

    /// OLDVERSION Check if a point is inside a mesh.
    bool point_in_solid(const DoFHandler<dim> &, const Point<dim> &);

    /*! check if the point is in the given cell, including a tolerence
     value. Note: for faster results, check if the point is inside the
     cell's bounding box before calling this function */
    bool point_in_cell(const typename DoFHandler<dim>::active_cell_iterator &,
                       const Point<dim> &);

    /*! if an Eulerian vertex is inside the solid, stores vertex id and the
     * corresponding solid cell iterator*/
    std::unordered_map<int,
                       const typename DoFHandler<dim>::active_cell_iterator>
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

    /*! \brief Apply contact model specific to VF simulation
     */
    void apply_contact_model(bool);

    /*! \brief Collect the solid boundary vertices, except for those fixed
     * boundaries. This can be used in computing nearest wall distance for some
     * turbulence models.
     */
    void collect_solid_boundary_vertices();

    /// Mesh adaption.
    void refine_mesh(const unsigned int, const unsigned int);

    void compute_fsi_energy_artificial();

    void compute_fsi_energy_solid();

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

    // This vector collects the solid boundaries for computing the winding
    // number.
    std::list<std::pair<typename Triangulation<dim>::active_cell_iterator,
                        unsigned int>>
      solid_boundaries;
    // A collection of moving solid boundary vertices and their adjacent free
    // boundary lines. Also stores their corresponding indices for shear
    // velocity values in shear_velocities
    SolidVerticesStorage solid_boundary_vertices;
    // For wall function
    Vector<double> shear_velocities;

    // A mask that marks local fluid vertices for solid bc interpolation
    // searching.
    std::vector<bool> vertices_mask;

    // Cell storage that stores hints of cell searching from last time step.
    CellDataStorage<
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
      typename DoFHandler<dim>::active_cell_iterator>
      cell_hints;

    // A function that determines if a point is penetrating the fluid domain
    std::shared_ptr<std::function<double(const Point<dim> &)>>
      penetration_criterion;
    Tensor<1, dim> penetration_direction;

    bool use_dirichlet_bc;
  };
} // namespace MPI

#endif

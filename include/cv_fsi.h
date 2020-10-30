#ifndef VF_SIM
#define VF_SIM

#include "mpi_fsi.h"

using namespace dealii;

namespace MPI
{
  template <int dim>
  class ControlVolumeFSI : public FSI<dim>
  {
  public:
    ControlVolumeFSI(Fluid::MPI::FluidSolver<dim> &,
                     Solid::MPI::SharedSolidSolver<dim> &,
                     const Parameters::AllParameters &,
                     bool use_dirichlet_bc = false);
    void run_with_cv_analysis();
    void set_control_volume_boundary(const std::vector<double> &);

    //! Destructor
    ~ControlVolumeFSI();

  protected:
    using FSI<dim>::collect_solid_boundaries;
    using FSI<dim>::setup_cell_hints;
    using FSI<dim>::update_solid_box;
    using FSI<dim>::update_vertices_mask;
    using FSI<dim>::point_in_solid;
    using FSI<dim>::update_indicator;
    using FSI<dim>::move_solid_mesh;
    using FSI<dim>::find_solid_bc;
    using FSI<dim>::apply_contact_model;
    using FSI<dim>::update_solid_displacement;
    using FSI<dim>::find_fluid_bc;
    using FSI<dim>::refine_mesh;
    using FSI<dim>::fluid_solver;
    using FSI<dim>::solid_solver;
    using FSI<dim>::parameters;
    using FSI<dim>::mpi_communicator;
    using FSI<dim>::pcout;
    using FSI<dim>::time;
    using FSI<dim>::timer;
    using FSI<dim>::solid_box;
    using FSI<dim>::solid_boundaries;
    using FSI<dim>::vertices_mask;
    using FSI<dim>::cell_hints;
    using FSI<dim>::use_dirichlet_bc;
    using FSI<dim>::penetration_criterion;
    using FSI<dim>::penetration_direction;

    struct CVValues;
    struct SurfaceCutter;

    void collect_control_volume_cells();
    void collect_inlet_outlet_cells();
    void control_volume_analysis();
    void compute_relative_velocity();
    void get_separation_point();
    void compute_flux();
    void compute_volume_integral();
    void compute_interface_integral();

    PETScWrappers::MPI::BlockVector fluid_previous_solution;

    std::vector<double> control_volume_boundaries;
    /* The outer map stores the inner maps by the sequence of x coordinates.
       the inner map stores the fluid cells in the control volume by the sequnce
       of y coordinates.
    */
    std::map<double,
             std::map<double, typename DoFHandler<dim>::active_cell_iterator>>
      cv_f_cells;
    std::set<typename DoFHandler<dim>::active_cell_iterator> inlet_cells;
    std::set<typename DoFHandler<dim>::active_cell_iterator> outlet_cells;
    CellDataStorage<
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
      Tensor<1, dim>>
      solid_surface_velocity;
    CellDataStorage<
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
      SurfaceCutter>
      surface_cutters;
    CVValues cv_values;

    struct CVValues
    {
      // Initialize the output file
      void initialize_output(const Utils::Time &time,
                             MPI_Comm &mpi_communicator);
      // Sum the results over all MPI ranks
      void reduce(MPI_Comm &mpi_communicator);
      // Reset all quantities to zero
      void reset();
      Point<dim> separation_point;
      // Defined as \int_S_{in/out}{u_1}dS
      double inlet_volume_flow;
      double outlet_volume_flow;
      // Defined as \int_S_{in/out}{p}dS
      double inlet_pressure_force;
      double outlet_pressure_force;
      // Defined as \int_V_{VF}{1}dV
      double VF_volume;
      struct momentum_equation
      {
        // Defined as \int_S_{in/out}{\rho u_1 (u_1 - w_1)n_1}dS
        // Here, w_1 is the wall velcotiy which is 0.
        double inlet_flux;
        double outlet_flux;
        // Defined as \frac{d}{dt}\int_V{\rho u_1}dV
        double rate_momentum;
        // Defined as \int_S_{VF}{p n_1}dS
        double VF_drag;
        // Defined as \int_S_{VF}{\tau_{1j}n_j}dS
        double VF_friction;
      };
      struct energy_equation
      {
        // Defined as \int_S_{in/out}{pQ}dS
        double inlet_pressure_work;
        double outlet_pressure_work;
        // Defined as 0.5 * \int_S_{in/out}{\rho u1 u_i u_i}dS
        double inlet_flux;
        double outlet_flux;
        // Defined as 0.5 * \frac{d}{dt}\int_V{\rho u_i u_i}dV
        double rate_kinetic_energy;
        // Defined as \int_V{\mu (u_{i,j}^2 + u_{i,j}u_{j,i})}dV
        double rate_dissipation;
        // Defined as \int_V{p u_{i,i}}dV
        double rate_compression_work;
        // Defined as \int_S_{VF}{\tau_{ij} u_i n_j}dS
        double rate_friction_work;
        // Defined as \int_S{p u_i n_i}dS
        double rate_vf_work;
      };
      momentum_equation momentum;
      energy_equation energy;
      std::fstream output;
    };

    /* Surface cutter is a triangulation that has only 1 element
     * corresponding to each element being cut. The cutters are used to compute
     * the surface integrals
     */
    struct SurfaceCutter
    {
      SurfaceCutter();
      Triangulation<dim - 1, dim> tria;
      DoFHandler<dim - 1, dim> dof_handler;
      FESystem<dim - 1, dim> fe;
      QGauss<dim - 1> quad_formula;
      // This quadrature is for the parent cell to interpolate the solution on
      Quadrature<dim> interpolate_q;
      // The volume fraction of the parent cell for the computation of the
      // volume integrals
      double volume_fraction;
    };
  };
} // namespace MPI

#endif
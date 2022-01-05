#ifndef CV_FSI
#define CV_FSI

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
    void set_output_solid_boundary(const bool);
    void set_pressure_probe(const Point<dim>);

    //! Destructor
    ~ControlVolumeFSI();

  protected:
    using cell_iterator = typename DoFHandler<dim>::active_cell_iterator;
    using FSI<dim>::collect_solid_boundaries;
    using FSI<dim>::setup_cell_hints;
    using FSI<dim>::update_solid_box;
    using FSI<dim>::update_vertices_mask;
    using FSI<dim>::point_in_solid;
    using FSI<dim>::update_indicator;
    using FSI<dim>::move_solid_mesh;
    using FSI<dim>::find_solid_bc;
    using FSI<dim>::apply_contact_model;
    using FSI<dim>::collect_solid_boundary_vertices;
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
    using FSI<dim>::solid_boundary_vertices;
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
    void compute_flux();
    void compute_volume_integral();
    void compute_interface_integral();
    void compute_bernoulli_terms();
    void output_solid_boundary_vertices();

    PETScWrappers::MPI::BlockVector fluid_previous_solution;

    /* Control volume boundaries. It has 2 * dim length with the following
       sequence: left right lower upper (front back)
    */
    std::vector<double> control_volume_boundaries;
    /* The outer map stores the inner maps by the sequence of x coordinates.
       the inner map stores the fluid cells in the control volume by the sequnce
       of y coordinates.
    */
    std::map<double, std::map<double, cell_iterator>> cv_f_cells;
    std::set<cell_iterator> inlet_cells;
    std::set<cell_iterator> outlet_cells;
    /* Two ends for Bernoulli contraction and jet region. The first is left
       (point A) and second is right (point B). If gap is not zero then B for
       contraction coincides with A for jet.
       A for contraction and B for jet region (local). Second arg is the
       fraction in the region for integral computation.
    */
    std::pair<std::tuple<cell_iterator, cell_iterator, double>,
              std::tuple<cell_iterator, cell_iterator, double>>
      bernoulli_start_end;
    /* Streamline path for Bernoulli analysis. The key is the center x
       coordinate for sorting. Firt iterator is for vector DoF and second is for
       scalar DoF handler. Note: this is local
    */
    std::map<double, std::pair<cell_iterator, cell_iterator>>
      streamline_path_cells;

    /* If this flag is set, the solid boundary nodes will be output every
    timestep for POD analysis.
    */
    bool output_solid_boundary;

    // Pressure probe location.
    bool pressure_probe_set;
    Point<dim> pressure_probe_location;
    std::optional<Utils::GridInterpolator<dim, PETScWrappers::MPI::BlockVector>>
      pressure_probe;

    CellDataStorage<
      typename parallel::distributed::Triangulation<dim>::active_cell_iterator,
      SurfaceCutter>
      surface_cutters;
    CVValues cv_values;

    /* Quantities related to gap volume flow
     */
    double solid_tip_x;

    struct CVValues
    {
      // Initialize the output file
      void initialize_output(const Utils::Time &time,
                             MPI_Comm &mpi_communicator);
      // Sum the results over all MPI ranks
      void reduce(MPI_Comm &mpi_communicator, double);
      // Reset all quantities to zero
      void reset();
      Point<dim> separation_point;
      // Defined as \int_S_{in/out}{u_1}dS
      double inlet_volume_flow;
      double outlet_volume_flow;
      double gap_volume_flow;
      // Defined as \int_S_{in/out}{p}dS
      double inlet_pressure_force;
      double outlet_pressure_force;
      // Defined as \int_V_{VF}{1}dV
      double VF_volume;
      // Maximum velocity magnitude
      double max_velocity;
      // Pressure probe
      double probed_pressure;
      struct bernoulli_equation
      {
        double rate_convection_contraction;
        double rate_convection_jet;
        double rate_pressure_grad_contraction;
        double rate_pressure_grad_jet;
        double acceleration_contraction;
        double acceleration_jet;
        double rate_density_contraction;
        double rate_density_jet;
        double rate_friction_contraction;
        double rate_friction_jet;
        // Bernoulli separation points
        double contraction_end_x;
        double jet_start_x;
      };
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
        // Turbulent energy efflux
        double rate_turbulence_efflux;
        // The integral convection term. \int_V{\rho u cdot \Nabla u \cdot u}dV
        double convective_KE;
        // The integral of KE flux through the solid. (should be zero)
        double penetrating_KE;
        // Previous and present KE
        double previous_KE;
        double present_KE;
        // \int_V{p_,i u_i}dV
        double pressure_convection;
        // Defined as 0.5 * \frac{d}{dt}\int_V{\rho u_i u_i}dV
        double rate_kinetic_energy;
        // Defined as 0.5 * \frac\int_V{\rho \frac{d(u_i u_i)}{dt}}
        double rate_kinetic_energy_direct;
        // Defined as \int_V{\mu (u_{i,j}^2 + u_{i,j}u_{j,i})}dV
        double rate_dissipation;
        // Defined as \int_V{p u_{i,i}}dV
        double rate_compression_work;
        // Numerical stabilization effects
        double rate_stabilization;
        // Turbulent momentum transfer
        double rate_turbulence;
        // Defined as \int_S_{VF}{\tau_{ij} u_i n_j}dS
        double rate_friction_work;
        // Defined as \int_S{p u_i n_i}dS
        double rate_vf_work;
        // Defined as \int_S{p u_i n_i}dS
        double rate_vf_work_from_solid;
      };
      bernoulli_equation bernoulli;
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

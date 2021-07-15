#ifndef PARAMETERS
#define PARAMETERS

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

#include <iostream>
#include <string>
#include <vector>

namespace Parameters
{
  using namespace dealii;

  struct Simulation
  {
    std::string simulation_type;
    int dimension;
    std::vector<int> global_refinements;
    double end_time;
    double time_step;
    double output_interval;
    double refinement_interval;
    double save_interval;
    std::vector<double> gravity;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct FluidFESystem
  {
    unsigned int fluid_pressure_degree;
    unsigned int fluid_velocity_degree;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct FluidMaterial
  {
    double viscosity;
    double fluid_rho;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct FluidSolver
  {
    double grad_div;
    unsigned int fluid_max_iterations;
    double fluid_tolerance;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct FluidDirichlet
  {
    /** Use the hard-coded bc values or the input ones. */
    int use_hard_coded_values;
    /** Number of fluid Dirichlet BCs. */
    unsigned int n_fluid_dirichlet_bcs;
    /**
     * Fluid Dirchlet BCs are stored as a map between solid boundary id and
     * a pair of an int which indicates the components to be constrained,
     * and a vector of the constrained values.
     */
    std::map<unsigned int, std::pair<unsigned int, std::vector<double>>>
      fluid_dirichlet_bcs;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct FluidNeumann
  {
    /** Number of inhomogeneous fluid Neumann BCs. */
    unsigned int n_fluid_neumann_bcs;
    /**
     * Same as Fluid Dirichelet BCs, Neumann BCs are also stored as a map, but
     * no component needs to be specified, since we only have a scalar pressure
     * inlet.
     */
    std::map<unsigned int, double> fluid_neumann_bcs;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SpalartAllmarasModel
  {
    /** Number of Spalart-Allmaras model Dirichlet BCs. */
    unsigned int n_spalart_allmaras_model_bcs;
    /**
     * Spalart-Allmaras model BCs are stored as a map between fluid boundary id
     * and an int which indicates what kind of condition is applied:
     * 0: wall (all boundaries with no-penetration fluid bc)
     * 1: inflow (fluid velocity / pressue inlet)
     */
    std::map<unsigned int, unsigned int> spalart_allmaras_model_bcs;
    double spalart_allmaras_initial_condition_coefficient;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SolidFESystem
  {
    unsigned int solid_degree;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SolidMaterial
  {
    std::string solid_type;
    unsigned int n_solid_parts;
    double solid_rho;        //!< density, used by all types.
    std::vector<double> E;   //!< Young's modulus, linear elastic material only.
    std::vector<double> nu;  //!< Poisson's ratio, linear elastic material only.
    std::vector<double> eta; //!< Viscosity, linear elastic material only.
    std::vector<std::vector<double>> C; //!< Hyperelastic material constants.
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SolidSolver
  {
    double damping; //!< The artifial damping in Newmark-beta method.
    unsigned int solid_max_iterations; //!< Max number of Newton iterations,
                                       //! hyperelastic only.
    double tol_f;                      //!< Force tolerance
    double tol_d; //!< Displacement tolerance, hyperelastic only.
    double contact_force_multiplier; //!< Multiplier of the penetration distance
                                     //!< to compute contact force.
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SolidDirichlet
  {
    /** Number of solid Dirichlet BCs. */
    unsigned int n_solid_dirichlet_bcs;
    /**
     * Solid Dirchlet BCs are stored as a map between solid boundary id and
     * an int which indicates the components to be constrained.
     */
    std::map<unsigned int, unsigned int> solid_dirichlet_bcs;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct SolidNeumann
  {
    /** Number of solid Neumann BCs. */
    unsigned int n_solid_neumann_bcs;
    /** Type of solid Neumann BC. */
    std::string solid_neumann_bc_type;
    /**
     * Solid Neumann BCs are stored as a map between solid boundary id
     * and a vector of prescribed values.
     * If traction is given, then the vector length should be dim,
     * if pressure is given, then the vector length should be 1.
     */
    std::map<unsigned int, std::vector<double>> solid_neumann_bcs;
    /**
     * We have to know how many components are expected in traction vector.
     * Although the dimension is specified in the Geometry subsection,
     * this structure is not aware of it.
     * To resolve this issue, we add an additional member, which will be
     * copied from the Geometry section.
     */
    int solid_neumann_bc_dim;
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };

  struct AllParameters : public Simulation,
                         public FluidFESystem,
                         public FluidMaterial,
                         public FluidSolver,
                         public FluidDirichlet,
                         public FluidNeumann,
                         public SpalartAllmarasModel,
                         public SolidFESystem,
                         public SolidMaterial,
                         public SolidSolver,
                         public SolidDirichlet,
                         public SolidNeumann
  {
    AllParameters(const std::string &);
    static void declareParameters(ParameterHandler &);
    void parseParameters(ParameterHandler &);
  };
} // namespace Parameters

#endif

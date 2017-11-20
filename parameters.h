#ifndef PARAMETERS
#define PARAMETERS

#include <deal.II/base/parameter_handler.h>
#include <iostream>
#include <string>
#include <vector>

namespace Parameters
{
  struct Simulation
  {
    int dimension;
    double end_time;
    double time_step;
    double output_interval;
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct FluidFESystem
  {
    unsigned int fluid_degree;
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct FluidMaterial
  {
    double viscosity;
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct FluidSolver
  {
    double grad_div;
    unsigned int fluid_max_iterations;
    double fluid_tolerance;
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct SolidFESystem
  {
    unsigned int solid_degree;
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct SolidMaterial
  {
    std::string solid_type;
    double rho;             //!< density, used by all types.
    double E;               //!< Young's modulus, linear elastic material only.
    double nu;              //!< Poisson's ratio, linear elastic material only.
    std::vector<double> C;  //!< Hyperelastic material constants.
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct SolidSolver
  {
    unsigned int solid_max_iterations; //!< Max number of Newton iterations, hyperelastic only.
    double tol_f;                      //!< Force tolerance
    double tol_d;                      //!< Displacement tolerance, hyperelastic only.
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };

  struct AllParameters: public Simulation,
                        public FluidFESystem,
                        public FluidMaterial,
                        public FluidSolver,
                        public SolidFESystem,
                        public SolidMaterial,
                        public SolidSolver
  {
    AllParameters(const std::string &);
    static void declareParameters(dealii::ParameterHandler &);
    void parseParameters(dealii::ParameterHandler &);
  };
}

#endif

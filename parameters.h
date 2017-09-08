#include <deal.II/base/parameter_handler.h>
#include <string>
#include <vector>
#include <iostream>

namespace IFEM
{
  namespace Parameters
  {
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct Material
    {
      std::string type;
      double rho;
      double E; // Linear elastic material only
      double nu; // Linear elastic material only
      std::vector<double> C; // Hyperelastic material only
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct LinearSolver
    {
      std::string type_lin;
      double tol_lin;
      double max_iterations_lin;
      std::string preconditioner_type;
      double preconditioner_relaxation;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    // NonLinearSolver

    struct Time
    {
      double end_time;
      double delta_t;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct AllParameters : public FESystem, public Material,
      public LinearSolver, public Time
    {
      AllParameters(const std::string &);
      static void declareParameters(dealii::ParameterHandler &prm);
      void parseParameters(dealii::ParameterHandler &prm);
    };
  }
}

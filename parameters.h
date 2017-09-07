#include <deal.II/base/parameter_handler.h>
#include <string>
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

    struct LinearMaterial
    {
      double lambda;
      double mu;
      double rho;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    // HyperelasticMaterial

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

    struct AllParameters : public FESystem, public LinearMaterial,
      public LinearSolver, public Time
    {
      AllParameters(const std::string &);
      static void declareParameters(dealii::ParameterHandler &prm);
      void parseParameters(dealii::ParameterHandler &prm);
    };
    /*
    std::ostream& operator<<(std::ostream&, const FESystem&);
    std::ostream& operator<<(std::ostream&, const LinearMaterial&);
    std::ostream& operator<<(std::ostream&, const LinearSolver&);
    std::ostream& operator<<(std::ostream&, const Time&);
    std::ostream& operator<<(std::ostream&, const AllParameters&);
    */
  }
}

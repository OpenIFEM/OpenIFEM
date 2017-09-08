#include "parameters.h"

namespace IFEM
{
  namespace Parameters
  {
    void FESystem::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "1", dealii::Patterns::Integer(1),
          "Displacement system polynomial order");
        prm.declare_entry("Quadrature order", "2", dealii::Patterns::Integer(2),
          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        this->poly_degree = prm.get_integer("Polynomial degree");
        this->quad_order = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    void Material::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Material type", "LinearElastic",
          dealii::Patterns::Selection("LinearElastic|NeoHookean"), "Type of material");
        prm.declare_entry("Density", "1.0", dealii::Patterns::Double(0.0),
          "Density");
        prm.declare_entry("Young's modulus", "0.0", dealii::Patterns::Double(0.0),
          "Young's modulus, only used by linear elastic materials");
        prm.declare_entry("Poisson's ratio", "0.0", dealii::Patterns::Double(0.0, 0.5),
          "Poisson's ratio, only used by linear elastic materials");
        const char *text =
          "A list of material constants separated by comma, "
          "only used by hyperelastic materials.";
        prm.declare_entry("Hyperelastic parameters", "",
          dealii::Patterns::List(dealii::Patterns::Double()), text);
      }
      prm.leave_subsection();
    }

    void Material::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        this->type = prm.get("Material type");
        this->rho = prm.get_double("Density");
        if (type == "LinearElastic")
        {
          this->E = prm.get_double("Young's modulus");
          this->nu = prm.get_double("Poisson's ratio");
        }
        else if (type == "NeoHookean")
        {
          std::string raw_input = prm.get("Hyperelastic parameters");
          std::vector<std::string> parsed_input =
            dealii::Utilities::split_string_list(raw_input);
          this->C = dealii::Utilities::string_to_double(parsed_input);
        }
      }
      prm.leave_subsection();
    }

    void LinearSolver::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type", "CG", dealii::Patterns::Selection("CG|Direct"),
          "Type of solver used to solve the linear system");
        prm.declare_entry("Residual", "1e-6", dealii::Patterns::Double(0.0),
          "Linear solver residual (scaled by residual norm)");
        prm.declare_entry("Max iteration multiplier", "1", dealii::Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");
        prm.declare_entry("Preconditioner type", "ssor", dealii::Patterns::Selection("jacobi|ssor"),
          "Type of preconditioner");
        prm.declare_entry("Preconditioner relaxation", "0.65", dealii::Patterns::Double(0.0),
          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void LinearSolver::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        this->type_lin = prm.get("Solver type");
        this->tol_lin = prm.get_double("Residual");
        this->max_iterations_lin = prm.get_double("Max iteration multiplier");
        this->preconditioner_type = prm.get("Preconditioner type");
        this->preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

    void Time::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", dealii::Patterns::Double(0.0), "End time");
        prm.declare_entry("Time step size", "0.1", dealii::Patterns::Double(0.0),
          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        this->end_time = prm.get_double("End time");
        this->delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

    AllParameters::AllParameters(const std::string &infile)
    {
      dealii::ParameterHandler prm;
      this->declareParameters(prm);
      prm.parse_input(infile);
      this->parseParameters(prm);
      prm.print_parameters(std::cout, dealii::ParameterHandler::Text);
    }

    void AllParameters::declareParameters(dealii::ParameterHandler &prm)
    {
      FESystem::declareParameters(prm);
      Material::declareParameters(prm);
      LinearSolver::declareParameters(prm);
      Time::declareParameters(prm);
    }

    void AllParameters::parseParameters(dealii::ParameterHandler &prm)
    {
      FESystem::parseParameters(prm);
      Material::parseParameters(prm);
      LinearSolver::parseParameters(prm);
      Time::parseParameters(prm);
    }
  }
}

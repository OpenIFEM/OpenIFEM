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
          "Element polynomial order");
        prm.declare_entry("Quadrature order", "2", dealii::Patterns::Integer(2),
          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void FESystem::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        polyDegree = prm.get_integer("Polynomial degree");
        quadOrder = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    void Geometry::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement", "2", dealii::Patterns::Integer(0),
          "Global refinement level");
        prm.declare_entry("Grid scale", "1e-3", dealii::Patterns::Double(0.0),
          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }

    void Geometry::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        globalRefinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
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
        typeLin = prm.get("Solver type");
        tolLin = prm.get_double("Residual");
        maxItrLin = prm.get_double("Max iteration multiplier");
        typePre = prm.get("Preconditioner type");
        relaxPre = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson", "10",
          dealii::Patterns::Integer(0), "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force", "1.0e-9", dealii::Patterns::Double(0.0),
          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement", "1.0e-6", dealii::Patterns::Double(0.0),
          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        maxItrNL = prm.get_integer("Max iterations Newton-Raphson");
        tolF = prm.get_double("Tolerance force");
        tolU = prm.get_double("Tolerance displacement");
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
        typeMat = prm.get("Material type");
        rho = prm.get_double("Density");
        if (typeMat == "LinearElastic")
        {
          this->E = prm.get_double("Young's modulus");
          this->nu = prm.get_double("Poisson's ratio");
        }
        else if (typeMat == "NeoHookean")
        {
          std::string raw_input = prm.get("Hyperelastic parameters");
          std::vector<std::string> parsed_input =
            dealii::Utilities::split_string_list(raw_input);
          C = dealii::Utilities::string_to_double(parsed_input);
        }
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
        endTime = prm.get_double("End time");
        deltaTime = prm.get_double("Time step size");
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

#include "parameters.h"

namespace IFEM
{
  namespace Parameters
  {
    void FESystem::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          dealii::Patterns::Integer(1),
                          "Element polynomial order");
        prm.declare_entry("Quadrature order",
                          "2",
                          dealii::Patterns::Integer(2),
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
        prm.declare_entry("Dimension",
                          "2",
                          dealii::Patterns::Integer(0),
                          "Dimension of the problem");
        prm.declare_entry("Global refinement",
                          "2",
                          dealii::Patterns::Integer(0),
                          "Global refinement level");
        prm.declare_entry("Grid scale",
                          "1e-3",
                          dealii::Patterns::Double(0.0),
                          "Global grid scaling factor");
      }
      prm.leave_subsection();
    }

    void Geometry::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        dimension = prm.get_integer("Dimension");
        globalRefinement = prm.get_integer("Global refinement");
        scale = prm.get_double("Grid scale");
      }
      prm.leave_subsection();
    }

    void LinearSolver::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "CG",
                          dealii::Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");
        prm.declare_entry("Residual",
                          "1e-6",
                          dealii::Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");
        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          dealii::Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");
        prm.declare_entry("Preconditioner type",
                          "ssor",
                          dealii::Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");
        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          dealii::Patterns::Double(0.0),
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
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          dealii::Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          dealii::Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          dealii::Patterns::Double(0.0),
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
        prm.declare_entry(
          "Material type",
          "LinearElastic",
          dealii::Patterns::Selection("LinearElastic|NeoHookean"),
          "Type of material");
        prm.declare_entry(
          "Density", "1.0", dealii::Patterns::Double(0.0), "Density");
        prm.declare_entry(
          "Young's modulus",
          "0.0",
          dealii::Patterns::Double(0.0),
          "Young's modulus, only used by linear elastic materials");
        prm.declare_entry(
          "Poisson's ratio",
          "0.0",
          dealii::Patterns::Double(0.0, 0.5),
          "Poisson's ratio, only used by linear elastic materials");
        const char *text = "A list of material constants separated by comma, "
                           "only used by hyperelastic materials. "
                           "NeoHookean requires C1, kappa;";
        prm.declare_entry("Hyperelastic parameters",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Double()),
                          text);
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
            E = prm.get_double("Young's modulus");
            nu = prm.get_double("Poisson's ratio");
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
        prm.declare_entry(
          "End time", "1", dealii::Patterns::Double(0.0), "End time");
        prm.declare_entry("Time step size",
                          "0.1",
                          dealii::Patterns::Double(0.0),
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

    void BoundaryConditions::declareParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        prm.declare_entry("Apply displacement",
                          "true",
                          dealii::Patterns::Bool(),
                          "Apply displacement boundary conditions");
        prm.declare_entry("Displacement IDs",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Integer()),
                          "IDs of boundaries to specify displacements");
        prm.declare_entry("Displacement flags",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Integer()),
                          "Displacement components to constrain");
        prm.declare_entry("Displacement values",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Double()),
                          "Displacement values");
        prm.declare_entry("Apply pressure",
                          "true",
                          dealii::Patterns::Bool(),
                          "Apply external pressure");
        prm.declare_entry("Pressure IDs",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Integer()),
                          "IDs of boundaries to apply pressure on");
        prm.declare_entry("Pressure values",
                          "",
                          dealii::Patterns::List(dealii::Patterns::Double()),
                          "Values of external pressure");
      }
      prm.leave_subsection();
    }

    void BoundaryConditions::parseParameters(dealii::ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        applyDisplacement = prm.get_bool("Apply displacement");
        if (applyDisplacement)
          {
            std::string raw_input = prm.get("Displacement IDs");
            std::vector<std::string> parsed_input =
              dealii::Utilities::split_string_list(raw_input);
            displacementIDs = dealii::Utilities::string_to_int(parsed_input);
            AssertThrow(
              !displacementIDs.empty(),
              dealii::ExcMessage("Displacement IDs should be non-empty!"));

            raw_input = prm.get("Displacement flags");
            parsed_input = dealii::Utilities::split_string_list(raw_input);
            displacementFlags = dealii::Utilities::string_to_int(parsed_input);

            raw_input = prm.get("Displacement values");
            parsed_input = dealii::Utilities::split_string_list(raw_input);
            displacementValues =
              dealii::Utilities::string_to_double(parsed_input);
          }

        applyPressure = prm.get_bool("Apply pressure");
        if (applyPressure)
          {
            std::string raw_input = prm.get("Pressure IDs");
            std::vector<std::string> parsed_input =
              dealii::Utilities::split_string_list(raw_input);
            pressureIDs = dealii::Utilities::string_to_int(parsed_input);
            AssertThrow(
              !pressureIDs.empty(),
              dealii::ExcMessage("Pressure IDs should be non-empty!"));

            raw_input = prm.get("Pressure values");
            parsed_input = dealii::Utilities::split_string_list(raw_input);
            pressureValues = dealii::Utilities::string_to_double(parsed_input);
            AssertThrow(
              pressureValues.size() == pressureIDs.size(),
              dealii::ExcMessage(
                "Pressure values should have the same size as pressure IDs!"));
          }
      }
      prm.leave_subsection();
    }

    AllParameters::AllParameters(const std::string &infile)
    {
      dealii::ParameterHandler prm;
      declareParameters(prm);
      prm.parse_input(infile);
      parseParameters(prm);
      prm.print_parameters(std::cout, dealii::ParameterHandler::Text);
    }

    void AllParameters::declareParameters(dealii::ParameterHandler &prm)
    {
      FESystem::declareParameters(prm);
      Geometry::declareParameters(prm);
      LinearSolver::declareParameters(prm);
      NonlinearSolver::declareParameters(prm);
      Material::declareParameters(prm);
      Time::declareParameters(prm);
      BoundaryConditions::declareParameters(prm);
    }

    void AllParameters::parseParameters(dealii::ParameterHandler &prm)
    {
      FESystem::parseParameters(prm);
      Geometry::parseParameters(prm);
      LinearSolver::parseParameters(prm);
      NonlinearSolver::parseParameters(prm);
      Material::parseParameters(prm);
      Time::parseParameters(prm);
      BoundaryConditions::parseParameters(prm);

      AssertThrow(
        displacementFlags.size() == dimension * displacementIDs.size(),
        dealii::ExcMessage("Size of displacement flags should equal dimension "
                           "times size of displacement IDs!"));
      AssertThrow(
        displacementValues.size() == dimension * displacementIDs.size(),
        dealii::ExcMessage("Size of displacement values should equal dimension "
                           "times size of displacement IDs!"));
    }
  }
}

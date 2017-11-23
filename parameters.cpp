#include "parameters.h"

namespace Parameters
{
  void Simulation::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Simulation");
    {
      prm.declare_entry("Dimension",
                        "2",
                        dealii::Patterns::Integer(0),
                        "Dimension of the problem");
      prm.declare_entry(
        "End time", "1", dealii::Patterns::Double(0.0), "End time");
      prm.declare_entry("Time step size",
                        "0.1",
                        dealii::Patterns::Double(0.0),
                        "Time step size");
      prm.declare_entry("Output interval",
                        "0.1",
                        dealii::Patterns::Double(0.0),
                        "Output interval");
    }
    prm.leave_subsection();
  }

  void Simulation::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Simulation");
    {
      dimension = prm.get_integer("Dimension");
      end_time = prm.get_double("End time");
      time_step = prm.get_double("Time step size");
      output_interval = prm.get_double("Output interval");
    }
    prm.leave_subsection();
  }

  void FluidFESystem::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      prm.declare_entry("Degree",
                        "1",
                        dealii::Patterns::Integer(1),
                        "Pressure element polynomial order");
    }
    prm.leave_subsection();
  }

  void FluidFESystem::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      fluid_degree = prm.get_integer("Degree");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      prm.declare_entry("Viscosity",
                        "1e-3",
                        dealii::Patterns::Double(0.0),
                        "Kinematic viscosity");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      viscosity = prm.get_double("Viscosity");
    }
    prm.leave_subsection();
  }

  void FluidSolver::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid solver control");
    {
      prm.declare_entry("Grad-Div stabilization",
                        "1.0",
                        dealii::Patterns::Double(1.0),
                        "Grad-Div stabilization");
      prm.declare_entry("Max Newton iterations",
                        "8",
                        dealii::Patterns::Integer(8),
                        "Maximum number of Newton iterations");
      prm.declare_entry(
        "Nonlinear system tolerance",
        "1e-10",
        dealii::Patterns::Double(0.0),
        "The absolute tolerance of the nonlinear system residual");
    }
    prm.leave_subsection();
  }

  void FluidSolver::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid solver control");
    {
      grad_div = prm.get_double("Grad-Div stabilization");
      fluid_max_iterations = prm.get_integer("Max Newton iterations");
      fluid_tolerance = prm.get_double("Nonlinear system tolerance");
    }
    prm.leave_subsection();
  }

  void SolidFESystem::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid finite element system");
    {
      prm.declare_entry("Degree",
                        "1",
                        dealii::Patterns::Integer(1),
                        "Polynomial degree of solid element");
    }
    prm.leave_subsection();
  }

  void SolidFESystem::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid finite element system");
    {
      solid_degree = prm.get_integer("Degree");
    }
    prm.leave_subsection();
  }

  void SolidMaterial::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid material properties");
    {
      prm.declare_entry("Solid type",
                        "LinearElastic",
                        dealii::Patterns::Selection("LinearElastic|NeoHookean"),
                        "Type of solid material");
      prm.declare_entry(
        "Density", "1.0", dealii::Patterns::Double(0.0), "Density");
      prm.declare_entry("Young's modulus",
                        "0.0",
                        dealii::Patterns::Double(0.0),
                        "Young's modulus, only used by linear elastic solver");
      prm.declare_entry("Poisson's ratio",
                        "0.0",
                        dealii::Patterns::Double(0.0, 0.5),
                        "Poisson's ratio, only used by linear elastic solver");
      const char *text = "A list of material constants separated by comma, "
                         "only used by hyperelastic materials."
                         "NeoHookean requires C1, kappa;";
      prm.declare_entry("Hyperelastic parameters",
                        "",
                        dealii::Patterns::List(dealii::Patterns::Double()),
                        text);
    }
    prm.leave_subsection();
  }

  void SolidMaterial::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid material properties");
    {
      solid_type = prm.get("Solid type");
      rho = prm.get_double("Density");
      if (solid_type == "LinearElastic")
        {
          E = prm.get_double("Young's modulus");
          nu = prm.get_double("Poisson's ratio");
        }
      else if (solid_type == "NeoHookean")
        {
          std::string raw_input = prm.get("Hyperelastic parameters");
          std::vector<std::string> parsed_input =
            dealii::Utilities::split_string_list(raw_input);
          C = dealii::Utilities::string_to_double(parsed_input);
        }
    }
    prm.leave_subsection();
  }

  void SolidSolver::declareParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid solver control");
    {
      prm.declare_entry("Damping",
                        "0.0",
                        dealii::Patterns::Double(0.0),
                        "The artifical damping in Newmark-beta method");
      prm.declare_entry("Max Newton iterations",
                        "8",
                        dealii::Patterns::Integer(8),
                        "Maximum number of Newton iterations");
      prm.declare_entry(
        "Displacement tolerance",
        "1e-10",
        dealii::Patterns::Double(0.0),
        "The tolerance of the displacement increment at each iteration");
      prm.declare_entry("Force tolerance",
                        "1e-10",
                        dealii::Patterns::Double(0.0),
                        "The tolerance of the force equilibrium");
    }
    prm.leave_subsection();
  }

  void SolidSolver::parseParameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Solid solver control");
    {
      damping = prm.get_double("Damping");
      solid_max_iterations = prm.get_integer("Max Newton iterations");
      tol_d = prm.get_double("Displacement tolerance");
      tol_f = prm.get_double("Force tolerance");
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
    Simulation::declareParameters(prm);
    FluidFESystem::declareParameters(prm);
    FluidMaterial::declareParameters(prm);
    FluidSolver::declareParameters(prm);
    SolidFESystem::declareParameters(prm);
    SolidMaterial::declareParameters(prm);
    SolidSolver::declareParameters(prm);
  }

  void AllParameters::parseParameters(dealii::ParameterHandler &prm)
  {
    Simulation::parseParameters(prm);
    FluidFESystem::parseParameters(prm);
    FluidMaterial::parseParameters(prm);
    FluidSolver::parseParameters(prm);
    SolidFESystem::parseParameters(prm);
    SolidMaterial::parseParameters(prm);
    SolidSolver::parseParameters(prm);
  }
}

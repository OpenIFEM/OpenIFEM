#include "parameters.h"

namespace Parameters
{
  using namespace dealii;

  void Simulation::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Simulation");
    {
      prm.declare_entry(
        "Dimension", "2", Patterns::Integer(2), "Dimension of the problem");
      prm.declare_entry("End time", "1", Patterns::Double(0.0), "End time");
      prm.declare_entry(
        "Time step size", "0.1", Patterns::Double(0.0), "Time step size");
      prm.declare_entry(
        "Output interval", "0.1", Patterns::Double(0.0), "Output interval");
    }
    prm.leave_subsection();
  }

  void Simulation::parseParameters(ParameterHandler &prm)
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

  void FluidFESystem::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      prm.declare_entry("Degree",
                        "1",
                        Patterns::Integer(1),
                        "Pressure element polynomial order");
    }
    prm.leave_subsection();
  }

  void FluidFESystem::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      fluid_degree = prm.get_integer("Degree");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      prm.declare_entry(
        "Viscosity", "1e-3", Patterns::Double(0.0), "Kinematic viscosity");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      viscosity = prm.get_double("Viscosity");
    }
    prm.leave_subsection();
  }

  void FluidSolver::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid solver control");
    {
      prm.declare_entry("Grad-Div stabilization",
                        "1.0",
                        Patterns::Double(0.0),
                        "Grad-Div stabilization");
      prm.declare_entry("Max Newton iterations",
                        "8",
                        Patterns::Integer(1),
                        "Maximum number of Newton iterations");
      prm.declare_entry(
        "Nonlinear system tolerance",
        "1e-10",
        Patterns::Double(0.0),
        "The absolute tolerance of the nonlinear system residual");
    }
    prm.leave_subsection();
  }

  void FluidSolver::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid solver control");
    {
      grad_div = prm.get_double("Grad-Div stabilization");
      fluid_max_iterations = prm.get_integer("Max Newton iterations");
      fluid_tolerance = prm.get_double("Nonlinear system tolerance");
    }
    prm.leave_subsection();
  }

  void FluidDirichlet::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid Dirichlet BCs");
    {
      prm.declare_entry("Number of Dirichlet BCs",
                        "1",
                        Patterns::Integer(),
                        "Number of boundaries with Dirichlet BCs");
      prm.declare_entry("Dirichlet boundary id",
                        "",
                        Patterns::List(dealii::Patterns::Integer()),
                        "Ids of the boundaries with Dirichlet BCs");
      prm.declare_entry("Dirichlet boundary components",
                        "",
                        Patterns::List(dealii::Patterns::Integer(1, 7)),
                        "Boundary components to constrain");
      prm.declare_entry("Dirichlet boundary values",
                        "",
                        Patterns::List(dealii::Patterns::Double()),
                        "Boundary values to constrain");
    }
    prm.leave_subsection();
  }

  void FluidDirichlet::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid Dirichlet BCs");
    {
      n_fluid_dirichlet_bcs = prm.get_integer("Number of Dirichlet BCs");
      std::string raw_input = prm.get("Dirichlet boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      AssertThrow(ids.size() == n_fluid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Dirichlet boundary components");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<int> components = Utilities::string_to_int(parsed_input);
      AssertThrow(components.size() == n_fluid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary components!"));
      raw_input = prm.get("Dirichlet boundary values");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      // The size of values should be exact the same as the number of
      // the given boundary values.
      unsigned int n = 0;
      for (unsigned int i = 0; i < n_fluid_dirichlet_bcs; ++i)
        {
          auto flag = components[i];
          std::vector<double> value;
          AssertThrow(n < values.size(),
                      ExcMessage("Inconsistent boundary values!"));
          // 1-x, 2-y, 3-xy, 4-z, 5-xz, 6-yz, 7-xyz
          if (flag == 1 || flag == 2 || flag == 4)
            {
              value.push_back(values[n]);
              n += 1;
            }
          else if (flag == 3 || flag == 5 || flag == 6)
            {
              value.push_back(values[n]);
              value.push_back(values[n + 1]);
              n += 2;
            }
          else
            {
              value.push_back(values[n]);
              value.push_back(values[n + 1]);
              value.push_back(values[n + 2]);
              n += 3;
            }
          fluid_dirichlet_bcs[ids[i]] = {components[i], value};
        }
      AssertThrow(n == values.size(),
                  ExcMessage("Inconsistent boundary values!"));
    }
    prm.leave_subsection();
  }

  void FluidNeumann::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid Neumann BCs");
    {
      prm.declare_entry("Number of Neumann BCs",
                        "1",
                        Patterns::Integer(),
                        "Number of boundaries with Neumann BCs");
      prm.declare_entry("Neumann boundary id",
                        "",
                        Patterns::List(dealii::Patterns::Integer()),
                        "Ids of the boundaries with Neumann BCs");
      prm.declare_entry("Neumann boundary values",
                        "",
                        Patterns::List(dealii::Patterns::Double()),
                        "Boundary values to specify");
    }
    prm.leave_subsection();
  }

  void FluidNeumann::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid Neumann BCs");
    {
      n_fluid_neumann_bcs = prm.get_integer("Number of Neumann BCs");
      std::string raw_input = prm.get("Neumann boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      AssertThrow(ids.size() == n_fluid_neumann_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Neumann boundary values");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      // The size of values should be exact the same as the number of
      // the given boundary values.
      AssertThrow(values.size() == n_fluid_neumann_bcs,
                  ExcMessage("Inconsistent boundary values!"));
      for (unsigned int i = 0; i < n_fluid_neumann_bcs; ++i)
        {
          fluid_neumann_bcs[ids[i]] = values[i];
        }
    }
    prm.leave_subsection();
  }

  void SolidFESystem::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid finite element system");
    {
      prm.declare_entry("Degree",
                        "1",
                        Patterns::Integer(1),
                        "Polynomial degree of solid element");
    }
    prm.leave_subsection();
  }

  void SolidFESystem::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid finite element system");
    {
      solid_degree = prm.get_integer("Degree");
    }
    prm.leave_subsection();
  }

  void SolidMaterial::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid material properties");
    {
      prm.declare_entry("Solid type",
                        "LinearElastic",
                        Patterns::Selection("LinearElastic|NeoHookean"),
                        "Type of solid material");
      prm.declare_entry("Density", "1.0", Patterns::Double(0.0), "Density");
      prm.declare_entry("Young's modulus",
                        "0.0",
                        Patterns::Double(0.0),
                        "Young's modulus, only used by linear elastic solver");
      prm.declare_entry("Poisson's ratio",
                        "0.0",
                        Patterns::Double(0.0, 0.5),
                        "Poisson's ratio, only used by linear elastic solver");
      const char *text = "A list of material constants separated by comma, "
                         "only used by hyperelastic materials."
                         "NeoHookean requires C1, kappa;";
      prm.declare_entry("Hyperelastic parameters",
                        "",
                        Patterns::List(dealii::Patterns::Double()),
                        text);
    }
    prm.leave_subsection();
  }

  void SolidMaterial::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid material properties");
    {
      solid_type = prm.get("Solid type");
      rho = prm.get_double("Density");
      E = prm.get_double("Young's modulus");
      nu = prm.get_double("Poisson's ratio");
      std::string raw_input = prm.get("Hyperelastic parameters");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      C = Utilities::string_to_double(parsed_input);
    }
    prm.leave_subsection();
  }

  void SolidSolver::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid solver control");
    {
      prm.declare_entry("Damping",
                        "0.0",
                        Patterns::Double(0.0),
                        "The artifical damping in Newmark-beta method");
      prm.declare_entry("Max Newton iterations",
                        "8",
                        Patterns::Integer(1),
                        "Maximum number of Newton iterations");
      prm.declare_entry(
        "Displacement tolerance",
        "1e-10",
        Patterns::Double(0.0),
        "The tolerance of the displacement increment at each iteration");
      prm.declare_entry("Force tolerance",
                        "1e-10",
                        Patterns::Double(0.0),
                        "The tolerance of the force equilibrium");
    }
    prm.leave_subsection();
  }

  void SolidSolver::parseParameters(ParameterHandler &prm)
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

  void SolidDirichlet::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid Dirichlet BCs");
    {
      prm.declare_entry("Number of Dirichlet BCs",
                        "1",
                        Patterns::Integer(),
                        "Number of boundaries with Dirichlet BCs");
      prm.declare_entry("Dirichlet boundary id",
                        "",
                        Patterns::List(dealii::Patterns::Integer()),
                        "Ids of the boundaries with Dirichlet BCs");
      prm.declare_entry("Dirichlet boundary components",
                        "",
                        Patterns::List(dealii::Patterns::Integer(1, 7)),
                        "Boundary components to constrain");
    }
    prm.leave_subsection();
  }

  void SolidDirichlet::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid Dirichlet BCs");
    {
      n_solid_dirichlet_bcs = prm.get_integer("Number of Dirichlet BCs");
      std::string raw_input = prm.get("Dirichlet boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      AssertThrow(ids.size() == n_solid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Dirichlet boundary components");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<int> components = Utilities::string_to_int(parsed_input);
      AssertThrow(components.size() == n_solid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary components!"));
      for (unsigned int i = 0; i < n_solid_dirichlet_bcs; ++i)
        {
          solid_dirichlet_bcs[ids[i]] = components[i];
        }
    }
    prm.leave_subsection();
  }

  void SolidNeumann::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid Neumann BCs");
    {
      prm.declare_entry("Number of Neumann BCs",
                        "1",
                        Patterns::Integer(0),
                        "Number of boundaries with Neumann BCs");
      prm.declare_entry("Neumann boundary id",
                        "",
                        Patterns::List(dealii::Patterns::Integer(0)),
                        "Ids of the boundaries with Neumann BCs");
      prm.declare_entry("Neumann boundary type",
                        "Traction",
                        Patterns::Selection("Traction|Pressure"),
                        "Type of Neumann BC");
      prm.declare_entry("Neumann boundary values",
                        "",
                        Patterns::List(dealii::Patterns::Double()),
                        "Neumann boundary values");
    }
    prm.leave_subsection();
  }

  void SolidNeumann::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid Neumann BCs");
    {
      n_solid_neumann_bcs = prm.get_integer("Number of Neumann BCs");
      std::string raw_input = prm.get("Neumann boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      AssertThrow(ids.size() == n_solid_neumann_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      solid_neumann_bc_type = prm.get("Neumann boundary type");
      unsigned int tmp =
        (solid_neumann_bc_type == "Traction" ? solid_neumann_bc_dim : 1);
      raw_input = prm.get("Neumann boundary values");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      AssertThrow(values.size() == tmp * n_solid_neumann_bcs,
                  ExcMessage("Inconsistent boundary values!"));
      for (unsigned int i = 0; i < n_solid_neumann_bcs; ++i)
        {
          std::vector<double> value;
          for (unsigned int j = 0; j < tmp; ++j)
            {
              value.push_back(values[i * tmp + j]);
            }
          solid_neumann_bcs[ids[i]] = value;
        }
    }
    prm.leave_subsection();
  }

  AllParameters::AllParameters(const std::string &infile)
  {
    ParameterHandler prm;
    declareParameters(prm);
    prm.parse_input(infile);
    parseParameters(prm);
    prm.print_parameters(std::cout, ParameterHandler::Text);
  }

  void AllParameters::declareParameters(ParameterHandler &prm)
  {
    Simulation::declareParameters(prm);
    FluidFESystem::declareParameters(prm);
    FluidMaterial::declareParameters(prm);
    FluidSolver::declareParameters(prm);
    FluidDirichlet::declareParameters(prm);
    FluidNeumann::declareParameters(prm);
    SolidFESystem::declareParameters(prm);
    SolidMaterial::declareParameters(prm);
    SolidSolver::declareParameters(prm);
    SolidDirichlet::declareParameters(prm);
    SolidNeumann::declareParameters(prm);
  }

  void AllParameters::parseParameters(ParameterHandler &prm)
  {
    Simulation::parseParameters(prm);
    FluidFESystem::parseParameters(prm);
    FluidMaterial::parseParameters(prm);
    FluidSolver::parseParameters(prm);
    FluidDirichlet::parseParameters(prm);
    FluidNeumann::parseParameters(prm);
    SolidFESystem::parseParameters(prm);
    SolidMaterial::parseParameters(prm);
    SolidSolver::parseParameters(prm);
    SolidDirichlet::parseParameters(prm);
    // Set the dummy member in Solid Neumann BCs subsection
    solid_neumann_bc_dim = dimension;
    SolidNeumann::parseParameters(prm);
  }
}

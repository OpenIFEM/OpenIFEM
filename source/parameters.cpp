#include "parameters.h"

namespace Parameters
{
  using namespace dealii;

  void Simulation::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Simulation");
    {
      prm.declare_entry("Simulation type",
                        "FSI",
                        Patterns::Selection("FSI|Fluid|Solid"),
                        "Simulation type");
      prm.declare_entry(
        "Dimension", "2", Patterns::Integer(2), "Dimension of the problem");
      prm.declare_entry("Global refinements",
                        "",
                        Patterns::List(dealii::Patterns::Integer()),
                        "Level of global refinements");
      prm.declare_entry("End time", "1.0", Patterns::Double(0.0), "End time");
      prm.declare_entry(
        "Time step size", "1.0", Patterns::Double(0.0), "Time step size");
      prm.declare_entry(
        "Output interval", "1.0", Patterns::Double(0.0), "Output interval");
      prm.declare_entry("Refinement interval",
                        "1.0",
                        Patterns::Double(0.0),
                        "Refinement interval");
      prm.declare_entry(
        "Save interval", "1.0", Patterns::Double(0.0), "Save interval");
      prm.declare_entry(
        "Gravity",
        "",
        Patterns::List(dealii::Patterns::Double()),
        "Gravity acceleration that applies to both fluid and solid");
    }
    prm.leave_subsection();
  }

  void Simulation::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Simulation");
    {
      simulation_type = prm.get("Simulation type");
      dimension = prm.get_integer("Dimension");
      std::string raw_input = prm.get("Global refinements");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      global_refinements = Utilities::string_to_int(parsed_input);
      AssertThrow(static_cast<int>(global_refinements.size()) == 2,
                  ExcMessage("Incorrect dimension of global_refinements!"));
      end_time = prm.get_double("End time");
      time_step = prm.get_double("Time step size");
      output_interval = prm.get_double("Output interval");
      refinement_interval = prm.get_double("Refinement interval");
      save_interval = prm.get_double("Save interval");
      raw_input = prm.get("Gravity");
      parsed_input = Utilities::split_string_list(raw_input);
      gravity = Utilities::string_to_double(parsed_input);
      AssertThrow(static_cast<int>(gravity.size()) == dimension,
                  ExcMessage("Inconsistent dimension of gravity!"));
    }
    prm.leave_subsection();
  }

  void FluidFESystem::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      prm.declare_entry("Pressure degree",
                        "1",
                        Patterns::Integer(1),
                        "Pressure element polynomial order");
      prm.declare_entry("Velocity degree",
                        "2",
                        Patterns::Integer(1),
                        "Velocity element polynomial order");
    }
    prm.leave_subsection();
  }

  void FluidFESystem::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid finite element system");
    {
      fluid_pressure_degree = prm.get_integer("Pressure degree");
      fluid_velocity_degree = prm.get_integer("Velocity degree");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      prm.declare_entry("Dynamic viscosity",
                        "1e-3",
                        Patterns::Double(0.0),
                        "Dynamic viscosity");
      prm.declare_entry(
        "Fluid density", "1.0", Patterns::Double(0.0), "Dynamic viscosity");
    }
    prm.leave_subsection();
  }

  void FluidMaterial::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid material properties");
    {
      viscosity = prm.get_double("Dynamic viscosity");
      fluid_rho = prm.get_double("Fluid density");
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
      prm.declare_entry("Use hard-coded boundary values",
                        "0",
                        Patterns::Integer(),
                        "Use hard-coded boundary values or the input ones");
      prm.declare_entry("Number of Dirichlet BCs",
                        "0",
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
      use_hard_coded_values = prm.get_integer("Use hard-coded boundary values");
      n_fluid_dirichlet_bcs = prm.get_integer("Number of Dirichlet BCs");
      std::string raw_input = prm.get("Dirichlet boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      AssertThrow(!n_fluid_dirichlet_bcs || ids.size() == n_fluid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Dirichlet boundary components");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<int> components = Utilities::string_to_int(parsed_input);
      AssertThrow(!n_fluid_dirichlet_bcs ||
                    components.size() == n_fluid_dirichlet_bcs,
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
                        "0",
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
      // Assert only when the user wants to impose Neumann BC
      AssertThrow(!n_fluid_neumann_bcs || ids.size() == n_fluid_neumann_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Neumann boundary values");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      // The size of values should be exactly the same as the number of
      // the given boundary values.
      AssertThrow(!n_fluid_neumann_bcs || values.size() == n_fluid_neumann_bcs,
                  ExcMessage("Inconsistent boundary values!"));
      for (unsigned int i = 0; i < n_fluid_neumann_bcs; ++i)
        {
          fluid_neumann_bcs[ids[i]] = values[i];
        }
    }
    prm.leave_subsection();
  }

  void SpalartAllmarasModel::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Spalart Allmaras model");
    {
      prm.declare_entry(
        "Number of S-A model BCs",
        "0",
        Patterns::Integer(),
        "Number of boundaries with Spalart-Allmaras turbulence model BCs");
      prm.declare_entry(
        "S-A model boundary id",
        "",
        Patterns::List(dealii::Patterns::Integer()),
        "Ids of the boundaries with Spalart-Allmaras turbulence model BCs");
      prm.declare_entry("S-A model boundary types",
                        "",
                        Patterns::List(dealii::Patterns::Integer(0, 1)),
                        "Boundary condition types to specify. 0 for walls and "
                        "1 for inflow condition");
      prm.declare_entry("Initial condition coefficient",
                        "0.0",
                        Patterns::Double(0.0),
                        "Coefficient of the laminar viscosity for the initial "
                        "condition of S-A model");
    }
    prm.leave_subsection();
  }

  void SpalartAllmarasModel::parseParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Spalart Allmaras model");
    {
      n_spalart_allmaras_model_bcs = prm.get_integer("Number of S-A model BCs");
      std::string raw_input = prm.get("S-A model boundary id");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      std::vector<int> ids = Utilities::string_to_int(parsed_input);
      // Assert only when the user wants to impose Neumann BC
      AssertThrow(!n_spalart_allmaras_model_bcs ||
                    ids.size() == n_spalart_allmaras_model_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("S-A model boundary types");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      // The size of values should be exactly the same as the number of
      // the given boundary values.
      AssertThrow(!n_spalart_allmaras_model_bcs ||
                    values.size() == n_spalart_allmaras_model_bcs,
                  ExcMessage("Inconsistent boundary values!"));
      for (unsigned int i = 0; i < n_spalart_allmaras_model_bcs; ++i)
        {
          spalart_allmaras_model_bcs[ids[i]] = values[i];
        }
      spalart_allmaras_initial_condition_coefficient =
        prm.get_double("Initial condition coefficient");
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
      prm.declare_entry("Number of solid parts",
                        "1",
                        Patterns::Integer(0),
                        "Number of different materials used in solid");
      prm.declare_entry("Solid density",
                        "1.0",
                        Patterns::List(dealii::Patterns::Double(0)),
                        "Solid density");
      prm.declare_entry("Young's modulus",
                        "0.0",
                        Patterns::List(dealii::Patterns::Double(0)),
                        "Young's modulus, only used by linear elastic solver");
      prm.declare_entry("Poisson's ratio",
                        "0.0",
                        Patterns::List(dealii::Patterns::Double(0, 0.5)),
                        "Poisson's ratio, only used by linear elastic solver");
      prm.declare_entry("Viscosity",
                        "0.0",
                        Patterns::List(dealii::Patterns::Double(0)),
                        "Viscous damping, only used by linear elastic solver");
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
      n_solid_parts = prm.get_integer("Number of solid parts");
      AssertThrow(n_solid_parts > 0,
                  ExcMessage("Number of solid part less than 1!"));
      E.resize(n_solid_parts, 0);
      nu.resize(n_solid_parts, 0);
      C.resize(n_solid_parts);
      solid_rho = prm.get_double("Solid density");
      std::string raw_input = prm.get("Young's modulus");
      std::vector<std::string> parsed_input =
        Utilities::split_string_list(raw_input);
      E = Utilities::string_to_double(parsed_input);
      AssertThrow(E.size() == n_solid_parts,
                  ExcMessage("Inconsistent Youngs' moduli!"));
      raw_input = prm.get("Poisson's ratio");
      parsed_input = Utilities::split_string_list(raw_input);
      nu = Utilities::string_to_double(parsed_input);
      AssertThrow(nu.size() == n_solid_parts,
                  ExcMessage("Inconsistent Poisson's ratios!"));
      raw_input = prm.get("Viscosity");
      parsed_input = Utilities::split_string_list(raw_input);
      eta = Utilities::string_to_double(parsed_input);
      AssertThrow(eta.size() == n_solid_parts,
                  ExcMessage("Inconsistent viscosity!"));
      raw_input = prm.get("Hyperelastic parameters");
      parsed_input = Utilities::split_string_list(raw_input);
      // declare the size for each vector defining one hyperelastic material
      unsigned int size_per_material = 1;
      if (solid_type == "NeoHookean")
        // Only NeoHookean for now
        size_per_material = 2;
      AssertThrow(parsed_input.size() >= size_per_material * n_solid_parts,
                  ExcMessage("Insufficient material properties input!"));
      std::vector<double> tmp_C = Utilities::string_to_double(parsed_input);
      for (unsigned int i = 0; i < n_solid_parts; ++i)
        {
          C[i].resize(size_per_material, 0);
          for (unsigned int j = 0; j < size_per_material; ++j)
            C[i][j] = tmp_C[i * size_per_material + j];
        }
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
      prm.declare_entry(
        "Contact force multiplier",
        "1e8",
        Patterns::Double(0.0),
        "Multiplier of the penetration distance to compute contact force.");
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
      contact_force_multiplier = prm.get_double("Contact force multiplier");
    }
    prm.leave_subsection();
  }

  void SolidDirichlet::declareParameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solid Dirichlet BCs");
    {
      prm.declare_entry("Number of Dirichlet BCs",
                        "0",
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
      AssertThrow(!n_solid_dirichlet_bcs || ids.size() == n_solid_dirichlet_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      raw_input = prm.get("Dirichlet boundary components");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<int> components = Utilities::string_to_int(parsed_input);
      AssertThrow(!n_solid_dirichlet_bcs ||
                    components.size() == n_solid_dirichlet_bcs,
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
                        "0",
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
      // Assert only when the user wants to impose Neumann BC
      AssertThrow(!n_solid_neumann_bcs || ids.size() == n_solid_neumann_bcs,
                  ExcMessage("Inconsistent boundary ids!"));
      solid_neumann_bc_type = prm.get("Neumann boundary type");
      unsigned int tmp =
        (solid_neumann_bc_type == "Traction" ? solid_neumann_bc_dim : 1);
      raw_input = prm.get("Neumann boundary values");
      parsed_input = Utilities::split_string_list(raw_input);
      std::vector<double> values = Utilities::string_to_double(parsed_input);
      AssertThrow(!n_solid_neumann_bcs ||
                    values.size() == tmp * n_solid_neumann_bcs,
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
  }

  void AllParameters::declareParameters(ParameterHandler &prm)
  {
    Simulation::declareParameters(prm);
    FluidFESystem::declareParameters(prm);
    FluidMaterial::declareParameters(prm);
    FluidSolver::declareParameters(prm);
    FluidDirichlet::declareParameters(prm);
    FluidNeumann::declareParameters(prm);
    SpalartAllmarasModel::declareParameters(prm);
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
    SpalartAllmarasModel::parseParameters(prm);
    SolidFESystem::parseParameters(prm);
    SolidMaterial::parseParameters(prm);
    SolidSolver::parseParameters(prm);
    SolidDirichlet::parseParameters(prm);
    // Set the dummy member in Solid Neumann BCs subsection
    solid_neumann_bc_dim = dimension;
    SolidNeumann::parseParameters(prm);
  }
} // namespace Parameters

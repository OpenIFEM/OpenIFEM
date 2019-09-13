#include "shell_fsi.h"

namespace MPI
{
  ShellFSI::ShellFSI(Fluid::MPI::FluidSolver<3> &f,
                     Solid::MPI::SharedSolidSolver<3> &s,
                     Solid::ShellSolidSolver &ss,
                     const Parameters::AllParameters &p,
                     bool use_dirichlet_bc)
    : FSI<3>(f, s, p, use_dirichlet_bc), shell_solver(ss)
  {
    Assert(use_dirichlet_bc == false,
           ExcMessage("ShellFSI cannot use dirichlet BC for fluid!"));
  }

  ShellFSI::~ShellFSI() { timer.print_summary(); }

  void ShellFSI::run() {}

  void ShellFSI::update_solid_box() {}

  void ShellFSI::construct_interpolatioin_rule()
  {
    const std::vector<Point<2>> &unit_points =
      shell_solver.fe.get_unit_support_points();
    Quadrature<2> dummy_q(unit_points);
    MappingQGeneric<2, 3> mapping(1);
    FEValues<2, 3> dummy_fe_values(
      mapping, shell_solver.fe, dummy_q, update_quadrature_points);
    cell_interpolators.initialize(shell_solver.triangulation.begin_active(),
                                  shell_solver.triangulation.end(),
                                  unit_points.size());
    for (auto cell = shell_solver.triangulation.begin_active();
         cell != shell_solver.triangulation.end();
         ++cell)
      {
        std::vector<std::shared_ptr<CellInterpolators>> p =
          cell_interpolators.get_data(cell);
        dummy_fe_values.reinit(cell);
        auto support_points = dummy_fe_values.get_quadrature_points();
        for (unsigned int i = 0; i < unit_points.size(); ++i)
          {
            p[i]->interpolator =
              std::make_unique<Utils::GridInterpolator<3, Vector<double>>>(
                Utils::GridInterpolator<3, Vector<double>>(
                  solid_solver.dof_handler, support_points[i]));
          }
      }
  }

  void ShellFSI::interpolate_to_shell()
  {
    Vector<double> localized_current_displacement(
      solid_solver.current_displacement);
    Vector<double> localized_current_velocity(solid_solver.current_velocity);
    AffineConstraints<double> velocity_constraints, displacement_constraints;
    velocity_constraints.clear();
    displacement_constraints.clear();

    const std::vector<Point<2>> &unit_points =
      shell_solver.fe.get_unit_support_points();
    Quadrature<2> dummy_q(unit_points);
    MappingQGeneric<2, 3> mapping(1);
    FEValues<2, 3> dummy_fe_values(
      mapping, shell_solver.fe, dummy_q, update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices(
      shell_solver.fe.dofs_per_cell);
    for (auto cell = shell_solver.dof_handler.begin_active();
         cell != shell_solver.dof_handler.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<CellInterpolators>> p =
          cell_interpolators.get_data(cell);
        dummy_fe_values.reinit(cell);
        auto support_points = dummy_fe_values.get_quadrature_points();
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < unit_points.size(); ++i)
          {
            const unsigned int index =
              shell_solver.fe.system_to_component_index(i).first;
            Vector<double> d_value(3), v_value(3);
            p[i]->interpolator->point_value(localized_current_displacement,
                                            d_value);
            p[i]->interpolator->point_value(localized_current_velocity,
                                            v_value);
            auto line = dof_indices[i];
            displacement_constraints.add_line(line);
            velocity_constraints.add_line(line);
            displacement_constraints.set_inhomogeneity(line, d_value[index]);
            velocity_constraints.set_inhomogeneity(line, v_value[index]);
          }
      }
    displacement_constraints.close();
    velocity_constraints.close();
    displacement_constraints.distribute(shell_solver.current_displacement);
    velocity_constraints.distribute(shell_solver.current_velocity);
  }
} // namespace MPI
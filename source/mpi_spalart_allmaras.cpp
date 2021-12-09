#include "mpi_spalart_allmaras.h"
#include "preconditioner_pilut.h"

#include <sstream>

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SpalartAllmaras<dim>::SpalartAllmaras(const FluidSolver<dim> &fluid_solver)
      : TurbulenceModel<dim>(fluid_solver)
    {
    }

    template <int dim>
    void SpalartAllmaras<dim>::update_moving_wall_distance(
      const SolidVerticesStorage &boundary_vertices,
      const std::list<
        std::pair<typename Triangulation<dim>::active_cell_iterator,
                  unsigned int>> &boundary_faces,
      const Vector<double> &shear_velocities)
    {
      TimerOutput::Scope timer_section(timer, "Update moving wall distance");

      const std::vector<Point<dim>> &unit_points =
        scalar_fe->get_unit_support_points();
      MappingQGeneric<dim> mapping(parameters->fluid_velocity_degree);
      Quadrature<dim> dummy_q(unit_points);
      FEValues<dim> dummy_fe_values(
        mapping, *scalar_fe, dummy_q, update_quadrature_points);
      // A pre-defined lambada that computes the dot product
      auto dot_product =
        [](Point<dim> p, Point<dim> p1, Point<dim> p2) -> double {
        // Note p is the point of angle, p1 and p2 are the point on the 2 edges
        double retval{0.0};
        for (unsigned d = 0; d < dim; ++d)
          {
            retval += (p[d] - p1[d]) * (p[d] - p2[d]);
          }
        return retval;
      };

      // Get stress tensor from given vertex iterator
      auto get_shear_velocity =
        [&boundary_vertices, &shear_velocities](
          typename Triangulation<dim>::vertex_iterator vertex) {
          int this_index{boundary_vertices.find(vertex)->second.second};
          return shear_velocities[this_index];
        };

      auto compute_y_plus = [&](const double shear_velocity,
                                const double dist) {
        return dist * shear_velocity /
               (parameters->viscosity / parameters->fluid_rho);
      };

      // Loop over fluid cells
      for (auto &f_cell : scalar_dof_handler->active_cell_iterators())
        {
          if (!f_cell->is_artificial())
            {
              // Compute dists on support points
              dummy_fe_values.reinit(f_cell);

              auto support_points = dummy_fe_values.get_quadrature_points();

              const std::vector<std::shared_ptr<WallDistance>> p =
                wall_distance.get_data(f_cell);
              for (unsigned v = 0; v < unit_points.size(); ++v)
                {
                  double min_dist{std::numeric_limits<double>::max()};
                  const auto &v0 = dummy_fe_values.get_quadrature_points()[v];
                  // Check edge distance first
                  if (dim == 2)
                    {
                      for (const auto &[cell, f] : boundary_faces)
                        {
                          auto s_face = cell->face(f);
                          const auto &v1 = s_face->vertex(0);
                          const auto &v2 = s_face->vertex(1);
                          // Check if the point is in the edge distance region
                          double v1_product{dot_product(v1, v0, v2)};
                          if (v1_product > 0 && dot_product(v2, v0, v1) > 0)
                            {
                              // Compute edge distance
                              double intersection_ratio{
                                v1_product /
                                (s_face->measure() * s_face->measure())};
                              double current_dist{v0.distance(
                                v1 + intersection_ratio * (v2 - v1))};
                              if (current_dist < min_dist)
                                {
                                  double shear_velocity_v1 = get_shear_velocity(
                                    s_face->vertex_iterator(0));
                                  double shear_velocity_v2 = get_shear_velocity(
                                    s_face->vertex_iterator(1));
                                  auto intersection_shear_velocity =
                                    shear_velocity_v1 +
                                    (shear_velocity_v2 - shear_velocity_v1) *
                                      intersection_ratio;
                                  min_dist = current_dist;
                                  // Update y_plus
                                  p[v]->y_plus = compute_y_plus(
                                    intersection_shear_velocity, min_dist);
                                }
                            }
                        }
                    }
                  // In the vertex distance region
                  for (const auto &[s_vert, face_and_indices] :
                       boundary_vertices)
                    {
                      double current_dist = v0.distance(s_vert->vertex(0));
                      if (current_dist < min_dist)
                        {
                          min_dist = current_dist;
                          // Update y_plus
                          p[v]->y_plus = compute_y_plus(
                            get_shear_velocity(s_vert), min_dist);
                        }
                    }
                  p[v]->moving_wall_distance.emplace(min_dist);
                }
            }
        }
    }

    template <int dim>
    void SpalartAllmaras<dim>::update_boundary_condition(bool first_step)
    {
      TimerOutput::Scope timer_section(timer, "Update boundary condition");

      // Check if indicator function has value
      AssertThrow(indicator_function.has_value(),
                  ExcMessage("No available indicator function for SA model!"));
      // Copy the zero constraints for non first step
      if (!first_step)
        {
          nonzero_constraints.clear();
          nonzero_constraints.copy_from(zero_constraints);
        }
      // Additional constraints
      AffineConstraints<double> inner_zero(*locally_relevant_scalar_dofs);
      AffineConstraints<double> inner_nonzero(*locally_relevant_scalar_dofs);

      std::vector<types::global_dof_index> dof_indices(
        scalar_fe->dofs_per_cell);
      std::vector<bool> touched_dofs(scalar_dof_handler->n_dofs(), 0);

      const std::vector<Point<dim>> &unit_points =
        scalar_fe->get_unit_support_points();

      // von Karman constan for wall function
      constexpr double kappa{0.41};
      for (const auto &cell : scalar_dof_handler->active_cell_iterators())
        {
          // Ghost cells must be considered
          if (cell->is_artificial())
            {
              continue;
            }
          // Get the indicator from indicator function
          auto ind = (*indicator_function)(cell);
          if (ind == 1)
            {
              cell->get_dof_indices(dof_indices);
              for (const auto &line : dof_indices)
                {
                  if (touched_dofs[line])
                    {
                      continue;
                    }
                  inner_zero.add_line(line);
                  inner_nonzero.add_line(line);
                  touched_dofs[line] = true;
                  inner_nonzero.set_inhomogeneity(line,
                                                  -present_solution(line));
                }
              continue;
            }
          // Get y plus
          auto p = wall_distance.get_data(cell);
          for (unsigned v = 0; v < unit_points.size(); ++v)
            {
              cell->get_dof_indices(dof_indices);
              if (p[v]->moving_wall_distance.value_or(2.0) <
                    parameters->spalart_allmaras_wall_function_distance &&
                  p[v]->y_plus < 200.0)
                {
                  auto line = dof_indices[v];
                  if (touched_dofs[line])
                    {
                      continue;
                    }
                  inner_zero.add_line(line);
                  inner_nonzero.add_line(line);
                  touched_dofs[line] = true;
                  inner_nonzero.set_inhomogeneity(line,
                                                  kappa * p[v]->y_plus *
                                                      parameters->viscosity /
                                                      parameters->fluid_rho -
                                                    present_solution(line));
                }
            }
        }
      inner_zero.close();
      inner_nonzero.close();
      nonzero_constraints.merge(
        inner_nonzero,
        AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
      zero_constraints.merge(
        inner_zero,
        AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
    }

    template <int dim>
    double SpalartAllmaras<dim>::get_shear_velocity(double vel,
                                                    double init_guess)
    {
      // Usually not belong to this process or out of bound
      if (std::fabs(vel) < 1e-10)
        {
          return 0.0;
        }
      // Kinematic viscosity
      const double nu{parameters->viscosity / parameters->fluid_rho};
      // Image distance
      const double dist{parameters->spalart_allmaras_image_distance};
      // If it's in viscous sublayer (y+ < 5, u+ = y+)
      if (vel * dist / nu < std::sqrt(5.0))
        {
          return vel / std::sqrt(vel * dist / nu);
        }
      // Threshold for initial guess
      init_guess = std::max(init_guess, 5.0 * nu / dist);
      // Constants for the analytical wall velocity profile
      constexpr double B{5.03339088}, a1{8.14822158}, a2{-6.92870938},
        b1{7.46008761}, b2{7.46814579}, c1{2.54967735}, c2{1.33016516},
        c3{3.59945911}, c4{3.63975319};
      // Avoid using std::pow().
      auto squared = [](double a) constexpr { return a * a; };
      // Analytical wall velocity profile
      auto u_plus =
        [ B, a1, a2, b1, b2, c1, c2, c3, c4, squared ](double yp) constexpr
      {
        return B + c1 * std::log(squared(yp + a1) + squared(b1)) -
               c2 * std::log(squared(yp + a2) + squared(b2)) -
               c3 * std::atan2(b1, yp + a1) - c4 * std::atan2(b2, yp + a2);
      };

      // Derivative of u+ in repect to y+
      constexpr double kappa{0.41}, c_nu1_cubed{7.1 * 7.1 * 7.1},
        kappa_cubed{kappa * kappa * kappa};
      auto dup_dyp = [ kappa_cubed, c_nu1_cubed ](double yp) constexpr
      {
        return (kappa_cubed * yp * yp * yp) /
               (c_nu1_cubed + kappa_cubed * yp * yp * yp);
      };

      // Newton iteration
      constexpr int max_iter{30};
      constexpr double tol{1e-2};
      // Shear velocity
      double ut{init_guess};
      for (int i = 0; i < max_iter; ++i)
        {
          double yp = ut * dist / nu;
          double up = u_plus(yp);
          double ut_next =
            ut - (ut * up - vel) / (up + ut * dist / nu * dup_dyp(yp));
          if (std::abs(ut_next - ut) < tol * std::abs(ut))
            {
              ut = ut_next;
              break;
            }
          ut = ut_next;
        }
      return ut;
    }

    template <int dim>
    void SpalartAllmaras<dim>::run_one_step(bool apply_nonzero_constraints)
    {
      std::cout.precision(6);
      std::cout.width(12);

      pcout << std::string(96, '*') << std::endl
            << "Solving for S-A turbulence model..." << std::endl;
      // Resetting
      double current_residual = 1.0;
      double initial_residual = 1.0;
      double relative_residual = 1.0;
      unsigned int outer_iteration = 0;
      evaluation_point = present_solution;
      while (relative_residual > parameters->fluid_tolerance &&
             current_residual > 1e-14)
        {
          AssertThrow(outer_iteration < parameters->fluid_max_iterations,
                      ExcMessage("Too many Newton iterations!"));

          newton_update = 0;

          // Since evaluation_point changes at every iteration,
          // we have to reassemble both the lhs and rhs of the system
          // before solving it.
          assemble(apply_nonzero_constraints && outer_iteration == 0);
          auto state = solve(apply_nonzero_constraints && outer_iteration == 0);
          current_residual = system_rhs.l2_norm();

          // Update evaluation_point. Since newton_update has been set to
          // the correct bc values, there is no need to distribute the
          // evaluation_point again. Note we have to use a non-ghosted
          // vector as a buffer in order to do addition.
          PETScWrappers::MPI::Vector tmp;
          tmp.reinit(*locally_owned_scalar_dofs, mpi_communicator);
          tmp = evaluation_point;
          tmp += newton_update;
          evaluation_point = tmp;

          if (outer_iteration == 0)
            {
              initial_residual = current_residual;
            }
          relative_residual = current_residual / initial_residual;

          pcout << std::scientific << std::left << " ITR = " << std::setw(2)
                << outer_iteration << " ABS_RES = " << current_residual
                << " REL_RES = " << relative_residual
                << " GMRES_ITR = " << std::setw(3) << state.first
                << " GMRES_RES = " << state.second << std::endl;
          outer_iteration++;
        }
      // Update solution increment
      PETScWrappers::MPI::Vector tmp1, tmp2;
      tmp1.reinit(*locally_owned_scalar_dofs, mpi_communicator);
      tmp2.reinit(*locally_owned_scalar_dofs, mpi_communicator);
      tmp1 = evaluation_point;
      tmp2 = present_solution;
      tmp2 -= tmp1;
      // Newton iteration converges, update time and solution
      present_solution = evaluation_point;
      // Update the eddy viscotiy for RANS equations based on the solution
      update_eddy_viscosity();
    }

    template <int dim>
    void SpalartAllmaras<dim>::make_constraints()
    {
      // In S-A turbulence model, there are only 2 types of boundary conditions
      // and both are Dirichlet BC. For walls, the eddy viscosity is 0, for
      // inflow, it is 5 times of the laminar viscosity.

      // The criteria to determine a boundary is wall or inflow: if it has
      // velocity non-penetration BC then it's wall. Otherwise inflow.
      // Gradient-free boundary condition (do-nothing) is applied to boundaries
      // that are neither walls are inflow.
      {
        zero_constraints.clear();
        nonzero_constraints.clear();
        zero_constraints.reinit(*locally_relevant_scalar_dofs);
        nonzero_constraints.reinit(*locally_relevant_scalar_dofs);
        DoFTools::make_hanging_node_constraints(*scalar_dof_handler,
                                                zero_constraints);
        DoFTools::make_hanging_node_constraints(*scalar_dof_handler,
                                                nonzero_constraints);
        for (auto itr = parameters->spalart_allmaras_model_bcs.begin();
             itr != parameters->spalart_allmaras_model_bcs.end();
             ++itr)
          {
            // First get the id and flag from the input file
            unsigned id = itr->first;
            unsigned type = itr->second;

            // Specify the BC type
            double augmented_value{0.0};
            switch (type)
              {
              case 0:
                break;
              case 1:
                augmented_value =
                  5.0 * parameters->viscosity / parameters->fluid_rho;
                break;
              default:
                AssertThrow(
                  false, ExcMessage("Unrecogonized Spalart-Allmaras BC type!"));
                break;
              }
            VectorTools::interpolate_boundary_values(
              MappingQGeneric<dim>(parameters->fluid_velocity_degree),
              *scalar_dof_handler,
              id,
              Functions::ConstantFunction<dim>(augmented_value),
              nonzero_constraints);
            VectorTools::interpolate_boundary_values(
              MappingQGeneric<dim>(parameters->fluid_velocity_degree),
              *scalar_dof_handler,
              id,
              Functions::ZeroFunction<dim>(),
              zero_constraints);
          }
      }
      nonzero_constraints.close();
      zero_constraints.close();
    }

    template <int dim>
    void SpalartAllmaras<dim>::setup_cell_property()
    {
      pcout << "   Setting up cell property for turbulence model..."
            << std::endl;
      auto n_unit_points{scalar_fe->get_unit_support_points().size()};
      for (auto &cell : triangulation->active_cell_iterators())
        {
          if (!cell->is_artificial())
            {
              wall_distance.initialize(cell, n_unit_points);
              const std::vector<std::shared_ptr<WallDistance>> p =
                wall_distance.get_data(cell);
              for (unsigned int v = 0; v < n_unit_points; ++v)
                {
                  p[v]->fixed_wall_distance = 0.0;
                  // The moving wall distance is initialized as empty. If
                  // it's not updated, i.e., for a pure fluid simulation, it
                  // will not be considered.
                  p[v]->moving_wall_distance = {};
                  // y+ is the non-dimensionzalied wall distance used in wall
                  // function
                  p[v]->y_plus = 201.0;
                }
            }
        }
      // Compute the fixed wall distance
      // Collect all the boundary vertices first
      std::map<unsigned, Point<dim>> boundary_points;
      // This is a vector of vector storing the coordinates of the boundary
      // points. The first element stores x-coordinates, second stores
      // y-coordinates, etc.
      std::vector<std::vector<double>> boundary_points_coord(dim);
      for (auto &cell : triangulation->active_cell_iterators())
        {
          if (cell->is_locally_owned() && cell->at_boundary())
            {
              for (unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                {
                  if (cell->face(f)->at_boundary())
                    {
                      unsigned boundary_id = cell->face(f)->boundary_id();
                      // Only walls (with type 0 BC) are counted
                      auto bc = parameters->spalart_allmaras_model_bcs.find(
                        boundary_id);
                      if (bc == parameters->spalart_allmaras_model_bcs.end() ||
                          bc->second != 0)
                        {
                          continue;
                        }
                      for (unsigned v = 0;
                           v < GeometryInfo<dim>::vertices_per_face;
                           ++v)
                        {
                          auto success = boundary_points.insert(
                            {cell->face(f)->vertex_index(v),
                             cell->face(f)->vertex(v)});
                          if (success.second)
                            {
                              for (unsigned d = 0; d < dim; ++d)
                                {
                                  boundary_points_coord[d].emplace_back(
                                    cell->face(f)->vertex(v)[d]);
                                }
                            }
                        }
                    }
                }
            }
        }
      // Reduce all boundary vertices to all MPI ranks
      auto combiner = [](const std::vector<double> &a,
                         const std::vector<double> &b) -> std::vector<double> {
        std::vector<double> ret_val{a};
        ret_val.insert(ret_val.end(), b.begin(), b.end());
        return ret_val;
      };
      std::vector<std::vector<double>> global_boundary_coord;
      for (unsigned d = 0; d < dim; ++d)
        {
          global_boundary_coord.emplace_back(
            Utilities::MPI::all_reduce<std::vector<double>>(
              boundary_points_coord[d], mpi_communicator, combiner));
        }
      for (unsigned d = 1; d < dim; ++d)
        {
          AssertThrow(global_boundary_coord[0].size() ==
                        global_boundary_coord[d].size(),
                      ExcMessage("Sizes of coordinates don't match!"));
        }
      FEValues<dim> scalar_fe_values(*scalar_fe,
                                     *volume_quad_formula,
                                     update_values | update_quadrature_points);
      const std::vector<Point<dim>> &unit_points =
        scalar_fe->get_unit_support_points();
      MappingQGeneric<dim> mapping(parameters->fluid_velocity_degree);
      Quadrature<dim> dummy_q(unit_points);
      FEValues<dim> dummy_fe_values(
        mapping, *scalar_fe, dummy_q, update_quadrature_points);
      // Compute minimal distance
      for (auto &cell : scalar_dof_handler->active_cell_iterators())
        {
          if (!cell->is_artificial())
            {
              // Compute dists on support points
              scalar_fe_values.reinit(cell);
              dummy_fe_values.reinit(cell);

              auto support_points = dummy_fe_values.get_quadrature_points();

              const std::vector<std::shared_ptr<WallDistance>> p =
                wall_distance.get_data(cell);
              for (unsigned v = 0; v < unit_points.size(); ++v)
                {
                  double min_dist{std::numeric_limits<double>::max()};
                  for (unsigned i = 0; i < global_boundary_coord[0].size(); ++i)
                    {
                      double current_dist{0.0};
                      for (unsigned d = 0; d < dim; ++d)
                        {
                          current_dist += pow(global_boundary_coord[d][i] -
                                                support_points[v][d],
                                              2);
                        }
                      current_dist = sqrt(current_dist);
                      min_dist = std::min(min_dist, current_dist);
                    }
                  p[v]->fixed_wall_distance = min_dist;
                }
            }
        }
    }

    template <int dim>
    void SpalartAllmaras<dim>::initialize_system()
    {
      // This function must be called after the fluid solver calls its
      // initialize_system()
      this->TurbulenceModel<dim>::initialize_system();

      // present_solution is ghosted because it is used in the
      // output and mesh refinement functions.
      present_solution.reinit(*locally_owned_scalar_dofs,
                              *locally_relevant_scalar_dofs,
                              mpi_communicator);
      evaluation_point.reinit(*locally_owned_scalar_dofs,
                              *locally_relevant_scalar_dofs,
                              mpi_communicator);
      // newton_update is non-ghosted because the linear solver needs
      // a completely distributed vector.
      newton_update.reinit(*locally_owned_scalar_dofs, mpi_communicator);

      // Apply the initial condition
      newton_update.add(
        parameters->spalart_allmaras_initial_condition_coefficient *
        parameters->viscosity / parameters->fluid_rho);
      zero_constraints.distribute(newton_update);
      present_solution = newton_update;
      setup_cell_property();
    }

    template <int dim>
    void SpalartAllmaras<dim>::save_checkpoint(
      std::optional<parallel::distributed::
                      SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
        &sol_trans)
    {
      sol_trans.emplace(*scalar_dof_handler);
      sol_trans->prepare_for_serialization(present_solution);
    }

    template <int dim>
    bool SpalartAllmaras<dim>::load_checkpoint()
    {
      // Suppose fluid solver already called make_constraints() and
      // initialize_system()
      parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector>
        sol_trans(*scalar_dof_handler);
      PETScWrappers::MPI::Vector tmp;
      tmp.reinit(*locally_owned_scalar_dofs, mpi_communicator);
      sol_trans.deserialize(tmp);
      present_solution = tmp;
      return true;
    }

    template <int dim>
    void SpalartAllmaras<dim>::pre_refine_mesh(
      std::optional<parallel::distributed::
                      SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
        &sol_trans)
    {
      sol_trans.emplace(*scalar_dof_handler);
      sol_trans->prepare_for_coarsening_and_refinement(present_solution);
    }

    template <int dim>
    void SpalartAllmaras<dim>::post_refine_mesh(
      std::optional<parallel::distributed::
                      SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
        &sol_trans)
    {
      AssertThrow(
        sol_trans.has_value(),
        ExcMessage("Solution transfer for turbulence model is empty!"));
      PETScWrappers::MPI::Vector tmp;
      tmp.reinit(*locally_owned_scalar_dofs, mpi_communicator);
      tmp = 0;
      sol_trans->interpolate(tmp);
      present_solution = tmp;
    }

    template <int dim>
    void SpalartAllmaras<dim>::assemble(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      // Defining constants
      constexpr double cv1{7.1}, cv2{0.7}, cv3{0.9};
      constexpr double cb1{0.1355}, cb2{0.622}, ct3{1.2}, ct4{0.5}, kappa{0.41};
      constexpr double cw2{0.3}, cw3{2.0};
      constexpr double cn1{16.0};
      constexpr double sigma = 2.0 / 3.0;
      constexpr double cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;

      system_matrix = 0;
      system_rhs = 0;

      FEValues<dim> fe_values(*fe,
                              *volume_quad_formula,
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);
      FEValues<dim> scalar_fe_values(*scalar_fe,
                                     *volume_quad_formula,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values);

      const double dt = time->get_delta_t();

      const unsigned int dofs_per_cell = scalar_fe->dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula->size();

      // For fluid solution interpolation
      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);
      const FEValuesExtractors::Scalar viscosity(0);

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      // The laminar dynamic viscosity (mu) and laminar kinematic viscosity (nu)
      const double laminar_viscosity = parameters->viscosity;

      // A vector to store nodal values of nearest wall distance d.
      std::vector<double> nodal_d(scalar_fe->get_unit_support_points().size());

      // For the linearized system, we create temporary storage for present
      // velocity and gradient, current eddy viscosity and gradient. In
      // practice, they are all obtained through their shape functions at
      // quadrature points.
      std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
      // The vorticity. Tensor<1, 1> for 2d and Tensor<1, 3> for 3d
      std::vector<Tensor<1, dim * 2 - 3>> present_velocity_curls(n_q_points);

      std::vector<double> current_nu_values(n_q_points);
      std::vector<Tensor<1, dim>> current_nu_gradients(n_q_points);
      std::vector<double> present_nu_values(n_q_points);

      // Test functions and trial functions
      std::vector<double> phi(dofs_per_cell);
      std::vector<Tensor<1, dim>> grad_phi(dofs_per_cell);

      auto cell = dof_handler->begin_active();
      auto scalar_cell = scalar_dof_handler->begin_active();
      for (; scalar_cell != scalar_dof_handler->end(); ++cell, ++scalar_cell)
        {
          if (scalar_cell->is_locally_owned())
            {
              auto p = wall_distance.get_data(cell);

              fe_values.reinit(cell);
              scalar_fe_values.reinit(scalar_cell);

              local_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                *fluid_present_solution, present_velocity_values);
              fe_values[velocities].get_function_curls(*fluid_present_solution,
                                                       present_velocity_curls);

              scalar_fe_values[viscosity].get_function_values(
                present_solution, present_nu_values);

              scalar_fe_values[viscosity].get_function_values(
                evaluation_point, current_nu_values);
              scalar_fe_values[viscosity].get_function_gradients(
                evaluation_point, current_nu_gradients);

              // Wall distance. Take the smaller value between fixed and
              // moving wall distance
              std::fill_n(nodal_d.begin(), nodal_d.size(), 0.0);
              for (unsigned v = 0; v < nodal_d.size(); ++v)
                {
                  nodal_d[v] = (p[v]->moving_wall_distance.value_or(
                                  p[v]->fixed_wall_distance + 1.0) <
                                p[v]->fixed_wall_distance)
                                 ? p[v]->moving_wall_distance.value()
                                 : p[v]->fixed_wall_distance;
                }
              double laminar_nu;
              if (indicator_function.has_value() &&
                  (*indicator_function)(cell) == 1)
                {
                  laminar_nu = 1 / parameters->fluid_rho;
                }
              else
                {
                  laminar_nu = laminar_viscosity / parameters->fluid_rho;
                }

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // Compute nearest wall distance
                  double d{0.0};
                  for (unsigned v = 0; v < nodal_d.size(); ++v)
                    {
                      d += scalar_fe_values.shape_value(v, q) * nodal_d[v];
                    }
                  // Evaluate chi (viscosity ratio) and the f coefficients (chi
                  // dependent)
                  const double chi = present_nu_values[q] / laminar_nu;
                  // For production term
                  const double ft2 = ct3 * exp(-ct4 * chi * chi);
                  const double fv1 =
                    chi * chi * chi / (chi * chi * chi + cv1 * cv1 * cv1);
                  const double fv2 = 1.0 - chi / (1.0 + chi * fv1);
                  // Evaluate vorticity term
                  const double S{present_velocity_curls[q].norm()};
                  const double S_bar =
                    present_nu_values[q] / (kappa * kappa * d * d) * fv2;
                  const double S_tilde =
                    S_bar >= -cv2 * S ? S + S_bar
                                      : S + S * (cv2 * cv2 * S - cv3 * S_bar) /
                                              ((cv3 - 2 * cv2) * S - S_bar);

                  // For destruction term
                  const double r = [&] {
                    double r;
                    if (std::fabs(S_tilde) > 1e-8)
                      {
                        std::min({present_nu_values[q] /
                                    (S_tilde * kappa * kappa * d * d),
                                  10.0});
                      }
                    else
                      {
                        r = 10.0;
                      }
                    return r;
                  }();
                  const double g = r + cw2 * (pow(r, 6) - r);
                  const double fw =
                    g * pow((1 + pow(cw3, 6)) / (pow(g, 6) + pow(cw3, 6)),
                            1.0 / 6.0);

                  // Determine to use positive or negative S-A
                  // P is the multiplier of $\nu\tilde$ in production term
                  const double P = present_nu_values[q] >= 0
                                     ? cb1 * (1 - ft2) * S_tilde
                                     : cb1 * (1 - ct3) * S;
                  // D is the multiplier of ${\nu\tilde}^2$ in the destruction
                  // term
                  const double D =
                    present_nu_values[q] >= 0
                      ? (cw1 * fw - cb1 / (kappa * kappa) * ft2) / (d * d)
                      : -cw1 / (d * d);
                  // fn is the multiplier of $\nu\tilde$ in the diffusion term
                  const double fn =
                    present_nu_values[q] >= 0
                      ? 1.0
                      : (cn1 + chi * chi * chi) / (cn1 - chi * chi * chi);

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      grad_phi[k] = scalar_fe_values.shape_grad(k, q);
                      phi[k] = scalar_fe_values.shape_value(k, q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          local_matrix(i, j) +=
                            (
                              // Total derivative
                              phi[i] * phi[j] / dt +
                              phi[i] * present_velocity_values[q] *
                                grad_phi[j] +
                              // Diffusion
                              1 / sigma *
                                (laminar_nu + fn * present_nu_values[q]) *
                                scalar_product(grad_phi[i], grad_phi[j]) -
                              2 * cb2 / sigma * phi[i] * grad_phi[j] *
                                current_nu_gradients[q] -
                              // Production
                              P * phi[i] * phi[j] +
                              // Destruction
                              2 * D * phi[i] * phi[j] * current_nu_values[q]) *
                            scalar_fe_values.JxW(q);
                        }

                      // RHS
                      local_rhs(i) +=
                        -(
                          // Total derivative
                          phi[i] *
                            (current_nu_values[q] - present_nu_values[q]) / dt +
                          phi[i] * (present_velocity_values[q] *
                                    current_nu_gradients[q]) +
                          // Diffusion
                          1 / sigma * (laminar_nu + fn * present_nu_values[q]) *
                            scalar_product(grad_phi[i],
                                           current_nu_gradients[q]) -
                          cb2 / sigma * phi[i] * current_nu_gradients[q] *
                            current_nu_gradients[q] -
                          // Production
                          P * phi[i] * current_nu_values[q] +
                          // Destruction
                          D * phi[i] * current_nu_values[q] *
                            current_nu_values[q]) *
                        scalar_fe_values.JxW(q);
                    }
                }

              scalar_cell->get_dof_indices(local_dof_indices);

              const AffineConstraints<double> &constraints_used =
                use_nonzero_constraints ? nonzero_constraints
                                        : zero_constraints;

              constraints_used.distribute_local_to_global(local_matrix,
                                                          local_rhs,
                                                          local_dof_indices,
                                                          system_matrix,
                                                          system_rhs,
                                                          true);
            }
        }

      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
    }

    template <int dim>
    std::pair<unsigned, double>
    SpalartAllmaras<dim>::solve(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Solve linear system");

      SolverControl solver_control(scalar_dof_handler->n_dofs() * 2,
                                   1e-8 * system_rhs.l2_norm());

      PreconditionEuclid preconditioner;
      preconditioner.initialize(system_matrix);
      // Because PETScWrappers::SolverGMRES requires preconditioner derived
      // from PETScWrappers::PreconditionBase, we use dealii SolverFGMRES.
      GrowingVectorMemory<PETScWrappers::MPI::Vector> vector_memory;
      SolverFGMRES<PETScWrappers::MPI::Vector> gmres(solver_control,
                                                     vector_memory);

      // The solution vector must be non-ghosted
      gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);

      const AffineConstraints<double> &constraints_used =
        use_nonzero_constraints ? nonzero_constraints : zero_constraints;
      constraints_used.distribute(newton_update);

      return {solver_control.last_step(), solver_control.last_value()};
    }

    template <int dim>
    void SpalartAllmaras<dim>::update_eddy_viscosity()
    {
      // From Spalart et al., eddy viscosity $\mu_t$ is evaluated as
      // $f_{\nu1}\tilde{\nu}\rho$ where $f_{\nu1} = \frac{\chi^3}{\chi^3 +
      // c_{\nu1}}$.
      constexpr double cv1{7.1};
      // The 3rd power of cv1
      constexpr double cv1_3 = cv1 * cv1 * cv1;
      const double laminar_nu = parameters->viscosity / parameters->fluid_rho;
      // Cache for the eddy viscosity because the original vector is ghosted.
      PETScWrappers::MPI::Vector tmp_eddy_viscosity(*locally_owned_scalar_dofs,
                                                    mpi_communicator);
      for (auto r = present_solution.local_range().first;
           r < present_solution.local_range().second;
           ++r)
        {
          const double chi = present_solution[r] / laminar_nu;
          tmp_eddy_viscosity[r] = chi * chi * chi / (chi * chi * chi + cv1_3) *
                                  present_solution[r] * parameters->fluid_rho;
        }
      tmp_eddy_viscosity.compress(VectorOperation::insert);
      eddy_viscosity = tmp_eddy_viscosity;
    }

    template class SpalartAllmaras<2>;
    template class SpalartAllmaras<3>;
  } // namespace MPI
} // namespace Fluid

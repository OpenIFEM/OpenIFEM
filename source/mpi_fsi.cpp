#include "mpi_fsi.h"
#include "mpi_spalart_allmaras.h"
#include <iostream>

namespace MPI
{
  template <int dim>
  FSI<dim>::~FSI()
  {
    timer.print_summary();
  }

  template <int dim>
  FSI<dim>::FSI(Fluid::MPI::FluidSolver<dim> &f,
                Solid::MPI::SharedSolidSolver<dim> &s,
                const Parameters::AllParameters &p,
                bool use_dirichlet_bc)
    : fluid_solver(f),
      solid_solver(s),
      parameters(p),
      mpi_communicator(MPI_COMM_WORLD),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval,
           parameters.save_interval),
      timer(
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
      penetration_criterion(nullptr),
      use_dirichlet_bc(use_dirichlet_bc)
  {
    solid_box.reinit(2 * dim);
  }

  template <int dim>
  void FSI<dim>::move_solid_mesh(bool move_forward)
  {
    TimerOutput::Scope timer_section(timer, "Move solid mesh");
    // All gather the information so each process has the entire solution.
    Vector<double> localized_displacement(solid_solver.current_displacement);
    // Exactly the same as the serial version, since we must update the
    // entire graph on every process.
    std::vector<bool> vertex_touched(solid_solver.triangulation.n_vertices(),
                                     false);
    for (auto cell = solid_solver.dof_handler.begin_active();
         cell != solid_solver.dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> vertex_displacement;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    vertex_displacement[d] =
                      localized_displacement(cell->vertex_dof_index(v, d));
                  }
                if (move_forward)
                  {
                    cell->vertex(v) += vertex_displacement;
                  }
                else
                  {
                    cell->vertex(v) -= vertex_displacement;
                  }
              }
          }
      }
  }

  template <int dim>
  void FSI<dim>::collect_solid_boundaries()
  {
    if (dim == 2)
      for (auto cell = solid_solver.triangulation.begin_active();
           cell != solid_solver.triangulation.end();
           ++cell)
        {
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  solid_boundaries.push_back(cell->face(f));
                }
            }
        }
  }

  template <int dim>
  void FSI<dim>::update_solid_box()
  {
    move_solid_mesh(true);
    solid_box = 0;
    for (unsigned int i = 0; i < dim; ++i)
      {
        solid_box(2 * i) =
          solid_solver.triangulation.get_vertices().begin()->operator()(i);
        solid_box(2 * i + 1) =
          solid_solver.triangulation.get_vertices().begin()->operator()(i);
      }
    for (auto v = solid_solver.triangulation.get_vertices().begin();
         v != solid_solver.triangulation.get_vertices().end();
         ++v)
      {
        for (unsigned int i = 0; i < dim; ++i)
          {
            if ((*v)(i) < solid_box(2 * i))
              solid_box(2 * i) = (*v)(i);
            else if ((*v)(i) > solid_box(2 * i + 1))
              solid_box(2 * i + 1) = (*v)(i);
          }
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::update_vertices_mask()
  {
    // Initilize vertices mask
    vertices_mask.clear();
    vertices_mask.resize(fluid_solver.triangulation.n_vertices(), false);
    for (auto cell = fluid_solver.triangulation.begin_active();
         cell != fluid_solver.triangulation.end();
         ++cell)
      {
        if (!cell->is_locally_owned())
          {
            continue;
          }
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            vertices_mask[cell->vertex_index(v)] = true;
          }
      }
  }

  template <int dim>
  bool FSI<dim>::point_in_solid(const DoFHandler<dim> &df,
                                const Point<dim> &point)
  {
    // Check whether the point is in the solid box first.
    for (unsigned int i = 0; i < dim; ++i)
      {
        if (point(i) < solid_box(2 * i) || point(i) > solid_box(2 * i + 1))
          return false;
      }

    // Compute its angle to each boundary face
    if (dim == 2)
      {
        unsigned int cross_number = 0;
        unsigned int half_cross_number = 0;
        for (auto f = solid_boundaries.begin(); f != solid_boundaries.end();
             ++f)
          {
            Point<dim> p1 = (*f)->vertex(0), p2 = (*f)->vertex(1);
            double y_diff1 = p1(1) - point(1);
            double y_diff2 = p2(1) - point(1);
            double x_diff1 = p1(0) - point(0);
            double x_diff2 = p2(0) - point(0);
            Tensor<1, dim> r1 = p1 - p2;
            Tensor<1, dim> r2;
            // r1[1] == 0 if the boundary is horizontal
            if (r1[1] != 0.0)
              r2 = r1 * (point(1) - p2(1)) / r1[1];
            if (y_diff1 * y_diff2 < 0)
              {
                // Point is on the left of the boundary
                if (r2[0] + p2(0) > point(0))
                  {
                    ++cross_number;
                  }
                // Point is on the boundary
                else if (r2[0] + p2(0) == point(0))
                  {
                    return true;
                  }
              }
            // Point is on the same horizontal line with one of the vertices
            else if (y_diff1 * y_diff2 == 0)
              {
                // The boundary is horizontal
                if (y_diff1 == 0 && y_diff2 == 0)
                  // The point is on it
                  if (x_diff1 * x_diff2 < 0)
                    {
                      return true;
                    }
                  // The point is not on it
                  else
                    continue;
                // On the left of the boundary
                else if (r2[0] + p2(0) > point(0))
                  { // The point must not be on the top or bottom of the box
                    // (because it can be tangential)
                    if (point(1) != solid_box(2) && point(1) != solid_box(3))
                      ++half_cross_number;
                  }
                // Point overlaps with the vertex
                else if (point == p1 || point == p2)
                  {
                    return true;
                  }
              }
          }
        cross_number += half_cross_number / 2;
        if (cross_number % 2 == 0)
          return false;
        return true;
      }
    for (auto cell = df.begin_active(); cell != df.end(); ++cell)
      {
        if (cell->point_inside(point))
          {
            return true;
          }
      }
    return false;
  }

  template <int dim>
  void FSI<dim>::setup_cell_hints()
  {
    unsigned int n_unit_points =
      fluid_solver.fe.get_unit_support_points().size();
    for (auto cell = fluid_solver.triangulation.begin_active();
         cell != fluid_solver.triangulation.end();
         ++cell)
      {
        if (!cell->is_artificial())
          {
            cell_hints.initialize(cell, n_unit_points);
            const std::vector<
              std::shared_ptr<typename DoFHandler<dim>::active_cell_iterator>>
              hints = cell_hints.get_data(cell);
            Assert(hints.size() == n_unit_points,
                   ExcMessage("Wrong number of cell hints!"));
            for (unsigned int v = 0; v < n_unit_points; ++v)
              {
                // Initialize the hints with the begin iterators!
                *(hints[v]) = solid_solver.dof_handler.begin_active();
              }
          }
      }
  }

  template <int dim>
  void FSI<dim>::update_solid_displacement()
  {
    move_solid_mesh(true);
    Vector<double> localized_solid_displacement(
      solid_solver.current_displacement);
    std::vector<bool> vertex_touched(solid_solver.dof_handler.n_dofs(), false);
    for (auto cell : solid_solver.dof_handler.active_cell_iterators())
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)] &&
                !solid_solver.constraints.is_constrained(cell->vertex_index(v)))
              {
                vertex_touched[cell->vertex_index(v)] = true;
                Point<dim> point = cell->vertex(v);
                Vector<double> tmp(dim + 1);
                VectorTools::point_value(fluid_solver.dof_handler,
                                         fluid_solver.present_solution,
                                         point,
                                         tmp);
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    localized_solid_displacement[cell->vertex_dof_index(
                      v, d)] += tmp[d] * time.get_delta_t();
                  }
              }
          }
      }
    move_solid_mesh(false);
    solid_solver.current_displacement = localized_solid_displacement;
  }

  // Dirichlet bcs are applied to artificial fluid cells, so fluid nodes
  // should be marked as artificial or real. Meanwhile, additional body force
  // is acted at the artificial fluid quadrature points. To accomodate these
  // two settings, we define indicator at quadrature points, but only when all
  // of the vertices of a fluid cell are found to be in solid domain,
  // set the indicators at all quadrature points to be 1.
  template <int dim>
  void FSI<dim>::update_indicator()
  {
    TimerOutput::Scope timer_section(timer, "Update indicator");
    move_solid_mesh(true);
    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        // Indicator is ghosted, as it will be used in constraints.
        if (f_cell->is_artificial())
          {
            continue;
          }
        auto p = fluid_solver.cell_property.get_data(f_cell);
        int inside_count = 0;
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!point_in_solid(solid_solver.dof_handler, f_cell->vertex(v)))
              {
                break;
              }
            ++inside_count;
          }
        p[0]->indicator =
          (inside_count == GeometryInfo<dim>::vertices_per_cell ? 1 : 0);
      }
    move_solid_mesh(false);
  }

  // This function interpolates the solid velocity into the fluid solver,
  // as the Dirichlet boundary conditions for artificial fluid vertices
  template <int dim>
  void FSI<dim>::find_fluid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find fluid BC");
    move_solid_mesh(true);

    // The nonzero Dirichlet BCs (to set the velocity) and zero Dirichlet
    // BCs (to set the velocity increment) for the artificial fluid domain.
    AffineConstraints<double> inner_nonzero, inner_zero;
    inner_nonzero.clear();
    inner_zero.clear();
    inner_nonzero.reinit(fluid_solver.locally_relevant_dofs);
    inner_zero.reinit(fluid_solver.locally_relevant_dofs);
    PETScWrappers::MPI::BlockVector tmp_fsi_acceleration;
    tmp_fsi_acceleration.reinit(fluid_solver.owned_partitioning,
                                fluid_solver.mpi_communicator);

    Vector<double> localized_solid_velocity(solid_solver.current_velocity);
    Vector<double> localized_solid_acceleration(
      solid_solver.current_acceleration);
    std::vector<std::vector<Vector<double>>> localized_stress(
      dim, std::vector<Vector<double>>(dim));
    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          {
            localized_stress[i][j] = solid_solver.stress[i][j];
          }
      }

    const std::vector<Point<dim>> &unit_points =
      fluid_solver.fe.get_unit_support_points();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    std::vector<SymmetricTensor<2, dim>> sym_grad_v(unit_points.size());
    std::vector<double> p(unit_points.size());
    std::vector<Tensor<2, dim>> grad_v(unit_points.size());
    std::vector<Tensor<1, dim>> v(unit_points.size());

    MappingQGeneric<dim> mapping(parameters.fluid_velocity_degree);
    Quadrature<dim> dummy_q(unit_points);
    FEValues<dim> dummy_fe_values(mapping,
                                  fluid_solver.fe,
                                  dummy_q,
                                  update_quadrature_points | update_values |
                                    update_gradients);
    std::vector<types::global_dof_index> dof_indices(
      fluid_solver.fe.dofs_per_cell);
    std::vector<unsigned int> dof_touched(fluid_solver.dof_handler.n_dofs(), 0);

    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        // Use is_artificial() instead of !is_locally_owned() because ghost
        // elements must be taken care of to set correct Dirichlet BCs!
        if (f_cell->is_artificial())
          {
            continue;
          }
        // Now skip the ghost elements because it's not store in cell property.
        if (!use_dirichlet_bc && f_cell->is_locally_owned())
          {
            auto ptr = fluid_solver.cell_property.get_data(f_cell);
            if (ptr[0]->indicator == 0)
              continue;

            auto hints = cell_hints.get_data(f_cell);
            dummy_fe_values.reinit(f_cell);
            f_cell->get_dof_indices(dof_indices);
            auto support_points = dummy_fe_values.get_quadrature_points();
            // Fluid velocity at support points
            dummy_fe_values[velocities].get_function_values(
              fluid_solver.present_solution, v);
            // Fluid velocity gradient at support points
            dummy_fe_values[velocities].get_function_gradients(
              fluid_solver.present_solution, grad_v);
            // Fluid symmetric velocity gradient at support points
            dummy_fe_values[velocities].get_function_symmetric_gradients(
              fluid_solver.present_solution, sym_grad_v);
            // Fluid pressure at support points
            dummy_fe_values[pressure].get_function_values(
              fluid_solver.present_solution, p);
            // Loop over the support points to calculate fsi acceleration.
            for (unsigned int i = 0; i < unit_points.size(); ++i)
              {
                // Skip the already-set dofs.
                if (dof_touched[dof_indices[i]] != 0)
                  continue;
                auto base_index = fluid_solver.fe.system_to_base_index(i);
                const unsigned int i_group = base_index.first.first;
                Assert(i_group < 2,
                       ExcMessage(
                         "There should be only 2 groups of finite element!"));
                if (i_group == 1)
                  continue; // skip the pressure dofs
                // Same as fluid_solver.fe.system_to_base_index(i).first.second;
                const unsigned int index =
                  fluid_solver.fe.system_to_component_index(i).first;
                Assert(index < dim,
                       ExcMessage("Vector component should be less than dim!"));
                dof_touched[dof_indices[i]] = 1;
                if (!point_in_solid(solid_solver.dof_handler,
                                    support_points[i]))
                  continue;
                Utils::CellLocator<dim, DoFHandler<dim>> locator(
                  solid_solver.dof_handler, support_points[i], *(hints[i]));
                *(hints[i]) = locator.search();
                Utils::GridInterpolator<dim, Vector<double>> interpolator(
                  solid_solver.dof_handler, support_points[i], {}, *(hints[i]));
                if (!interpolator.found_cell())
                  {
                    std::stringstream message;
                    message
                      << "Cannot find point in solid: " << support_points[i]
                      << std::endl;
                    AssertThrow(interpolator.found_cell(),
                                ExcMessage(message.str()));
                  }
                // Solid acceleration at fluid unit point
                Vector<double> solid_acc(dim);
                Vector<double> solid_vel(dim);
                interpolator.point_value(localized_solid_acceleration,
                                         solid_acc);
                interpolator.point_value(localized_solid_velocity, solid_vel);
                Tensor<1, dim> vs;
                for (int j = 0; j < dim; ++j)
                  {
                    vs[j] = solid_vel[j];
                  }
                // Fluid total acceleration at support points
                Tensor<1, dim> fluid_acc =
                  (vs - v[i]) / time.get_delta_t() + grad_v[i] * v[i];
                auto line = dof_indices[i];
                // Note that we are setting the value of the constraint to the
                // velocity delta!
                tmp_fsi_acceleration(line) =
                  fluid_acc[index] - solid_acc[index];
              }
          }
        // Dirichlet BCs
        if (use_dirichlet_bc)
          {
            dummy_fe_values.reinit(f_cell);
            f_cell->get_dof_indices(dof_indices);
            auto support_points = dummy_fe_values.get_quadrature_points();
            auto hints = cell_hints.get_data(f_cell);
            // Declare the fluid velocity for interpolating BC
            Vector<double> fluid_velocity(dim);
            // Loop over the support points to set Dirichlet BCs.
            for (unsigned int i = 0; i < unit_points.size(); ++i)
              {
                // Skip the already-set dofs.
                if (dof_touched[dof_indices[i]] != 0)
                  continue;
                auto base_index = fluid_solver.fe.system_to_base_index(i);
                const unsigned int i_group = base_index.first.first;
                Assert(i_group < 2,
                       ExcMessage(
                         "There should be only 2 groups of finite element!"));
                if (i_group == 1)
                  continue; // skip the pressure dofs
                int inside_dim_count = 0;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    if (0 < std::abs(unit_points[i][d]) &&
                        std::abs(unit_points[i][d]) < 1)
                      {
                        inside_dim_count++;
                      }
                  }
                if (inside_dim_count == dim)
                  {
                    continue; // skip the in-cell support point
                  }
                // Same as fluid_solver.fe.system_to_base_index(i).first.second;
                const unsigned int index =
                  fluid_solver.fe.system_to_component_index(i).first;
                Assert(index < dim,
                       ExcMessage("Vector component should be less than dim!"));
                dof_touched[dof_indices[i]] = 1;
                if (!point_in_solid(solid_solver.dof_handler,
                                    support_points[i]))
                  continue;
                Utils::CellLocator<dim, DoFHandler<dim>> locator(
                  solid_solver.dof_handler, support_points[i], *(hints[i]));
                *(hints[i]) = locator.search();
                Utils::GridInterpolator<dim, Vector<double>> interpolator(
                  solid_solver.dof_handler, support_points[i], {}, *(hints[i]));
                if (!interpolator.found_cell())
                  {
                    std::stringstream message;
                    message
                      << "Cannot find point in solid: " << support_points[i]
                      << std::endl;
                    AssertThrow(interpolator.found_cell(),
                                ExcMessage(message.str()));
                  }
                interpolator.point_value(localized_solid_velocity,
                                         fluid_velocity);
                auto line = dof_indices[i];
                inner_nonzero.add_line(line);
                inner_zero.add_line(line);
                // Note that we are setting the value of the constraint to the
                // velocity delta!
                inner_nonzero.set_inhomogeneity(
                  line,
                  fluid_velocity[index] - fluid_solver.present_solution(line));
              }
          }
      }
    tmp_fsi_acceleration.compress(VectorOperation::insert);
    fluid_solver.fsi_acceleration = tmp_fsi_acceleration;
    if (use_dirichlet_bc)
      {
        inner_nonzero.close();
        inner_zero.close();
        fluid_solver.nonzero_constraints.merge(
          inner_nonzero,
          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        fluid_solver.zero_constraints.merge(
          inner_zero,
          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
      }

    // If the fluid solver has a turbulence model, update the cell data in the
    // turbulence model
    if (auto SA_model = dynamic_cast<Fluid::MPI::SpalartAllmaras<dim> *>(
          fluid_solver.turbulence_model.get()))
      {
        SA_model->update_moving_wall_distance(solid_boundary_vertices);
      }

    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::find_solid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find solid BC");
    // Must use the updated solid coordinates
    move_solid_mesh(true);
    // Fluid FEValues to do interpolation
    FEValues<dim> fe_values(
      fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);

    for (unsigned int d = 0; d < dim; ++d)
      {
        solid_solver.fsi_stress_rows[d] = 0;
      }

    /**
     * A relevantly paritioned copy of nodal strain and stress obtained by
     * taking the average of surrounding cell-averaged strains and stresses.
     * They are used for stress interpolation.
     */
    std::vector<std::vector<PETScWrappers::MPI::Vector>>
      relevant_partition_stress =
        std::vector<std::vector<PETScWrappers::MPI::Vector>>(
          dim,
          std::vector<PETScWrappers::MPI::Vector>(
            dim,
            PETScWrappers::MPI::Vector(
              fluid_solver.locally_owned_scalar_dofs,
              fluid_solver.locally_relevant_scalar_dofs,
              mpi_communicator)));
    relevant_partition_stress = fluid_solver.stress;

    for (auto s_cell = solid_solver.dof_handler.begin_active(),
              scalar_s_cell = solid_solver.scalar_dof_handler.begin_active();
         s_cell != solid_solver.dof_handler.end();
         ++s_cell, ++scalar_s_cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            // Current face is at boundary and without Dirichlet bc.
            if (s_cell->face(f)->at_boundary())
              {
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_face;
                     ++v)
                  {
                    auto line = s_cell->face(f)->vertex_dof_index(v, 0);
                    auto scalar_line =
                      scalar_s_cell->face(f)->vertex_dof_index(v, 0);
                    // Get interpolated solution from the fluid
                    Vector<double> value(dim + 1);
                    Utils::GridInterpolator<dim,
                                            PETScWrappers::MPI::BlockVector>
                      interpolator(fluid_solver.dof_handler,
                                   s_cell->face(f)->vertex(v),
                                   vertices_mask);
                    interpolator.point_value(fluid_solver.present_solution,
                                             value);
                    // Compute the viscous traction
                    SymmetricTensor<2, dim> viscous_stress;
                    // Create the scalar interpolator for stresses based on the
                    // existing interpolator
                    auto f_cell = interpolator.get_cell();
                    if (f_cell.state() == IteratorState::IteratorStates::valid)
                      {
                        TriaActiveIterator<DoFCellAccessor<dim, dim, false>>
                          scalar_f_cell(&fluid_solver.triangulation,
                                        f_cell->level(),
                                        f_cell->index(),
                                        &fluid_solver.scalar_dof_handler);
                        Utils::GridInterpolator<dim, PETScWrappers::MPI::Vector>
                          scalar_interpolator(fluid_solver.scalar_dof_handler,
                                              s_cell->face(f)->vertex(v),
                                              vertices_mask,
                                              scalar_f_cell);
                        for (unsigned int i = 0; i < dim; i++)
                          {
                            for (unsigned int j = i; j < dim; j++)
                              {
                                Vector<double> stress_component(1);
                                scalar_interpolator.point_value(
                                  relevant_partition_stress[i][j],
                                  stress_component);
                                viscous_stress[i][j] = stress_component[0];
                              }
                          }
                      }
                    // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                    SymmetricTensor<2, dim> stress =
                      -value[dim] *
                        Physics::Elasticity::StandardTensors<dim>::I +
                      viscous_stress;
                    // Assign the cell stress to local row vectors
                    for (unsigned int d1 = 0; d1 < dim; ++d1)
                      {
                        for (unsigned int d2 = 0; d2 < dim; ++d2)
                          {
                            // fluid stress for traction computation
                            solid_solver.fsi_stress_rows[d1][line + d2] =
                              stress[d1][d2];
                          }
                        // fluid velocity for friction work computation
                        solid_solver.fluid_velocity[line + d1] = value[d1];
                      } // End assigning local fluid stress values
                    // fluid pressure for drag computation
                    solid_solver.fluid_pressure[scalar_line] = value[dim];
                  } // End looping support points
              }
          } // End looping cell faces
      }     // End looping solid cells
    // Add up the local vectors
    for (unsigned int d = 0; d < dim; ++d)
      {
        Utilities::MPI::sum(solid_solver.fsi_stress_rows[d],
                            solid_solver.mpi_communicator,
                            solid_solver.fsi_stress_rows[d]);
      }
    Utilities::MPI::sum(solid_solver.fluid_velocity,
                        solid_solver.mpi_communicator,
                        solid_solver.fluid_velocity);
    Utilities::MPI::sum(solid_solver.fluid_pressure,
                        solid_solver.mpi_communicator,
                        solid_solver.fluid_pressure);
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::apply_contact_model(bool first_step)
  {
    AssertThrow(penetration_criterion != nullptr,
                ExcMessage("No penetration criterion specified!"));
    // We need to increment the force until it does not penetrate
    bool still_penetrate = true;
    double force_increment = parameters.contact_force_multiplier;
    // Cache the current solutions
    Vector<double> cached_current_acceleration(
      solid_solver.current_acceleration);
    Vector<double> cached_current_velocity(solid_solver.current_velocity);
    Vector<double> cached_current_displacement(
      solid_solver.current_displacement);
    Vector<double> cached_previous_acceleration(
      solid_solver.previous_acceleration);
    Vector<double> cached_previous_velocity(solid_solver.previous_velocity);
    Vector<double> cached_previous_displacement(
      solid_solver.previous_displacement);
    // By default, the force to mimic contact model is towards
    // the bottom.
    Tensor<1, dim> traction;

    FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                     solid_solver.face_quad_formula,
                                     update_normal_vectors |
                                       update_quadrature_points);
    while (still_penetrate)
      {
        // Reset the flag
        still_penetrate = false;
        solid_solver.run_one_step(first_step);
        move_solid_mesh(true);
        // Loop over the solid cells and determine if they penetrate
        for (auto s_cell = solid_solver.dof_handler.begin_active();
             s_cell != solid_solver.dof_handler.end();
             ++s_cell)
          {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                // Current face is at boundary and without Dirichlet bc.
                if (s_cell->face(f)->at_boundary())
                  {
                    fe_face_values.reinit(s_cell, f);
                    for (unsigned int v = 0;
                         v < GeometryInfo<dim>::vertices_per_face;
                         ++v)
                      {
                        // Check if the vertex is penetrating
                        double penetration_value = std::invoke(
                          *penetration_criterion, s_cell->face(f)->vertex(v));
                        if (penetration_value > 1e-5)
                          {
                            still_penetrate = true;
                            traction = force_increment * penetration_value /
                                       penetration_direction.norm() *
                                       penetration_direction;
                          }
                        else
                          {
                            continue;
                          }
                        auto line = s_cell->face(f)->vertex_dof_index(v, 0);
                        // Compute the extra stress from the face
                        Tensor<2, dim> extra_stress;
                        for (unsigned int d = 0; d < dim; ++d)
                          {
                            extra_stress[d][dim - 1] =
                              fe_face_values.normal_vector(0)[d] > 1e-5
                                ? traction[d] /
                                    fe_face_values.normal_vector(0)[d]
                                : 0;
                          }
                        // Assign the extra stress to local row vectors
                        for (unsigned int d1 = 0; d1 < dim; ++d1)
                          {
                            for (unsigned int d2 = 0; d2 < dim; ++d2)
                              {
                                solid_solver.fsi_stress_rows[d1][line + d2] +=
                                  extra_stress[d1][d2];
                              }
                          }
                        // End assigning local fluid stress values
                      } // End looping support points
                  }
              } // End looping cell faces
          }     // End looping solid cells
        move_solid_mesh(false);
        if (still_penetrate)
          {
            pcout << "Penetrating, apply contact model!" << std::endl;
            solid_solver.current_acceleration = cached_current_acceleration;
            solid_solver.current_velocity = cached_current_velocity;
            solid_solver.current_displacement = cached_current_displacement;
            solid_solver.previous_acceleration = cached_previous_acceleration;
            solid_solver.previous_velocity = cached_previous_velocity;
            solid_solver.previous_displacement = cached_previous_displacement;
            solid_solver.time.decrement();
          }
      } // End adding extra stress
  }

  template <int dim>
  void FSI<dim>::collect_solid_boundary_vertices()
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
      {
        return;
      }
    unsigned fixed_bc_flag = (1 << dim) - 1;

    for (auto &face : solid_solver.triangulation.active_face_iterators())
      {
        if (face->at_boundary())
          {
            // Check if the boundary is fixed
            auto bc = parameters.solid_dirichlet_bcs.find(face->boundary_id());
            if (bc != parameters.solid_dirichlet_bcs.end() &&
                bc->second == fixed_bc_flag)
              {
                // Skip those fixed vertices
                continue;
              }

            for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_face; ++v)
              {
                solid_boundary_vertices.insert(face->vertex_iterator(v));
              }
          }
      }
  }

  template <int dim>
  void FSI<dim>::refine_mesh(const unsigned int min_grid_level,
                             const unsigned int max_grid_level)
  {
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    move_solid_mesh(true);
    std::vector<Point<dim>> solid_boundary_points;
    for (auto s_cell : solid_solver.dof_handler.active_cell_iterators())
      {
        bool is_boundary = false;
        Point<dim> point;
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          {
            if (s_cell->face(face)->at_boundary())
              {
                point = s_cell->face(face)->center();
                is_boundary = true;
                break;
              }
          }
        if (is_boundary)
          {
            solid_boundary_points.push_back(point);
          }
      }
    for (auto f_cell : fluid_solver.dof_handler.active_cell_iterators())
      {
        auto center = f_cell->center();
        double dist{std::numeric_limits<double>::max()};
        for (auto point : solid_boundary_points)
          {
            dist = std::min(center.distance(point), dist);
          }
        if (dist < f_cell->diameter())
          f_cell->set_refine_flag();
        else
          f_cell->set_coarsen_flag();
      }
    move_solid_mesh(false);
    if (fluid_solver.triangulation.n_levels() > max_grid_level)
      {
        for (auto cell =
               fluid_solver.triangulation.begin_active(max_grid_level);
             cell != fluid_solver.triangulation.end();
             ++cell)
          {
            cell->clear_refine_flag();
          }
      }

    for (auto cell = fluid_solver.triangulation.begin_active(min_grid_level);
         cell != fluid_solver.triangulation.end_active(min_grid_level);
         ++cell)
      {
        cell->clear_coarsen_flag();
      }

    parallel::distributed::SolutionTransfer<dim,
                                            PETScWrappers::MPI::BlockVector>
      solution_transfer(fluid_solver.dof_handler);

    fluid_solver.triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(
      fluid_solver.present_solution);

    // Preparation for turbulence model
    std::optional<
      parallel::distributed::SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
      turbulence_trans;
    if (fluid_solver.turbulence_model)
      {
        fluid_solver.turbulence_model->pre_refine_mesh(turbulence_trans);
      }

    fluid_solver.triangulation.execute_coarsening_and_refinement();

    fluid_solver.setup_dofs();
    fluid_solver.make_constraints();
    fluid_solver.initialize_system();

    PETScWrappers::MPI::BlockVector buffer;
    buffer.reinit(fluid_solver.owned_partitioning,
                  fluid_solver.mpi_communicator);
    buffer = 0;
    solution_transfer.interpolate(buffer);
    fluid_solver.present_solution = buffer;
    update_vertices_mask();

    // Transfer solution for turbulence model
    if (fluid_solver.turbulence_model)
      {
        fluid_solver.turbulence_model->post_refine_mesh(turbulence_trans);
      }
  }

  template <int dim>
  void FSI<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
    // Try load from previous computation.
    bool success_load =
      solid_solver.load_checkpoint() && fluid_solver.load_checkpoint();
    AssertThrow(
      solid_solver.time.current() == fluid_solver.time.current(),
      ExcMessage("Solid and fluid restart files have different time steps. "
                 "Check and remove inconsistent restart files!"));
    if (!success_load)
      {
        solid_solver.setup_dofs();
        solid_solver.initialize_system();
        fluid_solver.triangulation.refine_global(
          parameters.global_refinements[0]);
        fluid_solver.setup_dofs();
        fluid_solver.make_constraints();
        fluid_solver.initialize_system();
      }
    else
      {
        while (time.get_timestep() < solid_solver.time.get_timestep())
          {
            time.increment();
          }
      }

    collect_solid_boundaries();
    collect_solid_boundary_vertices();
    setup_cell_hints();
    update_vertices_mask();

    pcout << "Number of fluid active cells and dofs: ["
          << fluid_solver.triangulation.n_active_cells() << ", "
          << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
          << "Number of solid active cells and dofs: ["
          << solid_solver.triangulation.n_active_cells() << ", "
          << solid_solver.dof_handler.n_dofs() << "]" << std::endl;
    bool first_step = !success_load;
    if (parameters.refinement_interval < parameters.end_time)
      {
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        refine_mesh(parameters.global_refinements[0],
                    parameters.global_refinements[0] + 3);
        setup_cell_hints();
      }
    while (time.end() - time.current() > 1e-12)
      {
        find_solid_bc();
        if (success_load)
          {
            solid_solver.assemble_system(true);
          }
        {
          TimerOutput::Scope timer_section(timer, "Run solid solver");
          if (penetration_criterion)
            {
              apply_contact_model(first_step);
            }
          else
            {
              solid_solver.run_one_step(first_step);
            }
        }
        update_solid_box();
        update_indicator();
        fluid_solver.make_constraints();
        if (!first_step)
          {
            fluid_solver.nonzero_constraints.clear();
            fluid_solver.nonzero_constraints.copy_from(
              fluid_solver.zero_constraints);
          }
        if (fluid_solver.turbulence_model)
          {
            fluid_solver.turbulence_model->update_boundary_condition(
              first_step);
          }
        find_fluid_bc();
        {
          TimerOutput::Scope timer_section(timer, "Run fluid solver");
          if (fluid_solver.turbulence_model)
            {
              fluid_solver.turbulence_model->run_one_step(true);
            }
          fluid_solver.run_one_step(true);
        }
        first_step = false;
        time.increment();
        if (time.time_to_refine())
          {
            refine_mesh(parameters.global_refinements[0],
                        parameters.global_refinements[0] + 3);
            setup_cell_hints();
          }
        if (time.time_to_save())
          {
            solid_solver.save_checkpoint(time.get_timestep());
            fluid_solver.save_checkpoint(time.get_timestep());
          }
      }
  }

  template <int dim>
  void FSI<dim>::set_penetration_criterion(
    const std::function<double(const Point<dim> &)> &criterion,
    Tensor<1, dim> direction)
  {
    penetration_criterion.reset(
      new std::function<double(const Point<dim> &)>(criterion));
    penetration_direction = direction;
  }

  template class FSI<2>;
  template class FSI<3>;
} // namespace MPI

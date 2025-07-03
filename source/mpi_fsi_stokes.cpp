#include "mpi_fsi_stokes.h"
#include "mpi_spalart_allmaras.h"
#include <iostream>

// for building in some systems
#include <optional>

namespace MPI
{
  template <int dim>
  FSI_stokes<dim>::~FSI_stokes()
  {
    timer.print_summary();
  }

  template <int dim>
  FSI_stokes<dim>::FSI_stokes(Fluid::MPI::FluidSolver<dim> &f,
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
  void FSI_stokes<dim>::move_solid_mesh(bool move_forward)
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
  void FSI_stokes<dim>::collect_solid_boundaries()
  {
    for (auto cell = solid_solver.triangulation.begin_active();
         cell != solid_solver.triangulation.end();
         ++cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                solid_boundaries.push_back({cell, f});
              }
          }
      }
  }

  template <int dim>
  void FSI_stokes<dim>::update_solid_box()
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
  void FSI_stokes<dim>::update_vertices_mask()
  {
    // Initialize vertices mask
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
  bool FSI_stokes<dim>::point_in_solid(const DoFHandler<dim> &df,
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
        for (auto &[f_cell, face] : solid_boundaries)
          {
            auto f = f_cell->face(face);
            Point<dim> p1 = f->vertex(0), p2 = f->vertex(1);
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
  void FSI_stokes<dim>::setup_cell_hints()
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
  void FSI_stokes<dim>::update_solid_displacement()
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
  // is acted at the artificial fluid quadrature points. To accommodate these
  // two settings, we define indicator at quadrature points, but only when all
  // of the vertices of a fluid cell are found to be in solid domain,
  // set the indicators at all quadrature points to be 1.
  template <int dim>
  void FSI_stokes<dim>::update_indicator()
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

  template <int dim>
  void FSI_stokes<dim>::build_projection_constraints()
  {

    const unsigned int dofs_per_cell = fluid_solver.fe.dofs_per_cell;

    IndexSet arti_dofs(fluid_solver.dof_handler.n_dofs());

    for (auto cell : fluid_solver.dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() &&
          fluid_solver.cell_property.get_data(cell)[0]->indicator == 1)
        {
          std::vector<types::global_dof_index> local(dofs_per_cell);
          cell->get_dof_indices(local);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (fluid_solver.fe.system_to_component_index(i).first < dim)
              arti_dofs.add_index(local[i]);
        }
    arti_dofs.compress();

    IndexSet zero_velocity_dofs(fluid_solver.dof_handler.n_dofs());

    for (auto cell = fluid_solver.dof_handler.begin_active();
         cell != fluid_solver.dof_handler.end();
         ++cell)
      {
        if (cell->is_locally_owned())
          {
            const auto property = fluid_solver.cell_property.get_data(cell);
            if (property[0]->indicator == 0)
              {
                std::vector<types::global_dof_index> local_dof_indices(
                  dofs_per_cell);
                cell->get_dof_indices(local_dof_indices);
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    if (fluid_solver.fe.system_to_component_index(i).first <
                        dim)
                      {
                        if (!arti_dofs.is_element(local_dof_indices[i]))
                          {
                            zero_velocity_dofs.add_index(local_dof_indices[i]);
                          }
                      }
                  }
              }
          }
      }
    zero_velocity_dofs.compress();

    projection_constraints.clear();
    DoFTools::make_hanging_node_constraints(fluid_solver.dof_handler,
                                            projection_constraints);

    for (const auto idx : zero_velocity_dofs)
      {
        projection_constraints.add_line(idx);
        projection_constraints.set_inhomogeneity(idx, 0.0);
      }
    projection_constraints.close();
  }

  template <int dim>
  void FSI_stokes<dim>::build_arti_mass_matrix()
  {
    build_projection_constraints();

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fluid_solver.fe.dofs_per_cell;
    const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();

    // Identify velocity DoFs once
    std::vector<unsigned int> velocity_local_indices;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      if (fluid_solver.fe.system_to_component_index(i).first < dim)
        velocity_local_indices.push_back(i);
    const unsigned int n_velocity_dofs = velocity_local_indices.size();

    DynamicSparsityPattern dsp(fluid_solver.relevant_partitioning[0].size(),
                               fluid_solver.relevant_partitioning[0].size());

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim; ++c)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            coupling[c][d] = DoFTools::always;
          }
      }

    for (unsigned int c = dim; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            coupling[c][d] = DoFTools::none;
          }
      }

    DoFTools::make_sparsity_pattern(
      fluid_solver.dof_handler,
      coupling,
      dsp,
      // empty_constraints,
      projection_constraints,
      false,
      Utilities::MPI::this_mpi_process(mpi_communicator));

    SparsityTools::distribute_sparsity_pattern(
      dsp,
      fluid_solver.owned_partitioning[0],
      mpi_communicator,
      fluid_solver.relevant_partitioning[0]);

    dsp.compress();

    mass_matrix_velocity.reinit(fluid_solver.owned_partitioning[0],
                                fluid_solver.owned_partitioning[0],
                                dsp,
                                mpi_communicator);

    mass_matrix_velocity = 0.0;

    FullMatrix<double> local_mass(n_velocity_dofs, n_velocity_dofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> velocity_global_indices(
      n_velocity_dofs);

    for (auto cell = fluid_solver.dof_handler.begin_active();
         cell != fluid_solver.dof_handler.end();
         ++cell)

      {
        if (!(cell->is_locally_owned() &&
              fluid_solver.cell_property.get_data(cell)[0]->indicator == 1))
          {
            continue;
          }

        fe_values.reinit(cell);
        local_mass = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < n_velocity_dofs; ++k)
              {
                unsigned int i = velocity_local_indices[k];
                for (unsigned int l = 0; l < n_velocity_dofs; ++l)
                  {
                    unsigned int j = velocity_local_indices[l];

                    double val_dot = 0.0;

                    for (unsigned int c = 0; c < dim; ++c)
                      {
                        const double phi_i_c =
                          fe_values.shape_value_component(i, q, c);
                        const double phi_j_c =
                          fe_values.shape_value_component(j, q, c);
                        val_dot += phi_i_c * phi_j_c;
                      }

                    local_mass(k, l) += val_dot * fe_values.JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int k = 0; k < n_velocity_dofs; ++k)
          velocity_global_indices[k] =
            local_dof_indices[velocity_local_indices[k]];

        projection_constraints.distribute_local_to_global(
          local_mass, velocity_global_indices, mass_matrix_velocity);
      }
    mass_matrix_velocity.compress(VectorOperation::add);

    std::vector<types::global_dof_index> constrained_rows;
    constrained_rows.reserve(projection_constraints.n_constraints());

    for (const auto &c : projection_constraints.get_lines())
      {
        constrained_rows.push_back(c.index);
      }

    mass_matrix_velocity.clear_rows(constrained_rows, 1.0);
    mass_matrix_velocity.compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI_stokes<dim>::build_arti_scalar_mass_matrix()
  {

    const unsigned int dofs_per_cell = fluid_solver.scalar_fe.dofs_per_cell;

    IndexSet arti_scalar_dofs(fluid_solver.scalar_dof_handler.n_dofs());

    for (auto cell : fluid_solver.scalar_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() &&
            fluid_solver.cell_property.get_data(cell)[0]->indicator == 1)
          {
            std::vector<types::global_dof_index> local(dofs_per_cell);
            cell->get_dof_indices(local);
            for (auto id : local)
              {
                arti_scalar_dofs.add_index(id);
              }
          }
      }

    arti_scalar_dofs.compress();

    IndexSet zero_scalar_dofs(fluid_solver.scalar_dof_handler.n_dofs());

    for (auto cell : fluid_solver.scalar_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            const auto prop = fluid_solver.cell_property.get_data(cell);
            if (prop[0]->indicator == 0)
              {
                std::vector<types::global_dof_index> local(dofs_per_cell);
                cell->get_dof_indices(local);
                for (auto id : local)
                  {
                    if (!arti_scalar_dofs.is_element(id))
                      {
                        zero_scalar_dofs.add_index(id);
                      }
                  }
              }
          }
      }

    zero_scalar_dofs.compress();

    scalar_projection_constraints.clear();

    DoFTools::make_hanging_node_constraints(fluid_solver.scalar_dof_handler,
                                            scalar_projection_constraints);

    for (const auto id : zero_scalar_dofs)
      {
        scalar_projection_constraints.add_line(id);
        scalar_projection_constraints.set_inhomogeneity(id, 0.0);
      }

    scalar_projection_constraints.close();

    DynamicSparsityPattern dsp(fluid_solver.locally_relevant_scalar_dofs);

    DoFTools::make_sparsity_pattern(
      fluid_solver.scalar_dof_handler,
      dsp,
      scalar_projection_constraints,
      false,
      Utilities::MPI::this_mpi_process(mpi_communicator));

    SparsityTools::distribute_sparsity_pattern(
      dsp,
      fluid_solver.locally_owned_scalar_dofs,
      mpi_communicator,
      fluid_solver.locally_relevant_scalar_dofs);

    dsp.compress();

    mass_matrix_scalar.reinit(fluid_solver.locally_owned_scalar_dofs,
                              fluid_solver.locally_owned_scalar_dofs,
                              dsp,
                              mpi_communicator);

    mass_matrix_scalar = 0.0;

    FEValues<dim> fev(fluid_solver.scalar_fe,
                      fluid_solver.volume_quad_formula,
                      update_values | update_JxW_values);

    FullMatrix<double> local(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> ldofs(dofs_per_cell);

    for (auto cell : fluid_solver.scalar_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() &&
            fluid_solver.cell_property.get_data(cell)[0]->indicator == 1)
          {
            fev.reinit(cell);
            local = 0;
            for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local(i, j) += fev.shape_value(i, q) *
                                       fev.shape_value(j, q) * fev.JxW(q);
                      }
                  }
              }
            cell->get_dof_indices(ldofs);
            scalar_projection_constraints.distribute_local_to_global(
              local, ldofs, mass_matrix_scalar);
          }
      }
    mass_matrix_scalar.compress(VectorOperation::add);

    std::vector<types::global_dof_index> constrained_rows;
    constrained_rows.reserve(scalar_projection_constraints.n_constraints());
    for (const auto &c : scalar_projection_constraints.get_lines())
      {
        constrained_rows.push_back(c.index);
      }
    mass_matrix_scalar.clear_rows(constrained_rows, 1.0);
    mass_matrix_scalar.compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI_stokes<dim>::build_surface_mass_matrix()
  {

    IndexSet surface_dofs(solid_solver.dof_handler.n_dofs());

    for (const auto &cell : solid_solver.dof_handler.active_cell_iterators())
      {
        if (cell->subdomain_id() == solid_solver.this_mpi_process)
          {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (cell->face(f)->at_boundary())
                  {
                    std::vector<types::global_dof_index> face_dof_indices(
                      cell->get_fe().dofs_per_face);

                    cell->face(f)->get_dof_indices(face_dof_indices);

                    for (const auto id : face_dof_indices)
                      {
                        surface_dofs.add_index(id);
                      }
                  }
              }
          }
      }

    surface_dofs.compress();

    IndexSet interior_dofs = solid_solver.locally_owned_dofs;

    interior_dofs.subtract_set(surface_dofs);

    surface_constraints.clear();

    DoFTools::make_hanging_node_constraints(solid_solver.dof_handler,
                                            surface_constraints);

    for (const auto id : interior_dofs)
      {
        surface_constraints.add_line(id);
        surface_constraints.set_inhomogeneity(id, 0.);
      }

    surface_constraints.close();

    DynamicSparsityPattern dsp(solid_solver.dof_handler.n_dofs(),
                               solid_solver.dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(solid_solver.dof_handler,
                                    dsp,
                                    surface_constraints,
                                    /*keep_constrained_dofs =*/false);

    dsp.compress();

    surface_mass_matrix.reinit(solid_solver.locally_owned_dofs,
                               solid_solver.locally_owned_dofs,
                               dsp,
                               mpi_communicator);

    surface_mass_matrix = 0.0;

    FEFaceValues<dim> fe_face(solid_solver.fe,
                              solid_solver.face_quad_formula,
                              update_values | update_JxW_values);

    const unsigned dofs_per_cell = solid_solver.fe.dofs_per_cell;
    const unsigned n_q_face = solid_solver.face_quad_formula.size();
    FullMatrix<double> local(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> ldof(dofs_per_cell);

    for (const auto &cell : solid_solver.dof_handler.active_cell_iterators())
      {
        if (cell->subdomain_id() == solid_solver.this_mpi_process)
          {
            for (unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (cell->face(f)->at_boundary())
                  {
                    fe_face.reinit(cell, f);
                    local = 0.0;
                    cell->get_dof_indices(ldof);

                    for (unsigned q = 0; q < n_q_face; ++q)
                      {
                        for (unsigned i = 0; i < dofs_per_cell; ++i)
                          {
                            if (!surface_dofs.is_element(ldof[i]))
                              {
                                continue;
                              }

                            for (unsigned j = 0; j < dofs_per_cell; ++j)
                              {
                                if (!surface_dofs.is_element(ldof[j]))
                                  {
                                    continue;
                                  }
                                double val = 0.0;
                                for (unsigned int c = 0; c < dim; ++c)
                                  {
                                    const double phi_i =
                                      fe_face.shape_value_component(i, q, c);
                                    const double phi_j =
                                      fe_face.shape_value_component(j, q, c);
                                    val += phi_i * phi_j;
                                  }
                                local(i, j) += val * fe_face.JxW(q);
                              }
                          }
                      }

                    surface_constraints.distribute_local_to_global(
                      local, ldof, surface_mass_matrix);
                  }
              }
          }
      }
    surface_mass_matrix.compress(VectorOperation::add);

    IndexSet interior_owned = interior_dofs & solid_solver.locally_owned_dofs;

    std::vector<types::global_dof_index> interior_rows;

    interior_rows.reserve(interior_owned.n_elements());

    for (auto it = interior_owned.begin(); it != interior_owned.end(); ++it)
      {
        interior_rows.push_back(*it);
      }

    surface_mass_matrix.clear_rows(interior_rows, 1.0);

    surface_mass_matrix.compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI_stokes<dim>::compute_fluid_traction_projection()
  {
    build_surface_mass_matrix();

    for (unsigned d = 0; d < dim; ++d)
      {
        projected_fluid_traction[d] = 0.0;
      }

    std::array<PETScWrappers::MPI::Vector, dim> rhs;
    for (unsigned d = 0; d < dim; ++d)
      {
        rhs[d].reinit(solid_solver.locally_owned_dofs, mpi_communicator);
      }

    std::vector<std::vector<PETScWrappers::MPI::Vector>> fluid_stress(
      dim,
      std::vector<PETScWrappers::MPI::Vector>(
        dim,
        PETScWrappers::MPI::Vector(fluid_solver.locally_owned_scalar_dofs,
                                   fluid_solver.locally_relevant_scalar_dofs,
                                   mpi_communicator)));
    fluid_stress = fluid_solver.stress;

    FEFaceValues<dim> solid_face(solid_solver.fe,
                                 solid_solver.face_quad_formula,
                                 update_values | update_quadrature_points |
                                   update_normal_vectors | update_JxW_values);

    const unsigned int n_q_face = solid_solver.face_quad_formula.size();
    const unsigned int dofs_per_cell_s = solid_solver.fe.dofs_per_cell;
    std::vector<types::global_dof_index> s_dofs(dofs_per_cell_s);

    std::vector<Vector<double>> local_rhs(dim, Vector<double>(dofs_per_cell_s));

    for (const auto &s_cell : solid_solver.dof_handler.active_cell_iterators())
      {
        if (s_cell->subdomain_id() == solid_solver.this_mpi_process)
          {
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                if (s_cell->face(f)->at_boundary())
                  {
                    solid_face.reinit(s_cell, f);
                    s_cell->get_dof_indices(s_dofs);

                    for (unsigned d = 0; d < dim; ++d)
                      {
                        local_rhs[d] = 0.0;
                      }

                    for (unsigned int q = 0; q < n_q_face; ++q)
                      {
                        const Point<dim> x_q = solid_face.quadrature_point(q);
                        const Tensor<1, dim> n_q = solid_face.normal_vector(q);

                        Vector<double> value(dim + 1);

                        Utils::GridInterpolator<dim,
                                                PETScWrappers::MPI::BlockVector>
                          interp_vec(
                            fluid_solver.dof_handler, x_q, vertices_mask);

                        if (!interp_vec.found_cell())
                          {
                            continue;
                          }

                        interp_vec.point_value(fluid_solver.present_solution,
                                               value);

                        const double p_q = value[dim];

                        SymmetricTensor<2, dim> visc_stress;
                        Utils::GridInterpolator<dim, PETScWrappers::MPI::Vector>
                          interp_scalar(fluid_solver.scalar_dof_handler, x_q);

                        for (unsigned int i = 0, c = 0; i < dim; ++i)
                          {
                            for (unsigned int j = i; j < dim; ++j, ++c)
                              {
                                Vector<double> s_comp(1);
                                interp_scalar.point_value(fluid_stress[i][j],
                                                          s_comp);
                                visc_stress[i][j] = s_comp[0];
                              }
                          }

                        SymmetricTensor<2, dim> total =
                          visc_stress -
                          p_q * Physics::Elasticity::StandardTensors<dim>::I;
                        const Tensor<1, dim> t_q = total * n_q;

                        for (unsigned int i = 0; i < dofs_per_cell_s; ++i)
                          {

                            const unsigned int comp_i =
                              solid_solver.fe.system_to_component_index(i)
                                .first;
                            for (unsigned int d = 0; d < dim; ++d)
                              {
                                if (comp_i != d)
                                  {
                                    continue;
                                  }

                                const double phi_i =
                                  solid_face.shape_value_component(i, q, d);
                                local_rhs[d][i] +=
                                  t_q[d] * phi_i * solid_face.JxW(q);
                              }
                          }
                      }
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        surface_constraints.distribute_local_to_global(
                          local_rhs[d], s_dofs, rhs[d]);
                      }
                  }
              }
          }
      }
    for (auto &v : rhs)
      {
        v.compress(VectorOperation::add);
      }

    SolverControl ctl(surface_mass_matrix.m(), 1e-12);
    PETScWrappers::SparseDirectMUMPS solver(ctl, mpi_communicator);

    for (unsigned d = 0; d < dim; ++d)
      {
        solver.solve(surface_mass_matrix, projected_fluid_traction[d], rhs[d]);
        surface_constraints.distribute(projected_fluid_traction[d]);
        projected_fluid_traction[d].compress(VectorOperation::insert);
      }
  }

  template <int dim>
  void FSI_stokes<dim>::compute_stress_l2_projection()
  {
    build_arti_scalar_mass_matrix();

    std::vector<std::vector<Vector<double>>> loc_sigma(
      dim, std::vector<Vector<double>>(dim));

    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = i; j < dim; ++j)
          {
            loc_sigma[i][j] = solid_solver.stress[i][j];
          }
      }

    const unsigned s = dim + dim * (dim - 1) / 2;
    Assert(projected_solid_stress.size() == s, ExcInternalError());

    std::vector<PETScWrappers::MPI::Vector> rhs(
      s,
      PETScWrappers::MPI::Vector(fluid_solver.locally_owned_scalar_dofs,
                                 mpi_communicator));

    FEValues<dim> fev(fluid_solver.scalar_fe,
                      fluid_solver.volume_quad_formula,
                      update_values | update_quadrature_points |
                        update_JxW_values);

    const unsigned dofs_per_cell = fluid_solver.scalar_fe.dofs_per_cell;
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> ldofs(dofs_per_cell);

    for (auto cell : fluid_solver.scalar_dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() &&
            fluid_solver.cell_property.get_data(cell)[0]->indicator == 1)
          {
            fev.reinit(cell);
            cell->get_dof_indices(ldofs);

            for (unsigned comp = 0, i = 0; i < dim; ++i)
              {
                for (unsigned j = 0; j <= i; ++j, ++comp)
                  {
                    local_rhs = 0;
                    for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
                      {
                        const Point<dim> x_q = fev.quadrature_point(q);
                        Vector<double> sig_ij(1);

                        Utils::GridInterpolator<dim, PETScWrappers::MPI::Vector>
                          interp(solid_solver.scalar_dof_handler, x_q);

                        if (!interp.found_cell())
                          {
                            continue;
                          }

                        const auto &sig_vec = solid_solver.stress[i][j];
                        interp.point_value(sig_vec, sig_ij);

                        for (unsigned int a = 0; a < dofs_per_cell; ++a)
                          {
                            local_rhs[a] +=
                              sig_ij[0] * fev.shape_value(a, q) * fev.JxW(q);
                          }
                      }

                    scalar_projection_constraints.distribute_local_to_global(
                      local_rhs, ldofs, rhs[comp]);
                  }
              }
          }
      }

    for (auto &v : rhs)
      {
        v.compress(VectorOperation::add);
      }

    SolverControl ctl(rhs[0].size(), 1e-12);
    PETScWrappers::SparseDirectMUMPS direct(ctl, mpi_communicator);

    for (unsigned c = 0; c < s; ++c)
      {
        direct.solve(mass_matrix_scalar, projected_solid_stress[c], rhs[c]);
        scalar_projection_constraints.distribute(projected_solid_stress[c]);
        projected_solid_stress[c].compress(VectorOperation::insert);
      }
  }

  template <int dim>
  void FSI_stokes<dim>::compute_velocity_l2_projection()
  {
    PETScWrappers::MPI::Vector rhs_velocity(fluid_solver.owned_partitioning[0],
                                            mpi_communicator);
    rhs_velocity = 0.0;

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fluid_solver.fe.dofs_per_cell;
    const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();

    // Identify velocity DoFs
    std::vector<unsigned int> velocity_local_indices;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      if (fluid_solver.fe.system_to_component_index(i).first < dim)
        velocity_local_indices.push_back(i);
    const unsigned int n_velocity_dofs = velocity_local_indices.size();

    Vector<double> local_rhs(n_velocity_dofs);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> velocity_global_indices(
      n_velocity_dofs);

    build_projection_constraints();

    // Localize solid velocity for interpolation
    Vector<double> localized_solid_velocity(solid_solver.current_velocity);

    for (auto cell = fluid_solver.dof_handler.begin_active();
         cell != fluid_solver.dof_handler.end();
         ++cell)
      {

        if (!cell->is_locally_owned())
          continue;
        auto p = fluid_solver.cell_property.get_data(cell);
        if (p[0]->indicator == 0)
          continue;
        fe_values.reinit(cell);
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Point<dim> point = fe_values.quadrature_point(q);
            Vector<double> solid_vel(dim);
            Utils::GridInterpolator<dim, Vector<double>> interpolator(
              solid_solver.dof_handler, point);

            if (!interpolator.found_cell())
              {
                std::stringstream message;
                message << "Cannot find point in solid: " << point << std::endl;
                AssertThrow(interpolator.found_cell(),
                            ExcMessage(message.str()));
              }
            interpolator.point_value(localized_solid_velocity, solid_vel);

            for (unsigned int k = 0; k < n_velocity_dofs; ++k)
              {
                unsigned int i = velocity_local_indices[k];
                const unsigned int component =
                  fluid_solver.fe.system_to_component_index(i).first;

                local_rhs(k) +=
                  solid_vel[component] *
                  fe_values.shape_value_component(i, q, component) *
                  fe_values.JxW(q);
              }
          }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int k = 0; k < n_velocity_dofs; ++k)
          velocity_global_indices[k] =
            local_dof_indices[velocity_local_indices[k]];

        projection_constraints.distribute_local_to_global(
          local_rhs, velocity_global_indices, rhs_velocity);
      }
    rhs_velocity.compress(VectorOperation::add);

    // Solve
    PETScWrappers::MPI::Vector projected_velocity(
      fluid_solver.owned_partitioning[0], mpi_communicator);
    projected_velocity = 0.0;

    SolverControl solver_control(
      rhs_velocity.size(),
      rhs_velocity.l2_norm() > 1e-30 ? 1e-12 * rhs_velocity.l2_norm() : 1e-25);
    PETScWrappers::SparseDirectMUMPS direct_solver(solver_control,
                                                   mpi_communicator);

    direct_solver.solve(mass_matrix_velocity, projected_velocity, rhs_velocity);

    projection_constraints.distribute(projected_velocity);

    projected_solid_velocity.block(0) = projected_velocity;
    projected_solid_velocity.block(0).compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI_stokes<dim>::compute_acceleration_l2_projection()
  {
    PETScWrappers::MPI::Vector rhs_acc(fluid_solver.owned_partitioning[0],
                                       mpi_communicator);

    rhs_acc = 0.0;

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fluid_solver.fe.dofs_per_cell;
    const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();

    std::vector<unsigned int> vel_idx;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        if (fluid_solver.fe.system_to_component_index(i).first < dim)
          {
            vel_idx.push_back(i);
          }
      }
    const unsigned int n_vdofs = vel_idx.size();

    Vector<double> local_rhs(n_vdofs);
    std::vector<types::global_dof_index> ldof(dofs_per_cell), gdof(n_vdofs);

    build_projection_constraints();
    Vector<double> loc_acc(solid_solver.current_acceleration);

    for (auto cell : fluid_solver.dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() &&
            fluid_solver.cell_property.get_data(cell)[0]->indicator == 1)
          {
            fe_values.reinit(cell);
            local_rhs = 0;

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                Point<dim> x_q = fe_values.quadrature_point(q);
                Vector<double> a_s(dim);
                Utils::GridInterpolator<dim, Vector<double>> interp(
                  solid_solver.dof_handler, x_q);
                if (!interp.found_cell())
                  {
                    continue;
                  }

                interp.point_value(loc_acc, a_s);

                for (unsigned int k = 0; k < n_vdofs; ++k)
                  {
                    const unsigned int i = vel_idx[k];
                    const unsigned int c =
                      fluid_solver.fe.system_to_component_index(i).first;
                    local_rhs(k) += a_s[c] *
                                    fe_values.shape_value_component(i, q, c) *
                                    fe_values.JxW(q);
                  }
              }

            cell->get_dof_indices(ldof);
            for (unsigned int k = 0; k < n_vdofs; ++k)
              {
                gdof[k] = ldof[vel_idx[k]];
              }
            projection_constraints.distribute_local_to_global(
              local_rhs, gdof, rhs_acc);
          }
      }

    rhs_acc.compress(VectorOperation::add);

    PETScWrappers::MPI::Vector proj_acc(fluid_solver.owned_partitioning[0],
                                        mpi_communicator);

    const double rhs_norm = rhs_acc.l2_norm();

    SolverControl ctl(rhs_acc.size(),
                      rhs_norm > 1e-30 ? 1e-12 * rhs_norm : 1e-25);

    PETScWrappers::SparseDirectMUMPS solver(ctl, mpi_communicator);

    solver.solve(mass_matrix_velocity, proj_acc, rhs_acc);

    projection_constraints.distribute(proj_acc);

    projected_solid_acceleration.block(0) = proj_acc;
    projected_solid_acceleration.block(0).compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI_stokes<dim>::find_fluid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find fluid BC");
    move_solid_mesh(true);

    // get the L-2 projection
    build_arti_mass_matrix();
    compute_velocity_l2_projection();
    compute_acceleration_l2_projection();
    compute_stress_l2_projection();

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

    std::vector<PETScWrappers::MPI::Vector> tmp_fsi_stress;

    int stress_vec_size = dim + dim * (dim - 1) * 0.5;

    tmp_fsi_stress = std::vector<PETScWrappers::MPI::Vector>(
      stress_vec_size,
      PETScWrappers::MPI::Vector(fluid_solver.locally_owned_scalar_dofs,
                                 fluid_solver.locally_relevant_scalar_dofs,
                                 fluid_solver.mpi_communicator));

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

    // projected solid velocity and acc
    std::vector<Tensor<1, dim>> vs_projected(unit_points.size());
    std::vector<Tensor<1, dim>> as_projected(unit_points.size());

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

    // implementing the stress part for fsi force

    const std::vector<Point<dim>> &scalar_unit_points =
      fluid_solver.scalar_fe.get_unit_support_points();

    Quadrature<dim> scalar_dummy_q(scalar_unit_points);

    FEValues<dim> scalar_dummy_fe_values(
      fluid_solver.scalar_fe,
      scalar_dummy_q,
      update_values | update_quadrature_points | update_JxW_values |
        update_gradients);

    std::vector<types::global_dof_index> scalar_dof_indices(
      fluid_solver.scalar_fe.dofs_per_cell);

    std::vector<unsigned int> scalar_dof_touched(
      fluid_solver.scalar_dof_handler.n_dofs(), 0);

    std::vector<double> f_stress_component(scalar_unit_points.size());

    std::vector<std::vector<double>> f_cell_stress =
      std::vector<std::vector<double>>(
        fluid_solver.fsi_stress.size(),
        std::vector<double>(scalar_unit_points.size()));

    for (auto scalar_cell = fluid_solver.scalar_dof_handler.begin_active();
         scalar_cell != fluid_solver.scalar_dof_handler.end();
         ++scalar_cell)
      {

        if (!scalar_cell->is_locally_owned())
          continue;

        auto ptr = fluid_solver.cell_property.get_data(scalar_cell);
        if (ptr[0]->indicator == 0)
          continue;

        scalar_cell->get_dof_indices(scalar_dof_indices);

        scalar_dummy_fe_values.reinit(scalar_cell);

        int stress_index = 0;

        // get fluid stress at support points
        /*
        for (unsigned int i = 0; i < dim; i++)
          {
            for (unsigned int j = 0; j < i + 1; j++)
              {

                scalar_dummy_fe_values.get_function_values(
                  relevant_partition_stress[i][j], f_stress_component);

                f_cell_stress[stress_index] = f_stress_component;

                stress_index++;
              }
          }*/

        // get projected solid stress and the fluid stress

        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j <= i; ++j, ++stress_index)
              {
                scalar_dummy_fe_values.get_function_values(
                  relevant_partition_stress[i][j], f_stress_component);

                std::vector<double> sigma_proj(scalar_unit_points.size());

                scalar_dummy_fe_values.get_function_values(
                  projected_solid_stress[stress_index], sigma_proj);

                for (unsigned int k = 0; k < scalar_unit_points.size(); ++k)
                  {
                    f_cell_stress[stress_index][k] =
                      f_stress_component[k] - sigma_proj[k];
                  }
              }
          }

        for (unsigned int i = 0; i < scalar_unit_points.size(); ++i)
          {
            // Skip the already-set dofs.
            if (scalar_dof_touched[scalar_dof_indices[i]] != 0)
              continue;
            auto scalar_support_points =
              scalar_dummy_fe_values.get_quadrature_points();
            scalar_dof_touched[scalar_dof_indices[i]] = 1;
            if (!point_in_solid(solid_solver.scalar_dof_handler,
                                scalar_support_points[i]))
              continue;

            Utils::GridInterpolator<dim, Vector<double>> scalar_interpolator(
              solid_solver.scalar_dof_handler, scalar_support_points[i]);

            stress_index = 0;

            // store the fluid stress - projected solid stress into tmp fsi
            // stress

            for (unsigned int j = 0; j < dim; ++j)
              {
                for (unsigned int k = 0; k <= j; ++k, ++stress_index)
                  {
                    tmp_fsi_stress[stress_index][scalar_dof_indices[i]] =
                      f_cell_stress[stress_index][i];
                  }
              }

            // oldway for point-wise solid stress
            /*
            for (unsigned int j = 0; j < dim; j++)
              {
                for (unsigned int k = 0; k < j + 1; k++)
                  {
                    Vector<double> s_stress_component(1);

                    scalar_interpolator.point_value(localized_stress[j][k],
                                                    s_stress_component);

                    tmp_fsi_stress[stress_index][scalar_dof_indices[i]] =
                      f_cell_stress[stress_index][i] - s_stress_component[0];
                    stress_index++;
                  }
              }*/
          }
      }

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

            dummy_fe_values[velocities].get_function_values(
              projected_solid_velocity, vs_projected);
            dummy_fe_values[velocities].get_function_values(
              projected_solid_acceleration, as_projected);

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
                    vs[j] = solid_vel[j]; // pointwise-interpolation
                  }
                // Fluid total acceleration at support points
                Tensor<1, dim> fluid_acc =

                  (vs - v[i]) / time.get_delta_t(); // point-wise solid velocity
                //(vs_projected[i] - v[i]) / time.get_delta_t();  // L-2
                //projected solid velocity

                double theta = parameters.penalty_scale_factor;

                fluid_acc += theta *
                             //((vs - v[i]) / time.get_delta_t()); // point-wise
                             //solid velocity
                             (vs_projected[i] - v[i]) /
                             time.get_delta_t(); // L-2 projected solid velocity

                auto line = dof_indices[i];
                // Note that we are setting the value of the constraint to the
                // velocity delta!
                tmp_fsi_acceleration(line) =
                  // fluid_acc[index] - solid_acc[index];
                  fluid_acc[index] - as_projected[i][index];
              }
          }
      }
    tmp_fsi_acceleration.compress(VectorOperation::insert);
    fluid_solver.fsi_acceleration = tmp_fsi_acceleration;

    for (auto &stress_vector : tmp_fsi_stress)
      {
        stress_vector.compress(VectorOperation::insert);
      }

    for (unsigned int s = 0; s < fluid_solver.fsi_stress.size(); ++s)
      {
        fluid_solver.fsi_stress[s] = tmp_fsi_stress[s];
      }

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
        SA_model->update_moving_wall_distance(
          solid_boundary_vertices, solid_boundaries, shear_velocities);
      }

    move_solid_mesh(false);
  }

  template <int dim>
  void FSI_stokes<dim>::compute_penalty_energy()
  {

    double term2_local = 0.0;
    double penalty_term_2 = 0.0;
    double power_local = 0.0;
    double power_global = 0.0;

    double theta = parameters.penalty_scale_factor;

    const double alpha = (theta * parameters.solid_rho) / time.get_delta_t();

    double l2_local = 0.0;
    double solid_l2_global = 0.0;

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);
    const FEValuesExtractors::Vector velocities(0);

    Vector<double> localized_solid_velocity(solid_solver.current_velocity);

    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {

        if (!f_cell->is_locally_owned())
          continue;
        auto ptr = fluid_solver.cell_property.get_data(f_cell);
        if (ptr[0]->indicator == 0)
          continue;

        fe_values.reinit(f_cell);
        std::vector<Tensor<1, dim>> v_quad(fe_values.n_quadrature_points);
        fe_values[velocities].get_function_values(fluid_solver.present_solution,
                                                  v_quad);
        std::vector<Tensor<1, dim>> vs_quad(fe_values.n_quadrature_points);

        // use projected solid velocity
        fe_values[velocities].get_function_values(projected_solid_velocity,
                                                  vs_quad);

        // use point-wise interpolated velocity
        /*
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
          Point<dim> point = fe_values.quadrature_point(q);
          Utils::GridInterpolator<dim, Vector<double>> interpolator(
              solid_solver.dof_handler, point);
          Vector<double> solid_vel(dim);
          interpolator.point_value(localized_solid_velocity, solid_vel);
          for (unsigned int d = 0; d < dim; ++d) {
            vs_quad[q][d] = solid_vel[d];
          }
        }*/

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {

            double diff_norm_sq = 0.0;
            double dot = 0.0;
            double vs_norm_sq = 0.0;

            for (unsigned int d = 0; d < dim; ++d)
              {
                double diff = vs_quad[q][d] - v_quad[q][d];
                diff_norm_sq += diff * diff;
                dot += diff * v_quad[q][d];
                vs_norm_sq += vs_quad[q][d] * vs_quad[q][d];
              }

            term2_local += diff_norm_sq * fe_values.JxW(q);
            l2_local += vs_norm_sq * fe_values.JxW(q);
            power_local += dot * fe_values.JxW(q);
          }
      }

    MPI_Allreduce(
      &l2_local, &solid_l2_global, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    MPI_Allreduce(
      &term2_local, &penalty_term_2, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    MPI_Allreduce(
      &power_local, &power_global, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    // penalty_term_2 = (alpha/2)*penalty_term_2;

    power_global = alpha * power_global;

    const double solid_velocity_L2 = std::sqrt(solid_l2_global);

    const double velocity_diff_L2 = std::sqrt(penalty_term_2);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {

        std::ofstream file("penalty_energy.txt",
                           time.current() == 0 ? std::ios::out : std::ios::app);

        if (time.current() == 0)
          {
            file << "Time\tPenalty_power\tSolid_L2_Vel\tVel_diff_L2\n";
          }

        file << time.current() << "\t" << power_global << "\t"
             << solid_velocity_L2 << "\t" << velocity_diff_L2 << '\n';
        file.close();
      }
  }

  template <int dim>
  void FSI_stokes<dim>::compute_ke_rate()
  {

    double ke_rate_local = 0.0, ke_rate_global = 0.0;

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const FEValuesExtractors::Vector velocities(0);
    Vector<double> localized_solid_acc(solid_solver.current_acceleration);

    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        const auto property = fluid_solver.cell_property.get_data(f_cell);
        if (property[0]->indicator == 0)
          {
            continue;
          }

        fe_values.reinit(f_cell);

        const unsigned int n_q = fe_values.n_quadrature_points;
        std::vector<Tensor<1, dim>> u_f(n_q), a_s(n_q);
        fe_values[velocities].get_function_values(fluid_solver.present_solution,
                                                  u_f);

        fe_values[velocities].get_function_values(projected_solid_acceleration,
                                                  a_s);

        // Point-wise interpolation
        /*
        for (unsigned int q=0; q<n_q; ++q)
        {
          const Point<dim> x_q = fe_values.quadrature_point(q);
          Utils::GridInterpolator<dim,Vector<double>>
        interp(solid_solver.dof_handler, x_q); Vector<double> tmp(dim);
          interp.point_value(localized_solid_acc, tmp);
          for (unsigned int d=0; d<dim; ++d)
          {
            a_s[q][d] = tmp[d];
          }
        }*/

        for (unsigned int q = 0; q < n_q; ++q)
          {
            double dot = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
              {
                dot += a_s[q][d] * u_f[q][d];
              }
            ke_rate_local += dot * fe_values.JxW(q);
          }
      }

    MPI_Allreduce(&ke_rate_local,
                  &ke_rate_global,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  mpi_communicator);

    ke_rate_global *= parameters.solid_rho;

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::ofstream file("ke_rate.txt",
                           time.current() == 0 ? std::ios::out : std::ios::app);

        if (time.current() == 0)
          {
            file << "Time\tKE_rate\n";
          }

        file << time.current() << '\t' << ke_rate_global << '\n';

        file.close();
      }
  }

  template <int dim>
  void FSI_stokes<dim>::compute_stress_power()
  {

    double local_power = 0.0, global_power = 0.0;

    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEValues<dim> fe_values_s(
      fluid_solver.scalar_fe, fluid_solver.volume_quad_formula, update_values);

    const FEValuesExtractors::Vector velocities(0);

    std::vector<std::vector<Vector<double>>> localized_stress(
      dim, std::vector<Vector<double>>(dim));

    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = i; j < dim; ++j)
          {
            localized_stress[i][j] = solid_solver.stress[i][j];
          }
      }

    std::array<std::vector<double>,
               SymmetricTensor<2, dim>::n_independent_components>
      sigma_q;

    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)

      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }

        if (fluid_solver.cell_property.get_data(f_cell)[0]->indicator == 0)
          {
            continue;
          }

        fe_values.reinit(f_cell);

        typename DoFHandler<dim>::active_cell_iterator scalar_cell(
          &fluid_solver.triangulation,
          f_cell->level(),
          f_cell->index(),
          &fluid_solver.scalar_dof_handler);

        fe_values_s.reinit(scalar_cell);

        for (auto &vecs : sigma_q)
          {
            vecs.resize(fe_values_s.n_quadrature_points);
          }

        const unsigned int n_q = fe_values.n_quadrature_points;
        std::vector<Tensor<2, dim>> grad_u(n_q);
        fe_values[velocities].get_function_gradients(
          fluid_solver.present_solution, grad_u);

        // L-2 projected stress

        unsigned int comp = 0;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j <= i; ++j, ++comp)
              {
                fe_values_s.get_function_values(projected_solid_stress[comp],
                                                sigma_q[comp]);
              }
          }

        for (unsigned int q = 0; q < n_q; ++q)
          {
            SymmetricTensor<2, dim> sigma_s;

            comp = 0;
            for (unsigned int i = 0; i < dim; ++i)
              {
                for (unsigned int j = 0; j <= i; ++j, ++comp)
                  {
                    sigma_s[i][j] = sigma_q[comp][q];
                  }
              }

            // point-wise stress
            /*
            const Point<dim> x_q = fe_values.quadrature_point(q);

                Utils::GridInterpolator<dim,Vector<double>>
                   interp(solid_solver.scalar_dof_handler, x_q);

            for (unsigned int i=0;i<dim;++i)
            {
              for (unsigned int j=i;j<dim;++j)
              {
                Vector<double> tmp(1);
                interp.point_value(localized_stress[i][j], tmp);
                sigma_s[i][j] = tmp[0];
              }
            }*/

            double dot = 0.0;
            for (unsigned int i = 0; i < dim; ++i)
              {
                for (unsigned int j = 0; j < dim; ++j)
                  {
                    dot += sigma_s[i][j] * grad_u[q][i][j];
                  }
              }

            local_power += dot * fe_values.JxW(q);
          }
      }

    MPI_Allreduce(
      &local_power, &global_power, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::ofstream file("stress_power.txt",
                           time.current() == 0 ? std::ios::out : std::ios::app);

        if (time.current() == 0)
          {
            file << "Time\tStress_Power\n";
          }

        file << time.current() << '\t' << global_power << '\n';
        file.close();
      }
  }

  template <int dim>
  void FSI_stokes<dim>::compute_traction_power()
  {

    FEFaceValues<dim> fe_face_values(
      fluid_solver.fe,
      fluid_solver.face_quad_formula,
      update_values | update_gradients | update_quadrature_points |
        update_normal_vectors | update_JxW_values);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    const unsigned int n_q_points_face = fluid_solver.face_quad_formula.size();

    double local_P = 0.0, global_P = 0.0;

    std::vector<SymmetricTensor<2, dim>> sym_grad_v_face(n_q_points_face);
    std::vector<Tensor<1, dim>> vel_face(n_q_points_face);
    std::vector<double> pressure_values_face(n_q_points_face);

    for (auto cell = fluid_solver.dof_handler.begin_active();
         cell != fluid_solver.dof_handler.end();
         ++cell)

      {
        if (!cell->is_locally_owned())
          {
            continue;
          }

        if (fluid_solver.cell_property.get_data(cell)[0]->indicator == 0)
          {
            continue;
          }

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                continue; // for now ignore external boundary, change in the
                          // future
              }

            auto neigh = cell->neighbor(f);

            Assert(neigh.state() == IteratorState::valid,
                   ExcMessage("Invalid neighbour."));

            if (fluid_solver.cell_property.get_data(neigh)[0]->indicator == 1)
              {
                continue;
              }

            fe_face_values.reinit(cell, f);

            fe_face_values[velocities].get_function_values(
              fluid_solver.present_solution, vel_face);

            fe_face_values[pressure].get_function_values(
              fluid_solver.present_solution, pressure_values_face);

            fe_face_values[velocities].get_function_symmetric_gradients(
              fluid_solver.present_solution, sym_grad_v_face);

            for (unsigned int q = 0; q < n_q_points_face; ++q)
              {
                SymmetricTensor<2, dim> stress_tensor =
                  2 * parameters.viscosity * sym_grad_v_face[q];

                SymmetricTensor<2, dim> pressure_tensor =
                  -(pressure_values_face[q]) *
                  Physics::Elasticity::StandardTensors<dim>::I;

                stress_tensor += pressure_tensor;

                for (int i = 0; i < dim; i++)
                  {
                    for (int j = 0; j < dim; j++)
                      {
                        local_P += stress_tensor[i][j] * vel_face[q][i] *
                                   fe_face_values.normal_vector(q)[j] *
                                   fe_face_values.JxW(q);
                      }
                  }
              }
          }
      }

    MPI_Allreduce(
      &local_P, &global_P, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::ofstream file("traction_power.txt",
                           time.current() == 0 ? std::ios::out : std::ios::app);

        if (time.current() == 0)
          {
            file << "Time\tTraction_Power\n";
          }
        file << time.current() << '\t' << global_P << '\n';
        file.close();
      }
  }

  template <int dim>
  void FSI_stokes<dim>::find_solid_bc()
  {
    TimerOutput::Scope timer_section(timer, "Find solid BC");
    // Must use the updated solid coordinates
    move_solid_mesh(true);

    compute_fluid_traction_projection();

    for (unsigned int d = 0; d < dim; ++d)
      {
        solid_solver.fsi_traction_rows[d] = projected_fluid_traction[d];
      }

    // Fluid FEValues to do interpolation
    FEValues<dim> fe_values(
      fluid_solver.fe, fluid_solver.volume_quad_formula, update_values);
    // Solid FEValues for updating vertex normal vector
    FEFaceValues<dim> solid_fe_face_values(
      solid_solver.fe, solid_solver.face_quad_formula, update_normal_vectors);

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
            if (!s_cell->face(f)->at_boundary())
              {
                continue;
              }
            constexpr unsigned fixed_bc_flag{(1 << dim) - 1};
            auto bc = parameters.solid_dirichlet_bcs.find(
              s_cell->face(f)->boundary_id());
            if (bc != parameters.solid_dirichlet_bcs.end() &&
                bc->second == fixed_bc_flag)
              {
                // Skip those fixed faces
                continue;
              }
            // Start computing traction
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_face;
                 ++v)
              {
                auto line = s_cell->face(f)->vertex_dof_index(v, 0);
                auto scalar_line =
                  scalar_s_cell->face(f)->vertex_dof_index(v, 0);

                // Get interpolated solution from the fluid
                Vector<double> value(dim + 1);
                Utils::GridInterpolator<dim, PETScWrappers::MPI::BlockVector>
                  interpolator(fluid_solver.dof_handler,
                               s_cell->face(f)->vertex(v),
                               vertices_mask);
                interpolator.point_value(fluid_solver.present_solution, value);

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
                  -value[dim] * Physics::Elasticity::StandardTensors<dim>::I +
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
                // Update shear velocity for turbulence model wall
                // function
                if (fluid_solver.turbulence_model)
                  {
                    auto this_vertex = s_cell->face(f)->vertex_iterator(v);
                    // Find the normal of this vertex.
                    Tensor<1, dim> vertex_normal;
                    auto boundary_vertex_data =
                      solid_boundary_vertices.find(this_vertex);
                    AssertThrow(
                      boundary_vertex_data != solid_boundary_vertices.end(),
                      ExcMessage("Cannot find boundary vertex data!"));
                    auto &face_and_index = boundary_vertex_data->second;
                    for (auto [cell, face] : face_and_index.first)
                      {
                        solid_fe_face_values.reinit(cell, face);
                        vertex_normal += solid_fe_face_values.normal_vector(0);
                      }
                    vertex_normal /= face_and_index.first.size();
                    // Locate the image point
                    /* TODO: the image distance parameter is now unique to SA
                    model. Will need a separated turbulence model wall function
                    settings when we have more models.*/
                    double image_distance{
                      parameters.spalart_allmaras_image_distance};
                    Point<dim> image_point{this_vertex->vertex(0) +
                                           image_distance * vertex_normal};
                    // Interpolator for image point
                    Utils::GridInterpolator<dim,
                                            PETScWrappers::MPI::BlockVector>
                      image_point_interpolator(fluid_solver.dof_handler,
                                               s_cell->face(f)->vertex(v),
                                               vertices_mask);
                    int shear_velocity_index{face_and_index.second};
                    if (image_point_interpolator.get_cell().state() ==
                        IteratorState::IteratorStates::valid)
                      {
                        image_point_interpolator.point_value(
                          fluid_solver.present_solution, value);
                        // Compute tangential velocity
                        Tensor<1, dim> image_velocity;
                        for (unsigned d = 0; d < dim; ++d)
                          {
                            image_velocity[d] = value[d];
                          }
                        Tensor<1, dim> normal_velocity =
                          scalar_product(vertex_normal, image_velocity) *
                          vertex_normal;
                        double tangential_velocity =
                          (image_velocity - normal_velocity).norm();
                        // Use the shear velocity from last time step as
                        // the initial guess (last arg)
                        double shear_velocity =
                          fluid_solver.turbulence_model->get_shear_velocity(
                            tangential_velocity,
                            shear_velocities[shear_velocity_index]);
                        shear_velocities[shear_velocity_index] = shear_velocity;
                      }
                    else // Not belong to this process or out of bound
                      {
                        shear_velocities[shear_velocity_index] = 0.0;
                      }
                  }
              } // End looping support points
          }     // End looping cell faces
      }         // End looping solid cells
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
    if (fluid_solver.turbulence_model)
      {
        Utilities::MPI::sum(
          shear_velocities, mpi_communicator, shear_velocities);
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI_stokes<dim>::apply_contact_model(bool first_step)
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
  void FSI_stokes<dim>::collect_solid_boundary_vertices()
  {
    unsigned fixed_bc_flag = (1 << dim) - 1;

    int current_index{0};
    for (auto &cell : solid_solver.triangulation.active_cell_iterators())
      {
        if (!cell->at_boundary())
          {
            continue;
          }
        for (unsigned f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            auto face = cell->face(f);
            if (face->at_boundary())
              {
                // Check if the boundary is fixed
                auto bc =
                  parameters.solid_dirichlet_bcs.find(face->boundary_id());
                if (bc != parameters.solid_dirichlet_bcs.end() &&
                    bc->second == fixed_bc_flag)
                  {
                    // Skip those fixed vertices
                    continue;
                  }

                for (unsigned v = 0; v < GeometryInfo<dim>::vertices_per_face;
                     ++v)
                  {
                    // For shear velocity indexing
                    if (solid_boundary_vertices.find(face->vertex_iterator(
                          v)) == solid_boundary_vertices.end())
                      {
                        // Stores the index for the shear velocity on this
                        // vertex
                        solid_boundary_vertices[face->vertex_iterator(v)]
                          .second = current_index++;
                      }
                    solid_boundary_vertices[face->vertex_iterator(v)]
                      .first.push_back({cell, f});
                  }
              }
          }
      }
    // Initialize shear_velocities.
    shear_velocities.reinit(current_index);
    AssertThrow(
      shear_velocities.size() == solid_boundary_vertices.size(),
      ExcMessage("Size of solid vertices and shear velocities don't match!"));
  }

  template <int dim>
  void FSI_stokes<dim>::refine_mesh(const unsigned int min_grid_level,
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
  void FSI_stokes<dim>::run()
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

        Fluid::MPI::Stokes<dim> *stokes_solver =
          dynamic_cast<Fluid::MPI::Stokes<dim> *>(&fluid_solver);
        if (stokes_solver)
          {
            stokes_solver->initialize_bcs();

            projected_solid_velocity.reinit(fluid_solver.owned_partitioning,
                                            fluid_solver.relevant_partitioning,
                                            mpi_communicator);

            projected_solid_acceleration.reinit(
              fluid_solver.owned_partitioning,
              fluid_solver.relevant_partitioning,
              mpi_communicator);

            const unsigned int stress_vec_size = dim + dim * (dim - 1) / 2;
            projected_solid_stress.resize(stress_vec_size);
            for (auto &v : projected_solid_stress)
              {
                v.reinit(fluid_solver.locally_owned_scalar_dofs,
                         fluid_solver.locally_relevant_scalar_dofs,
                         mpi_communicator);
              }

            for (unsigned d = 0; d < dim; ++d)
              {
                projected_fluid_traction[d].reinit(
                  solid_solver.locally_owned_dofs, mpi_communicator);
              }

            compute_ke_rate();
            compute_stress_power();
            compute_traction_power();
            compute_penalty_energy();
          }
        else
          {
            fluid_solver.setup_dofs();
            fluid_solver.make_constraints();
            fluid_solver.initialize_system();
          }
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

        Fluid::MPI::Stokes<dim> *stokes_solver =
          dynamic_cast<Fluid::MPI::Stokes<dim> *>(&fluid_solver);
        if (stokes_solver)
          {
            stokes_solver->set_up_boundary_values();
          }
        else
          {
            fluid_solver.make_constraints();
          }
        // fluid_solver.make_constraints();

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
        compute_ke_rate();
        compute_stress_power();
        compute_traction_power();
        compute_penalty_energy();

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
  void FSI_stokes<dim>::set_penetration_criterion(
    const std::function<double(const Point<dim> &)> &criterion,
    Tensor<1, dim> direction)
  {
    penetration_criterion.reset(
      new std::function<double(const Point<dim> &)>(criterion));
    penetration_direction = direction;
  }

  template class FSI_stokes<2>;
  template class FSI_stokes<3>;
} // namespace MPI
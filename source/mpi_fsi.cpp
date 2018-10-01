#include "mpi_fsi.h"
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
                const Parameters::AllParameters &p)
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
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
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
  bool FSI<dim>::point_in_solid(const DoFHandler<dim> &df,
                                const Point<dim> &point)
  {
    // Check whether the point is in the solid box first.
    for (unsigned int i = 0; i < dim; ++i)
      {
        if (point(i) < solid_box(2 * i) || point(i) > solid_box(2 * i + 1))
          return false;
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
    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_quadrature_points);
    const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }
        // Only loop over local fluid cells.
        fe_values.reinit(f_cell);
        auto p = fluid_solver.cell_property.get_data(f_cell);
        bool is_solid = true;
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            Point<dim> point = f_cell->vertex(v);
            if (!point_in_solid(solid_solver.dof_handler, point))
              {
                is_solid = false;
                break;
              }
          }
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            p[q]->indicator = is_solid;
          }
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

    const unsigned int n_q_points = fluid_solver.volume_quad_formula.size();
    FEValues<dim> fe_values(fluid_solver.fe,
                            fluid_solver.volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    std::vector<SymmetricTensor<2, dim>> sym_grad_v(n_q_points);
    std::vector<double> p(n_q_points);
    std::vector<Tensor<2, dim>> grad_v(n_q_points);
    std::vector<Tensor<1, dim>> v(n_q_points);
    std::vector<Tensor<1, dim>> dv(n_q_points);

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
    Quadrature<dim> dummy_q(unit_points.size());
    MappingQGeneric<dim> mapping(1);
    FEValues<dim> dummy_fe_values(
      mapping, fluid_solver.fe, dummy_q, update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices(
      fluid_solver.fe.dofs_per_cell);

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
        // Check if the cell center is in the solid box.
        bool center_in_box = true;
        for (unsigned int i = 0; i < dim; ++i)
          {
            if ((f_cell->center())(i) < solid_box(2 * i) ||
                (f_cell->center())(i) > solid_box(2 * i + 1))
              center_in_box = false;
          }
        if (!center_in_box)
          {
            continue;
          }
        // Start working on the cell
        fe_values.reinit(f_cell);
        dummy_fe_values.reinit(f_cell);
        f_cell->get_dof_indices(dof_indices);
        auto support_points = dummy_fe_values.get_quadrature_points();
        // Fluid velocity increment
        fe_values[velocities].get_function_values(
          fluid_solver.solution_increment, dv);
        // Fluid velocity gradient
        fe_values[velocities].get_function_gradients(
          fluid_solver.present_solution, grad_v);
        // Fluid symmetric velocity gradient
        fe_values[velocities].get_function_symmetric_gradients(
          fluid_solver.present_solution, sym_grad_v);
        // Fluid pressure
        fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                p);
        // Loop over the support points to set Dirichlet BCs.
        for (unsigned int i = 0; i < unit_points.size(); ++i)
          {
            auto base_index = fluid_solver.fe.system_to_base_index(i);
            const unsigned int i_group = base_index.first.first;
            Assert(
              i_group < 2,
              ExcMessage("There should be only 2 groups of finite element!"));
            if (i_group == 1)
              continue; // skip the pressure dofs
            bool inside = true;
            for (unsigned int d = 0; d < dim; ++d)
              if (std::abs(unit_points[i][d]) < 1e-5)
                {
                  inside = false;
                  break;
                }
            if (inside)
              continue; // skip the in-cell support point
            // Same as fluid_solver.fe.system_to_base_index(i).first.second;
            const unsigned int index =
              fluid_solver.fe.system_to_component_index(i).first;
            Assert(index < dim,
                   ExcMessage("Vector component should be less than dim!"));
            if (!point_in_solid(solid_solver.dof_handler, support_points[i]))
              continue;
            Vector<double> fluid_velocity(dim);
            VectorTools::point_value(solid_solver.dof_handler,
                                     localized_solid_velocity,
                                     support_points[i],
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
        // Loop over all quadrature points to set FSI forces.
        // Now skip the ghost elements because it's not store in cell property.
        if (!f_cell->is_locally_owned())
          continue;
        auto ptr = fluid_solver.cell_property.get_data(f_cell);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Point<dim> point = fe_values.quadrature_point(q);
            ptr[q]->indicator = point_in_solid(solid_solver.dof_handler, point);
            ptr[q]->fsi_acceleration = 0;
            ptr[q]->fsi_stress = 0;
            if (ptr[q]->indicator == 0)
              continue;
            // acceleration: Dv^f/Dt - Dv^s/Dt
            Tensor<1, dim> fluid_acc =
              dv[q] / time.get_delta_t() + grad_v[q] * v[q];
            (void)fluid_acc;
            Vector<double> solid_acc(dim);
            VectorTools::point_value(solid_solver.dof_handler,
                                     localized_solid_acceleration,
                                     point,
                                     solid_acc);
            for (unsigned int i = 0; i < dim; ++i)
              {
                if (ptr[q]->indicator == 0)
                  ptr[q]->fsi_acceleration[i] =
                    (parameters.solid_rho - parameters.fluid_rho) *
                    (parameters.gravity[i] - solid_acc[i]);
              }
            // stress: sigma^f - sigma^s
            SymmetricTensor<2, dim> solid_sigma;
            for (unsigned int i = 0; i < dim; ++i)
              {
                for (unsigned int j = 0; j < dim; ++j)
                  {
                    Vector<double> sigma_ij(1);
                    VectorTools::point_value(solid_solver.scalar_dof_handler,
                                             localized_stress[i][j],
                                             point,
                                             sigma_ij);
                    solid_sigma[i][j] = sigma_ij[0];
                  }
              }
            if (ptr[q]->indicator == 0)
              ptr[q]->fsi_stress =
                -p[q] * Physics::Elasticity::StandardTensors<dim>::I +
                2 * parameters.viscosity * sym_grad_v[q] - solid_sigma;
          }
      }

    inner_nonzero.close();
    inner_zero.close();
    fluid_solver.nonzero_constraints.merge(
      inner_nonzero,
      AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
    fluid_solver.zero_constraints.merge(
      inner_zero,
      AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
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
    // Solid FEFaceValues to get the normal
    FEFaceValues<dim> fe_face_values(solid_solver.fe,
                                     solid_solver.face_quad_formula,
                                     update_quadrature_points |
                                       update_normal_vectors);

    const unsigned int n_face_q_points = solid_solver.face_quad_formula.size();

    for (auto s_cell = solid_solver.dof_handler.begin_active();
         s_cell != solid_solver.dof_handler.end();
         ++s_cell)
      {
        auto ptr = solid_solver.cell_property.get_data(s_cell);
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            // Current face is at boundary and without Dirichlet bc.
            if (s_cell->face(f)->at_boundary())
              {
                fe_face_values.reinit(s_cell, f);
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    Point<dim> q_point = fe_face_values.quadrature_point(q);
                    Tensor<1, dim> normal = fe_face_values.normal_vector(q);
                    Vector<double> value(dim + 1);
                    Utils::GridInterpolator<dim,
                                            PETScWrappers::MPI::BlockVector>
                      interpolator(fluid_solver.dof_handler, q_point);
                    interpolator.point_value(fluid_solver.present_solution,
                                             value);
                    std::vector<Tensor<1, dim>> gradient(dim + 1,
                                                         Tensor<1, dim>());
                    interpolator.point_gradient(fluid_solver.present_solution,
                                                gradient);
                    Vector<double> global_value(dim + 1);
                    std::vector<Tensor<1, dim>> global_gradient(
                      dim + 1, Tensor<1, dim>());
                    for (unsigned int i = 0; i < dim + 1; ++i)
                      {
                        global_value[i] =
                          Utilities::MPI::sum(value[i], mpi_communicator);
                        global_gradient[i] =
                          Utilities::MPI::sum(gradient[i], mpi_communicator);
                      }
                    // Compute stress
                    SymmetricTensor<2, dim> sym_deformation;
                    for (unsigned int i = 0; i < dim; ++i)
                      {
                        for (unsigned int j = 0; j < dim; ++j)
                          {
                            sym_deformation[i][j] =
                              (global_gradient[i][j] + global_gradient[j][i]) /
                              2;
                          }
                      }
                    // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                    SymmetricTensor<2, dim> stress =
                      -global_value[dim] *
                        Physics::Elasticity::StandardTensors<dim>::I +
                      2 * parameters.viscosity * sym_deformation;
                    ptr[f * n_face_q_points + q]->fsi_traction =
                      stress * normal;
                  }
              }
          }
      }
    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    solid_solver.triangulation.refine_global(parameters.global_refinements[1]);
    solid_solver.setup_dofs();
    solid_solver.initialize_system();
    fluid_solver.triangulation.refine_global(parameters.global_refinements[0]);
    fluid_solver.setup_dofs();
    fluid_solver.make_constraints();
    fluid_solver.initialize_system();
    pcout << "Number of fluid active cells and dofs: ["
          << fluid_solver.triangulation.n_active_cells() << ", "
          << fluid_solver.dof_handler.n_dofs() << "]" << std::endl
          << "Number of solid active cells and dofs: ["
          << solid_solver.triangulation.n_active_cells() << ", "
          << solid_solver.dof_handler.n_dofs() << "]" << std::endl;
    bool first_step = true;
    while (time.end() - time.current() > 1e-12)
      {
        find_solid_bc();
        solid_solver.run_one_step(first_step);
        update_solid_box();
        // update_indicator();
        fluid_solver.make_constraints();
        if (!first_step)
          {
            fluid_solver.nonzero_constraints.clear();
            fluid_solver.nonzero_constraints.copy_from(
              fluid_solver.zero_constraints);
          }
        find_fluid_bc();
        fluid_solver.run_one_step(true);
        first_step = false;
        time.increment();
      }
  }

  template class FSI<2>;
  template class FSI<3>;
} // namespace MPI

#include "mpi_fsi.h"
#include <iostream>

namespace MPI
{
  template <int dim>
  FSI<dim>::FSI(Fluid::MPI::FluidSolver<dim> &f,
                Solid::MPI::SharedSolidSolver<dim> &s,
                const Parameters::AllParameters &p)
    : fluid_solver(f),
      solid_solver(s),
      parameters(p),
      time(parameters.end_time,
           parameters.time_step,
           parameters.output_interval,
           parameters.refinement_interval,
           parameters.save_interval)
  {
    std::cout << "  Number of fluid active cells: "
              << fluid_solver.triangulation.n_global_active_cells() << std::endl
              << "  Number of solid active cells: "
              << solid_solver.triangulation.n_active_cells() << std::endl;
  }

  template <int dim>
  void FSI<dim>::move_solid_mesh(bool move_forward)
  {
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
  bool FSI<dim>::point_in_mesh(const DoFHandler<dim> &df,
                               const Point<dim> &point)
  {
    for (auto cell = df.begin_active(); cell != df.end(); ++cell)
      {
        if (cell->point_inside(point))
          {
            return true;
          }
      }
    return false;
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
    move_solid_mesh(true);
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
        auto p = fluid_solver.cell_property.get_data(f_cell);
        bool is_solid = true;
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            Point<dim> point = f_cell->vertex(v);
            if (!point_in_mesh(solid_solver.dof_handler, point))
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
    move_solid_mesh(true);

    ConstraintMatrix inner_nonzero, inner_zero;
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

    Vector<double> localized_solid_velocity(solid_solver.current_velocity);
    Vector<double> localized_solid_acceleration(
      solid_solver.current_acceleration);

    for (auto f_cell = fluid_solver.dof_handler.begin_active();
         f_cell != fluid_solver.dof_handler.end();
         ++f_cell)
      {
        if (!f_cell->is_locally_owned())
          {
            continue;
          }
        auto ptr = fluid_solver.cell_property.get_data(f_cell);
        if (ptr[0]->indicator != 1)
          {
            continue;
          }
        fe_values.reinit(f_cell);
        // Fluid symmetric velocity gradient
        fe_values[velocities].get_function_symmetric_gradients(
          fluid_solver.present_solution, sym_grad_v);
        // Fluid pressure
        fe_values[pressure].get_function_values(fluid_solver.present_solution,
                                                p);
        // Loop over the vertices to set Dirichlet BCs.
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            auto point = f_cell->vertex(v);
            Vector<double> fluid_velocity(dim);
            VectorTools::point_value(solid_solver.dof_handler,
                                     localized_solid_velocity,
                                     point,
                                     fluid_velocity);
            for (unsigned int i = 0; i < dim; ++i)
              {
                auto line = f_cell->vertex_dof_index(v, i);
                inner_nonzero.add_line(line);
                inner_zero.add_line(line);
                // Note that we are setting the value of the constraint to the
                // velocity delta!
                inner_nonzero.set_inhomogeneity(
                  line,
                  fluid_velocity[i] - fluid_solver.present_solution(line));
              }
          }
        // Loop over all quadrature points to set FSI forces.
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Point<dim> point = fe_values.quadrature_point(q);
            // acceleration
            Vector<double> fluid_acc(dim);
            VectorTools::point_value(solid_solver.dof_handler,
                                     localized_solid_acceleration,
                                     point,
                                     fluid_acc);
            for (unsigned int i = 0; i < dim; ++i)
              {
                ptr[q]->fsi_acceleration[i] = fluid_acc[i];
              }
            // stress
            ptr[q]->fsi_stress =
              -p[q] * Physics::Elasticity::StandardTensors<dim>::I +
              parameters.viscosity * sym_grad_v[q];
          }
      }

    inner_nonzero.close();
    inner_zero.close();
    fluid_solver.nonzero_constraints.merge(
      inner_nonzero, ConstraintMatrix::MergeConflictBehavior::left_object_wins);
    fluid_solver.zero_constraints.merge(
      inner_zero, ConstraintMatrix::MergeConflictBehavior::left_object_wins);

    move_solid_mesh(false);
  }

  template <int dim>
  void FSI<dim>::find_solid_bc()
  {
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
            if (s_cell->face(f)->at_boundary() &&
                parameters.solid_dirichlet_bcs.find(
                  s_cell->face(f)->boundary_id()) ==
                  parameters.solid_dirichlet_bcs.end())
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
                    // Communication
                    for (unsigned int i = 0; i < dim + 1; ++i)
                      {
                        Utilities::MPI::sum(value[i],
                                            fluid_solver.mpi_communicator);
                        Utilities::MPI::sum(gradient[i],
                                            fluid_solver.mpi_communicator);
                      }

                    // Compute stress
                    SymmetricTensor<2, dim> sym_deformation;
                    for (unsigned int i = 0; i < dim; ++i)
                      {
                        for (unsigned int j = 0; j < dim; ++j)
                          {
                            sym_deformation[i][j] =
                              (gradient[i][j] + gradient[j][i]) / 2;
                          }
                      }
                    // \f$ \sigma = -p\bold{I} + \mu\nabla^S v\f$
                    SymmetricTensor<2, dim> stress =
                      -value[dim] *
                        Physics::Elasticity::StandardTensors<dim>::I +
                      parameters.viscosity * sym_deformation;
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
    solid_solver.triangulation.refine_global(parameters.global_refinement);
    solid_solver.setup_dofs();
    solid_solver.initialize_system();
    fluid_solver.triangulation.refine_global(parameters.global_refinement);
    fluid_solver.setup_dofs();
    fluid_solver.make_constraints();
    fluid_solver.initialize_system();
    bool first_step = true;
    while (time.end() - time.current() > 1e-12)
      {
        find_solid_bc();
        solid_solver.run_one_step(first_step);
        update_indicator();
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
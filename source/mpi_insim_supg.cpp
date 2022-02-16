#include "mpi_insim_supg.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SUPGInsIM<dim>::SUPGInsIM(parallel::distributed::Triangulation<dim> &tria,
                              const Parameters::AllParameters &parameters)
      : SUPGFluidSolver<dim>(tria, parameters)
    {
    }

    template <int dim>
    void SUPGInsIM<dim>::assemble(const bool use_nonzero_constraints)
    {
      TimerOutput::Scope timer_section(timer, "Assemble system");

      Tensor<1, dim> gravity;
      for (unsigned int i = 0; i < dim; ++i)
        gravity[i] = parameters.gravity[i];

      system_matrix = 0;
      Abs_A_matrix = 0;
      schur_matrix = 0;
      B2pp_matrix = 0;

      system_rhs = 0;

      FEValues<dim> fe_values(fe,
                              volume_quad_formula,
                              update_values | update_quadrature_points |
                                update_JxW_values | update_gradients);
      FEFaceValues<dim> fe_face_values(fe,
                                       face_quad_formula,
                                       update_values | update_normal_vectors |
                                         update_quadrature_points |
                                         update_JxW_values);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int u_dofs = fe.base_element(0).dofs_per_cell;
      const unsigned int p_dofs = fe.base_element(1).dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();
      const unsigned int n_face_q_points = face_quad_formula.size();

      AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
                  ExcMessage("Wrong partitioning of dofs!"));

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dim);

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double> local_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      // For the linearized system, we create temporary storage for current
      // velocity
      // and gradient, current pressure, and present velocity. In practice, they
      // are
      // all obtained through their shape functions at quadrature points.

      std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
      std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
      std::vector<double> current_pressure_values(n_q_points);
      std::vector<Tensor<1, dim>> current_pressure_gradients(n_q_points);
      std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
      std::vector<Tensor<1, dim>> artificial_bf(n_q_points);

      std::vector<double> div_phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
      std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
      std::vector<double> phi_p(dofs_per_cell);
      std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);

              local_matrix = 0;
              local_rhs = 0;

              fe_values[velocities].get_function_values(
                evaluation_point, current_velocity_values);

              fe_values[velocities].get_function_gradients(
                evaluation_point, current_velocity_gradients);

              fe_values[pressure].get_function_values(evaluation_point,
                                                      current_pressure_values);

              fe_values[pressure].get_function_gradients(
                evaluation_point, current_pressure_gradients);

              fe_values[velocities].get_function_values(
                present_solution, present_velocity_values);

              if (body_force)
                {
                  body_force->tensor_value_list(
                    fe_values.get_quadrature_points(), artificial_bf);
                }

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const double rho = parameters.fluid_rho;
                  const double viscosity = parameters.viscosity;

                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      div_phi_u[k] = fe_values[velocities].divergence(k, q);
                      grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                      phi_u[k] = fe_values[velocities].value(k, q);
                      phi_p[k] = fe_values[pressure].value(k, q);
                      grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                    }

                  // Define the UGN based SUPG parameters (Tezduyar):
                  // tau_SUPG and tau_PSPG. They are
                  // evaluated based on the results from the last Newton
                  // iteration.
                  double tau_SUPG, tau_PSPG, tau_LSIC;
                  // the length scale h is the length of the element in the
                  // direction
                  // of convection
                  double h = 0;
                  for (unsigned int a = 0;
                       a < dofs_per_cell / fe.dofs_per_vertex;
                       ++a)
                    {
                      h += abs(present_velocity_values[q] *
                               fe_values.shape_grad(a, q));
                    }
                  if (h)
                    h = 2 * present_velocity_values[q].norm() / h;
                  else
                    h = 0;
                  double nu = viscosity / rho;
                  double v_norm = present_velocity_values[q].norm();
                  if (h)
                    tau_SUPG = 1 / sqrt((pow(2 / time.get_delta_t(), 2) +
                                         pow(2 * v_norm / h, 2) +
                                         pow(4 * nu / pow(h, 2), 2)));
                  else
                    tau_SUPG = time.get_delta_t() / 2;
                  tau_PSPG = tau_SUPG / rho;
                  double localRe = v_norm * h / (2 * nu);
                  double z = localRe <= 3 ? (localRe / 3) : 1;
                  tau_LSIC = h / 2 * v_norm * z;

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      double current_velocity_divergence =
                        trace(current_velocity_gradients[q]);
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          // Let the linearized diffusion, continuity
                          // terms be written as
                          // the bilinear operator: \f$A = a((\delta{u},
                          // \delta{p}), (\delta{v}, \delta{q}))\f$,
                          // the linearized convection term be: \f$C =
                          // c(u;\delta{u}, \delta{v})\f$,
                          // and the linearized inertial term be:
                          // \f$M = m(\delta{u}, \delta{v})$, then LHS is: $(A
                          // +
                          // C) + M/{\Delta{t}}\f$
                          local_matrix(i, j) +=
                            ((viscosity *
                                scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                              rho * current_velocity_gradients[q] * phi_u[j] *
                                phi_u[i] +
                              rho * grad_phi_u[j] * current_velocity_values[q] *
                                phi_u[i] -
                              div_phi_u[i] * phi_p[j]) +
                             rho * phi_u[i] * phi_u[j] / time.get_delta_t()) *
                            fe_values.JxW(q);
                          // Add SUPG and PSPG stabilization
                          local_matrix(i, j) +=
                            // SUPG Convection
                            (tau_SUPG * rho *
                               (current_velocity_values[q] * grad_phi_u[i]) *
                               (phi_u[j] * current_velocity_gradients[q]) +
                             tau_SUPG * rho *
                               (current_velocity_values[q] * grad_phi_u[i]) *
                               (current_velocity_values[q] * grad_phi_u[j]) +
                             tau_SUPG * rho * (phi_u[j] * grad_phi_u[i]) *
                               (current_velocity_values[q] *
                                current_velocity_gradients[q]) +
                             // SUPG Acceleration
                             tau_SUPG * rho * current_velocity_values[q] *
                               grad_phi_u[i] * phi_u[j] / time.get_delta_t() +
                             tau_SUPG * rho * phi_u[j] * grad_phi_u[i] *
                               (current_velocity_values[q] -
                                present_velocity_values[q]) /
                               time.get_delta_t() +
                             // SUPG Pressure
                             tau_SUPG * current_velocity_values[q] *
                               grad_phi_u[i] * grad_phi_p[j] +
                             tau_SUPG * phi_u[j] * grad_phi_u[i] *
                               current_pressure_gradients[q] -
                             // SUPG body force
                             tau_SUPG * phi_u[j] * grad_phi_u[i] * rho *
                               (gravity + artificial_bf[q]) +
                             // PSPG Convection
                             tau_PSPG * rho * grad_phi_p[i] *
                               (phi_u[j] * current_velocity_gradients[q]) +
                             tau_PSPG * rho * grad_phi_p[i] *
                               (current_velocity_values[q] * grad_phi_u[j]) +
                             // PSPG Acceleration
                             tau_PSPG * rho * grad_phi_p[i] * phi_u[j] /
                               time.get_delta_t() +
                             // PSPG Pressure
                             tau_PSPG * grad_phi_p[i] * grad_phi_p[j] +
                             // LSIC velocity divergence
                             tau_LSIC * rho * div_phi_u[i] * div_phi_u[j]) *
                            fe_values.JxW(q);
                          // For more clear demonstration, write continuity
                          // equation
                          // separately.
                          // The original strong form is:
                          // \f$p_{,t} + \frac{C_p}{C_v} * (p_0 + p) * (\nabla
                          // \times u) + u (\nabla p) = 0\f$
                          local_matrix(i, j) +=
                            div_phi_u[j] * phi_p[i] * fe_values.JxW(q);
                        }

                      // RHS is \f$-(A_{current} + C_{current}) -
                      // M_{present-current}/\Delta{t}\f$.
                      local_rhs(i) +=
                        ((-viscosity *
                            scalar_product(current_velocity_gradients[q],
                                           grad_phi_u[i]) -
                          rho * current_velocity_gradients[q] *
                            current_velocity_values[q] * phi_u[i] +
                          current_pressure_values[q] * div_phi_u[i]) -
                         rho *
                           (current_velocity_values[q] -
                            present_velocity_values[q]) *
                           phi_u[i] / time.get_delta_t() +
                         (gravity + artificial_bf[q]) * phi_u[i] * rho) *
                        fe_values.JxW(q);
                      local_rhs(i) +=
                        -(current_velocity_divergence * phi_p[i]) *
                        fe_values.JxW(q);
                      // Add SUPG and PSPG rhs terms.
                      local_rhs(i) +=
                        -((tau_SUPG * current_velocity_values[q] *
                           grad_phi_u[i]) *
                            (rho * ((current_velocity_values[q] -
                                     present_velocity_values[q]) /
                                      time.get_delta_t() +
                                    current_velocity_values[q] *
                                      current_velocity_gradients[q]) +
                             current_pressure_gradients[q] -
                             rho * (gravity + artificial_bf[q])) +
                          (tau_PSPG * grad_phi_p[i]) *
                            (rho * ((current_velocity_values[q] -
                                     present_velocity_values[q]) /
                                      time.get_delta_t() +
                                    current_velocity_values[q] *
                                      current_velocity_gradients[q]) +
                             current_pressure_gradients[q] -
                             rho * (gravity + artificial_bf[q]))) *
                        fe_values.JxW(q);
                      // Add LSIC rhs terms.
                      local_rhs(i) += -(tau_LSIC * rho * div_phi_u[i]) *
                                      current_velocity_divergence *
                                      fe_values.JxW(q);
                    }
                }

              // Impose pressure boundary here if specified, loop over faces on
              // the
              // cell
              // and apply pressure boundary conditions:
              // \f$\int_{\Gamma_n} -p\bold{n}d\Gamma\f$
              if (parameters.n_fluid_neumann_bcs != 0)
                {
                  for (unsigned int face_n = 0;
                       face_n < GeometryInfo<dim>::faces_per_cell;
                       ++face_n)
                    {
                      if (cell->at_boundary(face_n) &&
                          parameters.fluid_neumann_bcs.find(
                            cell->face(face_n)->boundary_id()) !=
                            parameters.fluid_neumann_bcs.end())
                        {
                          fe_face_values.reinit(cell, face_n);
                          unsigned int p_bc_id =
                            cell->face(face_n)->boundary_id();
                          double boundary_values_p =
                            parameters.fluid_neumann_bcs[p_bc_id];
                          for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                  local_rhs(i) += -(
                                    fe_face_values[velocities].value(i, q) *
                                    fe_face_values.normal_vector(q) *
                                    boundary_values_p * fe_face_values.JxW(q));
                                }
                            }
                        }
                    }
                }

              cell->get_dof_indices(local_dof_indices);

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

    template class SUPGInsIM<2>;
    template class SUPGInsIM<3>;
  } // namespace MPI
} // namespace Fluid

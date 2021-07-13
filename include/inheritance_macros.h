#ifndef INHERETANCE_MACROS
#define INHERETANCE_MACROS

#define MPIFluidSolverInheritanceMacro()                                       \
public:                                                                        \
  using FluidSolver<dim>::add_hard_coded_boundary_condition;                   \
  using FluidSolver<dim>::attach_turbulence_model;                             \
  using FluidSolver<dim>::set_body_force;                                      \
  using FluidSolver<dim>::set_sigma_pml_field;                                 \
  using FluidSolver<dim>::set_initial_condition;                               \
                                                                               \
private:                                                                       \
  using FluidSolver<dim>::setup_dofs;                                          \
  using FluidSolver<dim>::make_constraints;                                    \
  using FluidSolver<dim>::setup_cell_property;                                 \
  using FluidSolver<dim>::apply_initial_condition;                             \
  using FluidSolver<dim>::refine_mesh;                                         \
  using FluidSolver<dim>::output_results;                                      \
  using FluidSolver<dim>::save_checkpoint;                                     \
  using FluidSolver<dim>::load_checkpoint;                                     \
  using FluidSolver<dim>::update_stress;                                       \
                                                                               \
  using FluidSolver<dim>::dofs_per_block;                                      \
  using FluidSolver<dim>::triangulation;                                       \
  using FluidSolver<dim>::fe;                                                  \
  using FluidSolver<dim>::scalar_fe;                                           \
  using FluidSolver<dim>::dof_handler;                                         \
  using FluidSolver<dim>::scalar_dof_handler;                                  \
  using FluidSolver<dim>::volume_quad_formula;                                 \
  using FluidSolver<dim>::face_quad_formula;                                   \
  using FluidSolver<dim>::zero_constraints;                                    \
  using FluidSolver<dim>::nonzero_constraints;                                 \
  using FluidSolver<dim>::sparsity_pattern;                                    \
  using FluidSolver<dim>::system_matrix;                                       \
  using FluidSolver<dim>::mass_matrix;                                         \
  using FluidSolver<dim>::mass_schur;                                          \
  using FluidSolver<dim>::present_solution;                                    \
  using FluidSolver<dim>::solution_increment;                                  \
  using FluidSolver<dim>::system_rhs;                                          \
  using FluidSolver<dim>::fsi_acceleration;                                    \
  using FluidSolver<dim>::stress;                                              \
  using FluidSolver<dim>::parameters;                                          \
  using FluidSolver<dim>::mpi_communicator;                                    \
  using FluidSolver<dim>::pcout;                                               \
  using FluidSolver<dim>::owned_partitioning;                                  \
  using FluidSolver<dim>::relevant_partitioning;                               \
  using FluidSolver<dim>::locally_owned_scalar_dofs;                           \
  using FluidSolver<dim>::locally_relevant_dofs;                               \
  using FluidSolver<dim>::locally_relevant_scalar_dofs;                        \
  using FluidSolver<dim>::times_and_names;                                     \
  using FluidSolver<dim>::time;                                                \
  using FluidSolver<dim>::timer;                                               \
  using FluidSolver<dim>::timer2;                                              \
  using FluidSolver<dim>::cell_property;                                       \
  using FluidSolver<dim>::turbulence_model;                                    \
  using FluidSolver<dim>::hard_coded_boundary_values;                          \
  using FluidSolver<dim>::body_force;                                          \
  using FluidSolver<dim>::sigma_pml_field;                                     \
  using FluidSolver<dim>::initial_condition_field

#define MPISharedSolidSolverInheritanceMacro()                                 \
private:                                                                       \
  using SharedSolidSolver<dim>::triangulation;                                 \
  using SharedSolidSolver<dim>::parameters;                                    \
  using SharedSolidSolver<dim>::dof_handler;                                   \
  using SharedSolidSolver<dim>::scalar_dof_handler;                            \
  using SharedSolidSolver<dim>::fe;                                            \
  using SharedSolidSolver<dim>::scalar_fe;                                     \
  using SharedSolidSolver<dim>::volume_quad_formula;                           \
  using SharedSolidSolver<dim>::face_quad_formula;                             \
  using SharedSolidSolver<dim>::constraints;                                   \
  using SharedSolidSolver<dim>::system_matrix;                                 \
  using SharedSolidSolver<dim>::mass_matrix;                                   \
  using SharedSolidSolver<dim>::stiffness_matrix;                              \
  using SharedSolidSolver<dim>::damping_matrix;                                \
  using SharedSolidSolver<dim>::system_rhs;                                    \
  using SharedSolidSolver<dim>::current_acceleration;                          \
  using SharedSolidSolver<dim>::current_velocity;                              \
  using SharedSolidSolver<dim>::current_displacement;                          \
  using SharedSolidSolver<dim>::previous_acceleration;                         \
  using SharedSolidSolver<dim>::previous_velocity;                             \
  using SharedSolidSolver<dim>::previous_displacement;                         \
  using SharedSolidSolver<dim>::fsi_stress_rows;                               \
  using SharedSolidSolver<dim>::fluid_velocity;                                \
  using SharedSolidSolver<dim>::fluid_pressure;                                \
  using SharedSolidSolver<dim>::strain;                                        \
  using SharedSolidSolver<dim>::stress;                                        \
  using SharedSolidSolver<dim>::mpi_communicator;                              \
  using SharedSolidSolver<dim>::n_mpi_processes;                               \
  using SharedSolidSolver<dim>::this_mpi_process;                              \
  using SharedSolidSolver<dim>::pcout;                                         \
  using SharedSolidSolver<dim>::time;                                          \
  using SharedSolidSolver<dim>::timer;                                         \
  using SharedSolidSolver<dim>::locally_owned_dofs;                            \
  using SharedSolidSolver<dim>::locally_owned_scalar_dofs;                     \
  using SharedSolidSolver<dim>::locally_relevant_dofs;                         \
  using SharedSolidSolver<dim>::times_and_names

#define MPITurbulenceModelInheritanceMacro()                                   \
public:                                                                        \
  using TurbulenceModel<dim>::reinit;                                          \
                                                                               \
private:                                                                       \
  using TurbulenceModel<dim>::get_eddy_viscosity;                              \
  using TurbulenceModel<dim>::connect_indicator_field;                         \
                                                                               \
  using TurbulenceModel<dim>::triangulation;                                   \
  using TurbulenceModel<dim>::dof_handler;                                     \
  using TurbulenceModel<dim>::scalar_dof_handler;                              \
  using TurbulenceModel<dim>::fe;                                              \
  using TurbulenceModel<dim>::scalar_fe;                                       \
  using TurbulenceModel<dim>::volume_quad_formula;                             \
  using TurbulenceModel<dim>::face_quad_formula;                               \
  using TurbulenceModel<dim>::zero_constraints;                                \
  using TurbulenceModel<dim>::nonzero_constraints;                             \
  using TurbulenceModel<dim>::indicator_function;                              \
  using TurbulenceModel<dim>::sparsity_pattern;                                \
  using TurbulenceModel<dim>::system_matrix;                                   \
  using TurbulenceModel<dim>::system_rhs;                                      \
  using TurbulenceModel<dim>::fluid_present_solution;                          \
  using TurbulenceModel<dim>::eddy_viscosity;                                  \
  using TurbulenceModel<dim>::parameters;                                      \
  using TurbulenceModel<dim>::mpi_communicator;                                \
  using TurbulenceModel<dim>::pcout;                                           \
  using TurbulenceModel<dim>::owned_partitioning;                              \
  using TurbulenceModel<dim>::relevant_partitioning;                           \
  using TurbulenceModel<dim>::locally_owned_scalar_dofs;                       \
  using TurbulenceModel<dim>::locally_relevant_scalar_dofs;                    \
  using TurbulenceModel<dim>::time;                                            \
  using TurbulenceModel<dim>::timer

#endif

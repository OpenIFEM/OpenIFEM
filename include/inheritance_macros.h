#ifndef INHERETANCE_MACROS
#define INHERETANCE_MACROS

#define MPIFluidSolverInheritanceMacro()                                       \
public:                                                                        \
  using FluidSolver<dim>::add_hard_coded_boundary_condition;                   \
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
  using FluidSolver<dim>::hard_coded_boundary_values;                          \
  using FluidSolver<dim>::initial_condition_field

#endif

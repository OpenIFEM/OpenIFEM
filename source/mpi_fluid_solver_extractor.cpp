#include "mpi_fluid_solver_extractor.h"

namespace Fluid
{
  namespace MPI
  {
    template <int dim>
    SmartPointer<const parallel::distributed::Triangulation<dim>>
    FluidSolverExtractor<dim>::get_triangulation(
      const FluidSolver<dim> &fluid_solver)
    {
      return SmartPointer<const parallel::distributed::Triangulation<dim>>(
        &fluid_solver.triangulation);
    }

    template <int dim>
    std::pair<SmartPointer<const FESystem<dim>>,
              SmartPointer<const DoFHandler<dim>>>
    FluidSolverExtractor<dim>::get_dof_handler(
      const FluidSolver<dim> &fluid_solver)
    {
      SmartPointer<const FESystem<dim>> ptr1(&(fluid_solver.fe));
      SmartPointer<const DoFHandler<dim>> ptr2(&(fluid_solver.dof_handler));
      return {ptr1, ptr2};
    }

    template <int dim>
    std::pair<SmartPointer<const FE_Q<dim>>,
              SmartPointer<const DoFHandler<dim>>>
    FluidSolverExtractor<dim>::get_scalar_dof_handler(
      const FluidSolver<dim> &fluid_solver)
    {
      SmartPointer<const FE_Q<dim>> ptr1(&(fluid_solver.scalar_fe));
      SmartPointer<const DoFHandler<dim>> ptr2(
        &(fluid_solver.scalar_dof_handler));
      return {ptr1, ptr2};
    }

    template <int dim>
    const Parameters::AllParameters *FluidSolverExtractor<dim>::get_parameters(
      const FluidSolver<dim> &fluid_solver)
    {
      return &(fluid_solver.parameters);
    }

    template <int dim>
    std::tuple<const std::vector<IndexSet> *,
               const std::vector<IndexSet> *,
               const IndexSet *,
               const IndexSet *>
    FluidSolverExtractor<dim>::get_partitions(
      const FluidSolver<dim> &fluid_solver)
    {
      return std::make_tuple(&fluid_solver.owned_partitioning,
                             &fluid_solver.relevant_partitioning,
                             &fluid_solver.locally_owned_scalar_dofs,
                             &fluid_solver.locally_relevant_scalar_dofs);
    }

    template <int dim>
    const PETScWrappers::MPI::BlockVector *
    FluidSolverExtractor<dim>::get_solution(
      const FluidSolver<dim> &fluid_solver)
    {
      return &fluid_solver.present_solution;
    }

    template <int dim>
    const Utils::Time *
    FluidSolverExtractor<dim>::get_time(const FluidSolver<dim> &fluid_solver)
    {
      return &fluid_solver.time;
    }

    template class FluidSolverExtractor<2>;
    template class FluidSolverExtractor<3>;
  } // namespace MPI
} // namespace Fluid
#include "mpi_turbulence_model.h"
#include "mpi_spalart_allmaras.h"

namespace Fluid
{
  namespace MPI
  {
    //! class
    template <int dim>
    class SpalartAllmaras;

    template <int dim>
    TurbulenceModel<dim>::TurbulenceModel(const FluidSolver<dim> &fluid_solver)
      : mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout,
              Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
        timer(
          mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
    {
      reinit(fluid_solver);
    }

    template <int dim>
    TurbulenceModel<dim> *
    TurbulenceModel<dim>::create(const FluidSolver<dim> &fluid_solver,
                                 const std::string &model_name)
    {
      if (model_name == "Spalart-Allmaras")
        {
          return new SpalartAllmaras<dim>(fluid_solver);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
      return nullptr;
    }

    template <int dim>
    void TurbulenceModel<dim>::reinit(const FluidSolver<dim> &fluid_solver)
    {
      // Setup triangulation and dof handlers
      this->triangulation =
        FluidSolverExtractor<dim>::get_triangulation(fluid_solver);

      auto dof_system =
        FluidSolverExtractor<dim>::get_dof_handler(fluid_solver);
      this->fe = dof_system.first;
      this->dof_handler = dof_system.second;

      auto scalar_dof_system =
        FluidSolverExtractor<dim>::get_scalar_dof_handler(fluid_solver);
      this->scalar_fe = scalar_dof_system.first;
      this->scalar_dof_handler = scalar_dof_system.second;

      // Setup solution vectors. Since the dof handler is not initialized yet,
      // these vectors will not be initialized either.
      this->fluid_present_solution =
        FluidSolverExtractor<dim>::get_solution(fluid_solver);

      // Setup parameters and shape functions
      this->parameters =
        FluidSolverExtractor<dim>::get_parameters(fluid_solver);
      this->volume_quad_formula =
        std::make_shared<QGauss<dim>>(parameters->fluid_velocity_degree + 1);
      this->face_quad_formula = std::make_shared<QGauss<dim - 1>>(
        parameters->fluid_velocity_degree + 1);

      // Setup partitions
      auto partitions = FluidSolverExtractor<dim>::get_partitions(fluid_solver);
      this->owned_partitioning = std::get<0>(partitions);
      this->relevant_partitioning = std::get<1>(partitions);
      this->locally_owned_scalar_dofs = std::get<2>(partitions);
      this->locally_relevant_scalar_dofs = std::get<3>(partitions);

      // Time will be the same as the fluid solver
      this->time = FluidSolverExtractor<dim>::get_time(fluid_solver);
    }

    template <int dim>
    TurbulenceModel<dim>::~TurbulenceModel()
    {
      timer.print_summary();
    }

    template <int dim>
    const PETScWrappers::MPI::Vector &
    TurbulenceModel<dim>::get_eddy_viscosity() noexcept
    {
      return eddy_viscosity;
    }

    template <int dim>
    void TurbulenceModel<dim>::initialize_system()
    {
      // This function must be called after the fluid solver calls its
      // initialize_system()
      system_matrix.clear();

      DynamicSparsityPattern dsp(scalar_dof_handler->n_dofs(),
                                 scalar_dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(
        *scalar_dof_handler, dsp, nonzero_constraints, false);
      sparsity_pattern.copy_from(dsp);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        Utilities::MPI::all_gather(mpi_communicator,
                                   scalar_dof_handler->n_locally_owned_dofs()),
        mpi_communicator,
        *locally_relevant_scalar_dofs);

      system_matrix.reinit(*locally_owned_scalar_dofs,
                           *locally_owned_scalar_dofs,
                           dsp,
                           mpi_communicator);
      system_rhs.reinit(*locally_owned_scalar_dofs, mpi_communicator);
      eddy_viscosity.reinit(*locally_owned_scalar_dofs,
                            *locally_relevant_scalar_dofs,
                            mpi_communicator);
    }

    template class TurbulenceModel<2>;
    template class TurbulenceModel<3>;
  } // namespace MPI
} // namespace Fluid

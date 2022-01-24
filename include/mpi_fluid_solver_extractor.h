#ifndef MPI_FLUID_EXTRACTOR
#define MPI_FLUID_EXTRACTOR

#include "mpi_fluid_solver.h"
#include "utilities.h"

#include <memory>
#include <tuple>

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    /**! \brief An extractor that can access internal data (private attributes)
     * of a fluid solver. Note: All the returned data is non-modifiable.
     */
    template <int dim>
    class FluidSolverExtractor
    {
    public:
      /// Returns the triangulation.
      static SmartPointer<const parallel::distributed::Triangulation<dim>>
      get_triangulation(const FluidSolver<dim> &);

      /// Returns the FEsystem object and corresponding DoFHandler of the fluid
      /// solver.
      static std::pair<SmartPointer<const FESystem<dim>>,
                       SmartPointer<const DoFHandler<dim>>>
      get_dof_handler(const FluidSolver<dim> &);

      /// Returns the scalar FE object and corresponding DoFHandler of the fluid
      /// solver.
      static std::pair<SmartPointer<const FE_Q<dim>>,
                       SmartPointer<const DoFHandler<dim>>>
      get_scalar_dof_handler(const FluidSolver<dim> &);

      /// Returns the parameter handler.
      static const Parameters::AllParameters *
      get_parameters(const FluidSolver<dim> &);

      /// Retruns the partitions. The first and second items in the returned
      /// tuple are owned and relevant partitions for velocity-pressure dofs.
      /// The third and fourth items are owned and relevant partitions for
      /// scalar dofs.
      static std::tuple<const std::vector<IndexSet> *,
                        const std::vector<IndexSet> *,
                        const IndexSet *,
                        const IndexSet *>
      get_partitions(const FluidSolver<dim> &);

      /// Returns the present fluid solution.
      static const PETScWrappers::MPI::BlockVector *
      get_solution(const FluidSolver<dim> &);

      /// Returns the time handler.
      static const Utils::Time *get_time(const FluidSolver<dim> &);
    };

  } // namespace MPI
} // namespace Fluid

#endif

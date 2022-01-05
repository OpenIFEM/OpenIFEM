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
    }

    template class SUPGInsIM<2>;
    template class SUPGInsIM<3>;
  } // namespace MPI
} // namespace Fluid

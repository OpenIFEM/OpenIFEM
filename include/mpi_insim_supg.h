#ifndef MPI_INSIM_SUPG
#define MPI_INSIM_SUPG

#include "mpi_supg_solver.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    extern template class SUPGFluidSolver<2>;
    extern template class SUPGFluidSolver<3>;

    /*! \brief the parallel SUPG Incompresisble Navier Stokes equation solver
     *
     * This is an instance of MPI SUPG fluid solver that implements a
     * incompressible Navier Stokes formulation. It only implements assembly
     * method. Other components are implemented in the base class
     * Fluid::MPI::SUPGFluidSolver<dim>.
     */
    template <int dim>
    class SUPGInsIM : public SUPGFluidSolver<dim>
    {
      MPIFluidSolverInheritanceMacro();
      MPISUPGFluidSolverInheritanceMacro();

    public:
      //! Constructor.
      SUPGInsIM(parallel::distributed::Triangulation<dim> &,
                const Parameters::AllParameters &);
      ~SUPGInsIM(){};
      //! Run the simulation.
      using SUPGFluidSolver<dim>::run;

    protected:
      /*! \brief Assemble the system matrix and the RHS.
       *
       * The implementation of assembly of the incompressible
       * Navier-Stokes equations.
       */
      virtual void assemble(const bool use_nonzero_constraints) override;
    };
  } // namespace MPI
} // namespace Fluid

#endif

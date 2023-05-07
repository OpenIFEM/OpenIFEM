#ifndef MPI_SCNSIM
#define MPI_SCNSIM

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

    /*! \brief the parallel Slightly Compresisble Navier Stokes equation solver
     *
     * This is an instance of MPI SUPG fluid solver that implements a slightly
     * compressible Navier Stokes formulation. It only implements assembly
     * method. Other components are implemented in the base class
     * Fluid::MPI::SUPGFluidSolver<dim>.
     */
    template <int dim>
    class SCnsIM : public SUPGFluidSolver<dim>
    {
      MPIFluidSolverInheritanceMacro();
      MPISUPGFluidSolverInheritanceMacro();

    public:
      //! Constructor.
      SCnsIM(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &);
      ~SCnsIM(){};
      //! Run the simulation.
      using SUPGFluidSolver<dim>::run;

    protected:
      /*! \brief Assemble the system matrix and the RHS.
       *
       * The implementation of assembly of the slightly compressible
       * Navier-Stokes equations.
       */
      virtual void assemble(const bool use_nonzero_constraints) override;
    };
  } // namespace MPI
} // namespace Fluid

#endif

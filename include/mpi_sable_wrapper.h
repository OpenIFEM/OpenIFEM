#ifndef MPI_SABLE
#define MPI_SABLE

#include "mpi_fluid_solver.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    template <int dim>
    class SableWrap : public FluidSolver<dim>
    {
      MPIFluidSolverInheritanceMacro();

    public:
      //! Constructor.
      SableWrap(parallel::distributed::Triangulation<dim> &,
                const Parameters::AllParameters &,
                std::vector<int> &,
                MPI_Comm);
      ~SableWrap(){};
      //! Run the simulation.
      void run();

    private:
      using FluidSolver<dim>::initialize_system;

      /*! \brief Run the simulation for one time step.
       *
       *  The two input arguments are not used in the wrapper
       */
      void run_one_step(bool apply_nonzero_constraints = true,
                        bool assemble_system = true) override;

      // Vector which stores Sable processor ids
      std::vector<int> sable_ids;

      // No. of nodes/elements in Sable
      int sable_no_nodes, sable_no_ele, sable_no_nodes_one_dir;

      bool is_comm_active = true;

      bool All(bool my_b);

      void get_dt_sable();

      void Max(int &send_buffer);

      void Max(double &send_biffer);

      // Vectors to store non ghost nodes and cells ids
      std::vector<int> non_ghost_cells;
      std::vector<int> non_ghost_nodes;

      // Function which finds out ghost nodes and cells ids in Sable mesh
      void find_ghost_nodes();

      // Recieve solution from Sable
      void rec_data(double **rec_buffer,
                    const std::vector<int> &cmapp,
                    const std::vector<int> &cmapp_sizes,
                    int data_size);

      // Send solution to Sable
      void send_data(double **send_buffer,
                     const std::vector<int> &cmapp,
                     const std::vector<int> &cmapp_sizes);
    };
  } // namespace MPI
} // namespace Fluid

#endif

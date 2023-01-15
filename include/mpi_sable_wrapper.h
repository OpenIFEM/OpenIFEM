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
      void initialize_system() override;

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

      PETScWrappers::MPI::BlockVector fsi_force;
      // Vector to store Dirichlet bc values for artificial fluid
      PETScWrappers::MPI::BlockVector fsi_velocity;

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

      // SABLE to OpenIFEM DOF map
      std::vector<int> sable_openifem_dof_map;
      // OpenIFEM to SABLE DOF map
      Vector<double> openifem_sable_dof_map;
      // Create map of DOF numbering between OpenIFEM and SABLE
      void create_dof_map();

      // Recieve solution from Sable
      void rec_data(double **rec_buffer,
                    const std::vector<int> &cmapp,
                    const std::vector<int> &cmapp_sizes,
                    int data_size);

      // Send solution to Sable
      void send_data(double **send_buffer,
                     const std::vector<int> &cmapp,
                     const std::vector<int> &cmapp_sizes);

      // Recieve velocity from SABLE
      void rec_velocity(const int &sable_n_nodes);

      void send_fsi_force(const int &sable_n_nodes);
    };
  } // namespace MPI
} // namespace Fluid

#endif
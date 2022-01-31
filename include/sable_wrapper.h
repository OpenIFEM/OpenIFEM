#ifndef SABLE
#define SABLE

#include "fluid_solver.h"
#include <mpi.h>

template <int>
class FSI;
template <int>
class OpenIFEM_Sable_FSI;

namespace Fluid
{
  using namespace dealii;

  extern template class FluidSolver<2>;
  extern template class FluidSolver<3>;

  /*! \brief Coupling wrapper for SABLE
   */
  template <int dim>
  class SableWrap : public FluidSolver<dim>
  {
  public:
    friend FSI<dim>;
    friend OpenIFEM_Sable_FSI<dim>;
    SableWrap(Triangulation<dim> &,
              const Parameters::AllParameters &,
              std::vector<int> &,
              std::shared_ptr<Function<dim>> bc =
                std::make_shared<Functions::ZeroFunction<dim>>(
                  Functions::ZeroFunction<dim>(dim + 1)));
    ~SableWrap(){};
    void run() override;

  private:
    struct CellStress;

    using FluidSolver<dim>::setup_dofs;
    using FluidSolver<dim>::initialize_system;
    using FluidSolver<dim>::dofs_per_block;
    using FluidSolver<dim>::triangulation;
    using FluidSolver<dim>::fe;
    using FluidSolver<dim>::scalar_fe;
    using FluidSolver<dim>::dof_handler;
    using FluidSolver<dim>::scalar_dof_handler;
    using FluidSolver<dim>::volume_quad_formula;
    using FluidSolver<dim>::face_quad_formula;
    using FluidSolver<dim>::zero_constraints;
    using FluidSolver<dim>::nonzero_constraints;
    using FluidSolver<dim>::present_solution;
    using FluidSolver<dim>::system_rhs;
    using FluidSolver<dim>::time;
    using FluidSolver<dim>::timer;
    using FluidSolver<dim>::parameters;
    using FluidSolver<dim>::cell_property;
    using FluidSolver<dim>::boundary_values;
    using FluidSolver<dim>::stress;
    using FluidSolver<dim>::fsi_force;
    using FluidSolver<dim>::fsi_force_acceleration_part;
    using FluidSolver<dim>::fsi_force_stress_part;

    /// Specify the sparsity pattern and reinit matrices and vectors based on
    /// the dofs and constraints.
    void initialize_system() override;

    /*! assemble interaction force. Integrate force calculaed in find_fluid_bc()
     * in openifem_sable_fsi.cpp over the Eulerian grid cells.
     */
    void assemble_force();

    /*! \brief Run the simulation for one time step.
     *
     *  The two input arguments are not used in the wrapper
     */
    void run_one_step(bool apply_nonzero_constraints = true,
                      bool assemble_system = true) override;

    /// Output in vtu format.
    virtual void output_results(const unsigned int) const override;

    // FESystem and DofHandler defined only for outputing vector values
    // quantities
    FESystem<dim> fe_vector_output;
    DoFHandler<dim> dof_handler_vector_output;

    /// Block vector to store nodal fsi acceleration
    BlockVector<double> fsi_acceleration;

    // Vector to store nodal fsi stress
    std::vector<Vector<double>> fsi_stress;

    // Vector to store Dirichlet bc values for artificial fluid
    BlockVector<double> fsi_velocity;

    // Vector which stores Sable processor ids
    std::vector<int> sable_ids;

    // No. of nodes/elements in Sable
    int sable_no_nodes, sable_no_ele, sable_no_nodes_one_dir;

    // Recieve solution from Sable
    void rec_data(double **rec_buffer,
                  const std::vector<int> &cmapp,
                  const std::vector<int> &cmapp_sizes,
                  int data_size);

    void rec_velocity(const int &sable_n_nodes);

    void rec_stress(const int &sable_n_elements);

    bool All(bool my_b);

    void get_dt_sable();

    void Max(int &send_buffer);

    void Max(double &send_biffer);

    bool is_comm_active = true;

    // Send solution to Sable
    void send_data(double **send_buffer,
                   const std::vector<int> &cmapp,
                   const std::vector<int> &cmapp_sizes);

    void send_fsi_force(const int &sable_n_nodes);

    void send_indicator(const int &sable_n_elements, const int &sable_n_nodes);

    // Function which finds out ghost nodes and cells ids in Sable mesh
    void find_ghost_nodes();

    // Vectors to store non ghost nodes and cells ids
    std::vector<int> non_ghost_cells;
    std::vector<int> non_ghost_nodes;

    CellDataStorage<typename Triangulation<dim>::active_cell_iterator,
                    CellStress>
      cell_stress;

    /// A data structure that stores cell-wise stress received from SABLE
    struct CellStress
    {
      // cell-wise stress considering all background materials and averaged with
      // volume fraction
      std::vector<double> cell_stress_vf_avg;
      // cell-wise stress for only selected background material without volume
      // fraction averaging
      std::vector<double> cell_stress_not_vf_avg;
    };
  };
} // namespace Fluid

#endif

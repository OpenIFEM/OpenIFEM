#ifndef STOKES
#define STOKES

#include "fluid_solver.h"

template <int>
class FSI;

namespace Fluid
{
  using namespace dealii;

  extern template class FluidSolver<2>;
  extern template class FluidSolver<3>;

  template <int dim>
  struct InnerPreconditioner;

  // In 2D, we are going to use a sparse direct solver as preconditioner:
  template <>
  struct InnerPreconditioner<2>
  {
    using type = SparseDirectUMFPACK;
  };

  // And the ILU preconditioning in 3D, called by SparseILU:
  template <>
  struct InnerPreconditioner<3>
  {
    using type = SparseILU<double>;
  };

  template <int dim>
  class Stokes : public FluidSolver<dim>
  {
  public:
    friend FSI<dim>;

    Stokes(Triangulation<dim> &,
           const Parameters::AllParameters &,
           std::shared_ptr<Function<dim>> bc =
             std::make_shared<Functions::ZeroFunction<dim>>(
               Functions::ZeroFunction<dim>(dim + 1)));
    ~Stokes(){};
    void run() override;

  private:
    // class BlockSchurPreconditioner;

    using FluidSolver<dim>::setup_dofs;
    using FluidSolver<dim>::make_constraints;
    using FluidSolver<dim>::setup_cell_property;
    using FluidSolver<dim>::initialize_system;
    using FluidSolver<dim>::refine_mesh;
    // using FluidSolver<dim>::output_results;
    using FluidSolver<dim>::update_stress;

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
    using FluidSolver<dim>::sparsity_pattern;
    using FluidSolver<dim>::system_matrix;
    using FluidSolver<dim>::mass_matrix;
    using FluidSolver<dim>::mass_schur_pattern;
    using FluidSolver<dim>::mass_schur;
    using FluidSolver<dim>::present_solution;
    using FluidSolver<dim>::solution_increment;
    using FluidSolver<dim>::system_rhs;
    using FluidSolver<dim>::time;
    using FluidSolver<dim>::timer;
    using FluidSolver<dim>::parameters;
    using FluidSolver<dim>::cell_property;
    using FluidSolver<dim>::boundary_values;
    using FluidSolver<dim>::stress;
    // using FluidSolver<dim>::run_one_step;

    // override the following functions to folow step 22, these needs to be
    // fixed to follow openIFEM in the future

    // void setup_dofs();

    void initialize_system() override;

    void output_results(const unsigned int) const;

    void set_up_boundary_values();

    //

    void assemble();

    std::pair<unsigned int, double> solve();

    void run_one_step(bool apply_nonzero_constraints,
                      bool assemble_system = true) override;

    void run_one_step();

    BlockVector<double> solution;

    // from step 22

    BlockSparsityPattern preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;
    AffineConstraints<double> constraints;

    std::shared_ptr<typename InnerPreconditioner<dim>::type> A_preconditioner;
  };

} // namespace Fluid

#endif // STOKES
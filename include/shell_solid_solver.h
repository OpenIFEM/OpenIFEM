#ifndef SHELL_ELEMENT
#define SHELL_ELEMENT

#include "fem-shell.h"
#include "solid_solver.h"
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

template <int>
class FSI;

namespace Solid
{
  using namespace dealii;

  extern template class SolidSolver<2, 3>;

  /// Base class for all solid solvers.
  class ShellSolidSolver : public SolidSolver<2, 3>
  {
  public:
    friend FSI<3>;

    ShellSolidSolver(Triangulation<2, 3> &, const Parameters::AllParameters &);
    ~ShellSolidSolver(){};

  private:
    using SolidSolver<2, 3>::triangulation;
    using SolidSolver<2, 3>::parameters;
    using SolidSolver<2, 3>::dof_handler;
    using SolidSolver<2, 3>::scalar_dof_handler;
    using SolidSolver<2, 3>::fe;
    using SolidSolver<2, 3>::scalar_fe;
    using SolidSolver<2, 3>::volume_quad_formula;
    using SolidSolver<2, 3>::face_quad_formula;
    using SolidSolver<2, 3>::constraints;
    using SolidSolver<2, 3>::pattern;
    using SolidSolver<2, 3>::system_matrix;
    using SolidSolver<2, 3>::mass_matrix;
    using SolidSolver<2, 3>::system_rhs;
    using SolidSolver<2, 3>::current_acceleration;
    using SolidSolver<2, 3>::current_velocity;
    using SolidSolver<2, 3>::current_displacement;
    using SolidSolver<2, 3>::previous_acceleration;
    using SolidSolver<2, 3>::previous_velocity;
    using SolidSolver<2, 3>::previous_displacement;
    using SolidSolver<2, 3>::strain;
    using SolidSolver<2, 3>::stress;
    using SolidSolver<2, 3>::time;
    using SolidSolver<2, 3>::timer;
    using SolidSolver<2, 3>::cell_property;

    void initialize_system();

    virtual void update_strain_and_stress() override;

    /** Assemble the lhs and rhs at the same time. */
    void assemble_system(bool);

    /// Run one time step.
    void run_one_step(bool);

    // std::unique_ptr<shellsolid> m_shell;

    std::vector<int> vertex_mapping;

    std::vector<int> cell_mapping;

    void synchronize();
  };
} // namespace Solid
#endif
#ifndef SHELL_ELEMENT
#define SHELL_ELEMENT

#include "fem-shell.h"
#include "solid_solver.h"
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

template <int>
class FSI;

namespace ShellSolid
{
}

namespace libMesh
{
}

namespace Solid
{
  using namespace dealii;

  extern template class SolidSolver<2, 3>;

  /// Base class for all solid solvers.
  class ShellSolidSolver : public SolidSolver<2, 3>
  {
  public:
    friend FSI<3>;

    ShellSolidSolver(Triangulation<2, 3> &,
                     const Parameters::AllParameters &,
                     libMesh::LibMeshInit *);
    ~ShellSolidSolver(){};

    void get_forcing_file(const std::string &);

    void run();

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
    using SolidSolver<2, 3>::current_displacement;
    using SolidSolver<2, 3>::strain;
    using SolidSolver<2, 3>::stress;
    using SolidSolver<2, 3>::time;
    using SolidSolver<2, 3>::timer;
    using SolidSolver<2, 3>::cell_property;

    void initialize_system();

    void construct_mesh();

    void setup_dofs();

    virtual void update_strain_and_stress() override;

    /** Assemble the lhs and rhs at the same time. */
    void assemble_system(bool);

    /// Run one time step.
    void run_one_step(bool);

    /// Synchronize the solution and stress
    void synchronize();

    // Method to get solution from m_shell
    void get_solution();

    void get_stress();

    void output_results(const unsigned int);

    libMesh::LibMeshInit *libmesh_init;

    libMesh::SerialMesh m_mesh;

    ShellSolid::shellparam shell_params;

    Vector<double> current_drilling;

    std::unique_ptr<ShellSolid::shellsolid> m_shell;
  };
} // namespace Solid
#endif
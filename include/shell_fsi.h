#ifndef SHELL_FSI_H
#define SHELL_FSI_H

#include "mpi_fsi.h"
#include "shell_solid_solver.h"

using namespace dealii;

extern template class Utils::GridInterpolator<3, PETScWrappers::MPI::Vector>;

namespace MPI
{
  class ShellFSI : public FSI<3>
  {
  public:
    ShellFSI(Fluid::MPI::FluidSolver<3> &,
             Solid::MPI::SharedSolidSolver<3> &,
             Solid::ShellSolidSolver &,
             const Parameters::AllParameters &,
             bool use_dirichlet_bc = false);
    virtual void run() override;
    ~ShellFSI();

  protected:
    struct CellInterpolators;

    using FSI<3>::point_in_solid;
    using FSI<3>::update_indicator;
    using FSI<3>::move_solid_mesh;
    using FSI<3>::find_solid_bc;
    using FSI<3>::update_solid_displacement;
    using FSI<3>::find_fluid_bc;
    using FSI<3>::refine_mesh;
    using FSI<3>::fluid_solver;
    using FSI<3>::solid_solver;
    using FSI<3>::parameters;
    using FSI<3>::time;
    using FSI<3>::timer;
    using FSI<3>::solid_box;
    using FSI<3>::use_dirichlet_bc;

    virtual void update_solid_box();

    void construct_interpolatioin_rule();

    void interpolate_to_shell();

    Solid::ShellSolidSolver &shell_solver;

    CellDataStorage<typename Triangulation<2, 3>::active_cell_iterator,
                    CellInterpolators>
      cell_interpolators;

    struct CellInterpolators
    {
      std::unique_ptr<Utils::GridInterpolator<3, Vector<double>>>
        interpolator;
    };
  };
} // namespace MPI
#endif
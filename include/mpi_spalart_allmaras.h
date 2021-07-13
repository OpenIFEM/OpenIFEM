#ifndef MPI_SA_MODEL
#define MPI_SA_MODEL

#include "inheritance_macros.h"
#include "mpi_turbulence_model.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    extern template class TurbulenceModel<2>;
    extern template class TurbulenceModel<3>;

    template <int dim>
    class SpalartAllmaras : public TurbulenceModel<dim>
    {
      MPITurbulenceModelInheritanceMacro();

    public:
      friend TurbulenceModelFactory<dim>;

      virtual void run_one_step(bool) override;

      virtual void make_constraints() override;

      virtual void setup_cell_property() override;

      virtual void initialize_system() override;

      virtual void save_checkpoint(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>> &)
        override;

      virtual bool load_checkpoint() override;

      virtual void pre_refine_mesh(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>> &)
        override;

      virtual void post_refine_mesh(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>> &)
        override;

      //! Update the moving boundaries
      void update_moving_wall_distance(
        const std::set<typename Triangulation<dim>::active_vertex_iterator> &);

      virtual void update_boundary_condition(bool) override;

      //! Desctructor
      ~SpalartAllmaras(){};

    protected:
      struct WallDistance;

      //! Constructor
      SpalartAllmaras() = delete;
      SpalartAllmaras(const FluidSolver<dim> &);

      void assemble(bool);

      std::pair<unsigned, double> solve(const bool);

      void update_eddy_viscosity();

      //! Solution of $\tilde{\nu}$
      PETScWrappers::MPI::Vector present_solution;

      /// The increment at a certain Newton iteration.
      PETScWrappers::MPI::Vector newton_update;

      /**
       * The latest know solution plus the cumulation of all newton_updates
       * in the current time step, which approaches to the new present_solution.
       */
      PETScWrappers::MPI::Vector evaluation_point;

      CellDataStorage<
        typename parallel::distributed::Triangulation<dim>::cell_iterator,
        WallDistance>
        wall_distance;

      // A data structure that caches the shortest distance from a field point
      // to a wall. This value is used in the computation of eddy production and
      // destruction terms.
      struct WallDistance
      {
        // The minimum distance to a fluid boundary. This value is only computed
        // once.
        double fixed_wall_distance;
        // The minimum distance to an FSI interface. This value is computed
        // every time step. It may (FSI) or may not (CFD) present.
        std::optional<double> moving_wall_distance;
      };
    };
  } // namespace MPI
} // namespace Fluid

#endif

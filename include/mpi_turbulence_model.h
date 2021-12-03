#ifndef MPI_TURBULENCE
#define MPI_TURBULENCE

#include "mpi_fluid_solver.h"
#include "mpi_fluid_solver_extractor.h"
#include "utilities.h"

namespace Fluid
{
  namespace MPI
  {
    using namespace dealii;

    /*! \brief Factory class for turbulence model. A turbulence model can only
     * be created by factory method.
     */
    template <int dim>
    class TurbulenceModelFactory
    {
      friend FluidSolver<dim>;
      //! Factory creator
      static TurbulenceModel<dim> *create(const FluidSolver<dim> &,
                                          const std::string &);
    };

    /*! \brief Base class for all RANS turbulence models. The common idea of
     * RANS turbulence models is to provide a eddy viscosity in addition to
     * laminar viscosity in assembly. To use them, call
     * Fluid::MPI::FluidSolver::attach_turbulence_model() before the run.
     */
    template <int dim>
    class TurbulenceModel
    {
    public:
      friend TurbulenceModelFactory<dim>;

      void reinit(const FluidSolver<dim> &);

      //! Pass eddy viscosity to the fluid solver.
      const PETScWrappers::MPI::Vector &get_eddy_viscosity() noexcept;

      //! Connect FSI indicator field. In FSI simulation the turbulence model
      //! needs to know which elements are artificial.
      void connect_indicator_field(
        const std::function<
          int(const typename DoFHandler<dim>::active_cell_iterator &)>);

      //! Update boundary condition for turbulence model. This is used in FSI
      //! problems.
      virtual void update_boundary_condition(bool){};

      //! Given the wall velocity and distance. Return the shear velocity
      virtual double get_shear_velocity(double, double) = 0;

      /// Virtual method to be called in fluid solver time loop.
      virtual void run_one_step(bool) = 0;

      /// Setup zero and nonzero constraints.
      virtual void make_constraints() = 0;

      /// Initialize cell properties, this could vary with different models. For
      /// example, nearest wall distance in S-A model.
      virtual void setup_cell_property(){};

      /// Specify the sparsity pattern and reinit matrices and vectors based on
      /// fluid dofs.
      virtual void initialize_system();

      /// Save checkpoint is virtual because different models has different
      /// variables to solve.
      virtual void save_checkpoint(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
          &) = 0;

      /// Load checkpoint is virtual for the same reason as save checkpoint
      virtual bool load_checkpoint() = 0;

      /// Transfer the solution to the locally refined mesh. pre_refine_mesh
      /// should be called before adapative refinement.
      virtual void pre_refine_mesh(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
          &) = 0;

      /// Transfer the solution to the locally refined mesh. post_refine_mesh
      /// should be called after adapative refinement.
      virtual void post_refine_mesh(
        std::optional<parallel::distributed::
                        SolutionTransfer<dim, PETScWrappers::MPI::Vector>>
          &) = 0;

      //! Desctructor
      virtual ~TurbulenceModel();

    protected:
      //! Constructor
      TurbulenceModel() = delete;
      TurbulenceModel(const FluidSolver<dim> &);

      //! Pointers for the triangulation and dof handlers stored in fluid solver
      SmartPointer<const parallel::distributed::Triangulation<dim>>
        triangulation;
      SmartPointer<const DoFHandler<dim>> dof_handler;
      SmartPointer<const DoFHandler<dim>> scalar_dof_handler;
      SmartPointer<const FESystem<dim>> fe;
      SmartPointer<const FE_Q<dim>> scalar_fe;

      //! locally stored data
      std::shared_ptr<QGauss<dim>> volume_quad_formula;
      std::shared_ptr<QGauss<dim - 1>> face_quad_formula;

      //! Constraints
      AffineConstraints<double> zero_constraints;
      AffineConstraints<double> nonzero_constraints;

      //! Used in FSI. Function that returns the indicator
      std::optional<std::function<int(
        const typename DoFHandler<dim>::active_cell_iterator &)>>
        indicator_function;

      //! Matrix and vector for the linear algebraic problem
      SparsityPattern sparsity_pattern;
      PETScWrappers::MPI::SparseMatrix system_matrix;

      PETScWrappers::MPI::Vector system_rhs;

      //! Present solution of the fluid solver
      const PETScWrappers::MPI::BlockVector *fluid_present_solution;

      //! Solution from the RANS turbulence model
      PETScWrappers::MPI::Vector eddy_viscosity;

      const Parameters::AllParameters *parameters;

      MPI_Comm mpi_communicator;

      ConditionalOStream pcout;

      /// The IndexSets of owned and relevant velocity and pressure
      /// respectively.
      const std::vector<IndexSet> *owned_partitioning;
      const std::vector<IndexSet> *relevant_partitioning;

      /// The IndexSets of owned and relevant sclalar dofs
      const IndexSet *locally_owned_scalar_dofs;
      const IndexSet *locally_relevant_scalar_dofs;

      //! Time of the fluid solver
      const Utils::Time *time;
      //! Individual timer for turbulence model
      mutable TimerOutput timer;
    };
  } // namespace MPI
} // namespace Fluid

#endif

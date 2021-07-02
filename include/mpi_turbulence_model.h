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

    template <int dim>
    class SCnsIM;

    template <int dim>
    class TurbulenceModel
    {
    public:
      //! Fluid solver needs to copy information to a turbulence model
      friend FluidSolver<dim>;
      friend SCnsIM<dim>;

      //! Constructor
      TurbulenceModel() = delete;
      TurbulenceModel(const FluidSolver<dim> &);

      //! Factory creator
      static TurbulenceModel<dim> *create(const FluidSolver<dim> &,
                                          const std::string &);

      void reinit(const FluidSolver<dim> &);

      //! Desctructor
      virtual ~TurbulenceModel();

    protected:
      virtual void run_one_step(bool) = 0;

      virtual void make_constraints() = 0;

      virtual void setup_cell_property() = 0;

      virtual void initialize_system();

      //! Pass eddy viscosity to the fluid solver
      const PETScWrappers::MPI::Vector &get_eddy_viscosity() noexcept;

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

      AffineConstraints<double> zero_constraints;
      AffineConstraints<double> nonzero_constraints;

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

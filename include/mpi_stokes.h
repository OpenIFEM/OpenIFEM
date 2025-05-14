#ifndef MPI_STOKES
#define MPI_STOKES

#include "mpi_fluid_solver.h"
#include <cmath>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/vector.h>

namespace Fluid

{
  namespace MPI

  {
    using namespace dealii;

    namespace NumericalConstants
    {
      constexpr double DOMAIN_HEIGHT = 0.5;
      constexpr double DOMAIN_LENGTH = 1.5;
      constexpr double INLET_U_MAX = 1.0;
    } // namespace NumericalConstants

    template <int dim>
    class InletVelocity : public Function<dim>
    {
    public:
      InletVelocity(const double ramp_time)
        : Function<dim>(dim) // pass 'dim' to base so vector_value has size=dim
          ,
          ramp_time(ramp_time)
      {
      }

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &values) const override
      {
        // The current simulation time, which must be set via set_time() outside
        const double t = this->get_time();
        // Simple linear ramp up from t=0 to t=ramp_time
        const double alpha =
          (ramp_time > 1e-14) ? std::min(1.0, t / ramp_time) : 1.0;

        // Parabolic profile in y
        const double y = p[1];
        const double H = NumericalConstants::DOMAIN_HEIGHT;

        // Scale the usual 4 * Umax * (y/H)*(1-y/H) by alpha
        values(0) = alpha * (4.0 * NumericalConstants::INLET_U_MAX * (y / H) *
                             (1.0 - y / H));

        // Zero out other components
        for (unsigned int i = 1; i < dim; ++i)
          values(i) = 0.0;
      }

    private:
      const double ramp_time;
    };

    extern template class FluidSolver<2>;
    extern template class FluidSolver<3>;

    namespace InternalWrappers
    {
      template <int dim>
      class InitialVelocityWrapper : public Function<dim>
      {
      public:
        InitialVelocityWrapper(
          const std::function<double(const Point<dim> &, unsigned int)> &f,
          const unsigned int n_components = dim)
          : Function<dim>(n_components), user_function(f)
        {
        }

        // Overload scalar value() to call user_function
        virtual double value(const Point<dim> &p,
                             const unsigned int component = 0) const override
        {
          AssertIndexRange(component, this->n_components);
          return user_function(p, component);
        }

        virtual void vector_value(const Point<dim> &p,
                                  Vector<double> &values) const override
        {
          AssertDimension(values.size(), this->n_components);

          for (unsigned int c = 0; c < this->n_components; ++c)
            values[c] = user_function(p, c);
        }

      private:
        const std::function<double(const Point<dim> &, unsigned int)>
          &user_function;
      };
    } // end namespace InternalWrappers

    template <int dim>
    class Stokes : public FluidSolver<dim>
    {

    public:
      Stokes(parallel::distributed::Triangulation<dim> &,
             const Parameters::AllParameters &);

      ~Stokes(){};
      void run();
      void initialize_bcs();
      void set_up_boundary_values();
      //void build_velocity_constraints(AffineConstraints <double> &velocity_constraints);
      AffineConstraints<double> constraints;

      friend ::MPI::FSI<dim>;

    private:
      using FluidSolver<dim>::setup_dofs;
      using FluidSolver<dim>::make_constraints;
      using FluidSolver<dim>::setup_cell_property;
      using FluidSolver<dim>::initialize_system;
      using FluidSolver<dim>::refine_mesh;
      using FluidSolver<dim>::update_stress;
      using FluidSolver<dim>::dofs_per_block;
      using FluidSolver<dim>::triangulation;
      using FluidSolver<dim>::fe;
      using FluidSolver<dim>::scalar_fe;
      using FluidSolver<dim>::dof_handler;
      using FluidSolver<dim>::scalar_dof_handler;
      using FluidSolver<dim>::volume_quad_formula;
      using FluidSolver<dim>::face_quad_formula;
      using FluidSolver<dim>::sparsity_pattern;
      using FluidSolver<dim>::system_matrix;
      using FluidSolver<dim>::mass_matrix;
      using FluidSolver<dim>::mass_schur;
      using FluidSolver<dim>::present_solution;
      using FluidSolver<dim>::system_rhs;
      using FluidSolver<dim>::time;
      using FluidSolver<dim>::timer;
      using FluidSolver<dim>::parameters;
      using FluidSolver<dim>::cell_property;
      using FluidSolver<dim>::fsi_acceleration;
      using FluidSolver<dim>::fsi_stress;
      using FluidSolver<dim>::stress;
      using FluidSolver<dim>::mpi_communicator;
      using FluidSolver<dim>::pcout;
      using FluidSolver<dim>::owned_partitioning;
      using FluidSolver<dim>::relevant_partitioning;
      using FluidSolver<dim>::locally_owned_scalar_dofs;
      using FluidSolver<dim>::locally_relevant_dofs;
      using FluidSolver<dim>::locally_relevant_scalar_dofs;
      using FluidSolver<dim>::pvd_writer;
      using FluidSolver<dim>::apply_initial_condition;
      using FluidSolver<dim>::initial_condition_field;
      // using FluidSolver<dim>::set_initial_condition;

      void initialize_system() override;

      //void apply_initial_condition() override;

      void output_results(const unsigned int) const;

      // void set_up_boundary_values();

      void assemble();

      std::pair<unsigned int, double> solve();

      void run_one_step(bool apply_nonzero_constraints,
                        bool assemble_system = true) override;

      void run_one_step_new();

      void compute_ind_norms() const;

      void compute_drag_lift_coefficients();

      void compute_fluid_norms();

      void compute_energy_estimates();

      void build_mass_matrix(); // for debugging, delete later

      void compute_pressure_gradient_norm();

      PETScWrappers::MPI::SparseMatrix mass_matrix_velocity;
      
      PETScWrappers::MPI::BlockVector solution;

      BlockSparsityPattern preconditioner_sparsity_pattern;

      PETScWrappers::MPI::BlockSparseMatrix preconditioner_matrix;

      PETScWrappers::MPI::BlockVector fsi_force_acceleration_part;

      PETScWrappers::MPI::BlockVector fsi_force_stress_part;

      PETScWrappers::MPI::BlockVector fsi_force;

      InletVelocity<dim> inlet_velocity;

      PETScWrappers::MPI::BlockVector previous_solution;

     
    };

  } // namespace MPI

} // namespace Fluid

#endif // MPI_STOKES
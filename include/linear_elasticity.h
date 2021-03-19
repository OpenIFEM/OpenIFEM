#ifndef LINEAR_ELASTICITY
#define LINEAR_ELASTICITY

#include "linear_elastic_material.h"
#include "solid_solver.h"

template <int>
class FSI;

namespace Solid
{
  using namespace dealii;

  extern template class SolidSolver<2>;
  extern template class SolidSolver<3>;
  extern template class LinearElasticMaterial<2>;
  extern template class LinearElasticMaterial<3>;

  /*! \brief A time-dependent solver for linear elasticity.
   *
   * We use Newmark-beta method for time-stepping which can be either
   * explicit or implicit, first order accurate or second order accurate.
   * We fix \f$\beta = \frac{1}{2}\gamma\f$, which corresponds to
   * average acceleration method.
   */
  template <int dim>
  class LinearElasticity : public SolidSolver<dim>
  {
  public:
    friend FSI<dim>;

    /*! \brief Constructor.
     *
     * The triangulation can either be generated using dealii functions or
     * from Abaqus input file.
     * Also we use a parameter handler to specify all the input parameters.
     */
    LinearElasticity(Triangulation<dim> &, const Parameters::AllParameters &);
    /*! \brief Destructor. */
    ~LinearElasticity(){};

  private:
    /**
     * Members in its template-base class.
     * Annoying C++ feature: the compiler does not know how to access
     * template-derived members unless you tell it explicitly by using
     * declarations or this->Foo.
     */
    using SolidSolver<dim>::triangulation;
    using SolidSolver<dim>::parameters;
    using SolidSolver<dim>::dof_handler;
    using SolidSolver<dim>::scalar_dof_handler;
    using SolidSolver<dim>::fe;
    using SolidSolver<dim>::scalar_fe;
    using SolidSolver<dim>::volume_quad_formula;
    using SolidSolver<dim>::face_quad_formula;
    using SolidSolver<dim>::constraints;
    using SolidSolver<dim>::pattern;
    using SolidSolver<dim>::system_matrix;
    using SolidSolver<dim>::stiffness_matrix;
    using SolidSolver<dim>::system_rhs;
    using SolidSolver<dim>::current_acceleration;
    using SolidSolver<dim>::current_velocity;
    using SolidSolver<dim>::current_displacement;
    using SolidSolver<dim>::previous_acceleration;
    using SolidSolver<dim>::previous_velocity;
    using SolidSolver<dim>::previous_displacement;
    using SolidSolver<dim>::strain;
    using SolidSolver<dim>::stress;
    using SolidSolver<dim>::cellwise_stress;
    using SolidSolver<dim>::time;
    using SolidSolver<dim>::timer;
    using SolidSolver<dim>::cell_property;

    /**
     * Assembles lhs and rhs. At time step 0, the lhs is the mass matrix;
     * at all the following steps, it is \f$ M + \beta{\Delta{t}}^2K \f$.
     * It can also be used to assemble the RHS only, in case of time-dependent
     * Neumann boundary conditions.
     */
    void assemble(bool is_initial, bool assemble_matrix);

    /**
     * Assembles both the LHS and RHS of the system.
     */
    void assemble_system(bool is_initial);

    /**
     * Assembles only the RHS of the system.
     */
    void assemble_rhs();

    /**
     * Update the strain and stress, used in output_results and FSI.
     */
    virtual void update_strain_and_stress() override;

    /**
     * Run one time step.
     */
    void run_one_step(bool);

    std::vector<LinearElasticMaterial<dim>> material;
  };
} // namespace Solid

#endif

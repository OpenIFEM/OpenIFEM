#ifndef HYPO_ELASTICITY
#define HYPO_ELASTICITY

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <mfree_iwf/body.h>
#include <mfree_iwf/cont_mech.h>
#include <mfree_iwf/derivatives.h>
#include <mfree_iwf/material.h>
#include <mfree_iwf/neighbor_search.h>
#include <mfree_iwf/particle.h>
#include <mfree_iwf/vtk_writer.h>

#include "solid_solver.h"

template <int>
class FSI;

namespace Solid
{
  using namespace dealii;

  extern template class SolidSolver<2>;
  extern template class SolidSolver<3>;

  template <int dim>
  class HypoElasticity : public SolidSolver<dim>
  {
  public:
    friend FSI<dim>;

    HypoElasticity(Triangulation<dim> &,
                   const Parameters::AllParameters &,
                   double dx,
                   double hdx);
    ~HypoElasticity() {}

    void test();

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
    using SolidSolver<dim>::mass_matrix;
    using SolidSolver<dim>::system_rhs;
    using SolidSolver<dim>::current_acceleration;
    using SolidSolver<dim>::current_velocity;
    using SolidSolver<dim>::current_displacement;
    using SolidSolver<dim>::previous_acceleration;
    using SolidSolver<dim>::previous_velocity;
    using SolidSolver<dim>::previous_displacement;
    using SolidSolver<dim>::strain;
    using SolidSolver<dim>::stress;
    using SolidSolver<dim>::time;
    using SolidSolver<dim>::timer;
    using SolidSolver<dim>::cell_property;

    void initialize_system();

    virtual void update_strain_and_stress() override;

    /** Assemble the lhs and rhs at the same time. */
    void assemble_system(bool);

    /// Run one time step.
    void run_one_step(bool);

    body<particle_tl_weak> m_body;

    std::vector<int> vertex_mapping;

    void construct_particles();

    void synchronize();

    double dx;

    double hdx;
  };
} // namespace Solid

#endif

#ifndef UTILITIES
#define UTILITIES

#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <queue>
#include <unordered_set>

namespace Utils
{
  using namespace dealii;

  /// This class manages simulation time and output frequency.
  class Time
  {
  public:
    Time(const double time_end,
         const double delta_t,
         const double output_interval,
         const double refinement_interval,
         const double save_interval)
      : timestep(0),
        time_current(0.0),
        delta_t(delta_t),
        time_end(time_end),
        output_interval(output_interval),
        refinement_interval(refinement_interval),
        save_interval(save_interval)
    {
    }
    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    bool time_to_output() const;
    bool time_to_refine() const;
    bool time_to_save() const;
    void increment();
    void set_delta_t(double delta);

  private:
    unsigned int timestep;
    double time_current;
    double delta_t;
    const double time_end;
    const double output_interval;
    const double refinement_interval;
    const double save_interval;
  };

  /*! \brief A helper class to generate triangulations and specify boundary ids.
   *
   *  dealii::GridGenerator can be used to generate a few standard grids such as
   * hyperrectangle,
   *  sphere and so on. This class is based on dealii::GridGenerator to generate
   * more complicated
   *  grids that are useful to us.
   */
  template <int dim>
  class GridCreator
  {
  public:
    /*! \brief Generate triangulation for the flow around cylinder benchmark
     * case.
     *
     *  The geometry and benchmark results are reported in Turek (1996), or
     * [this webpage]
     *  (http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
     *  The boundaries are numbered such that the lower one in x-direction is 0,
     *  the upper one is 1; the lower one in y-direction is 2, the upper one is
     * 3;
     *  the lower one in z-direction is 4, the upper one is 5.
     *  the cylindrical surface is marked with boundary id 4 in 2d and 6 in 3d.
     */
    static void flow_around_cylinder(Triangulation<dim> &);
    /*! \brief Generate a nice mesh for a sphere.
     *
     * Adapted from [dealii tutorials step-6]
     * (http://www.dealii.org/developer/doxygen/deal.II/step_6.html)
     */
    static void sphere(Triangulation<dim> &tria,
                       const Point<dim> &center = Point<dim>(),
                       double radius = 1);

    static void cylinder(Triangulation<dim> &tria,
                         const double radius,
                         const double length);

  private:
    /// A helper function used by flow_around_cylinder.
    static void flow_around_cylinder_2d(Triangulation<2> &,
                                        bool compute_in_2d = true);
  };

  /*! \brief Interpolate the solution value or gradient at an arbitrary point.
   *
   * The implementation is a combination of VectorTools::point_value and
   * VectorTools::point_graident. However, this class avoids locating the given
   * point
   * for multiple times when we need to interpolate different values.
   *
   * Due to the floating point errors, locating a point can be undeterministic.
   * In addition,
   * in parallel applications, it can only be owned by one processor. In those
   * situations
   * zero is returned.
   */
  template <int dim, typename VectorType>
  class GridInterpolator
  {
  public:
    GridInterpolator(const DoFHandler<dim> &,
                     const Point<dim> &,
                     const std::vector<bool> &mask = {},
                     const typename DoFHandler<dim>::active_cell_iterator &
                       cell = typename DoFHandler<dim>::active_cell_iterator());
    void point_value(const VectorType &,
                     Vector<typename VectorType::value_type> &);
    void point_gradient(
      const VectorType &,
      std::vector<Tensor<1, dim, typename VectorType::value_type>> &);

    bool found_cell() const { return cell_found; };
    const typename DoFHandler<dim>::active_cell_iterator get_cell() const;

  private:
    const DoFHandler<dim> &dof_handler;
    const Point<dim> &point;
    bool cell_found;
    MappingQ1<dim> mapping;
    std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>>
      cell_point;
  };

  template <int dim, typename VectorType>
  class SPHInterpolator
  {
  public:
    SPHInterpolator(const DoFHandler<dim> &, const Point<dim> &);
    void point_value(const VectorType &,
                     Vector<typename VectorType::value_type> &);
    void point_gradient(
      const VectorType &,
      std::vector<Tensor<1, dim, typename VectorType::value_type>> &);

  private:
    /// Source DoFHandler
    const DoFHandler<dim> &dof_handler;
    /// Target point to interpolate to
    const Point<dim> target;
    /**
     * Source nodes that contribute to the target point,
     * denoted as pairs of cell iterator, kernel value,
     * and kernel gradient.
     */
    std::vector<
      std::pair<typename DoFHandler<dim>::active_cell_iterator, double>>
      sources;

    double cubic_spline(const Point<dim> &, const Point<dim> &, double);
  };

  template <int dim, typename MeshType>
  class CellLocator
  {
  public:
    CellLocator(DoFHandler<dim> &,
                const Point<dim> &,
                const typename MeshType::active_cell_iterator &);
    // Use breadth first search from the hint to find and return the iterator
    // of the cell where the point is inside.
    const typename MeshType::active_cell_iterator search();
    bool found_cell() const { return cell_found; };

  private:
    DoFHandler<dim> &dof_handler;
    const Point<dim> &point;
    const typename MeshType::active_cell_iterator hint;
    bool cell_found;
  };
} // namespace Utils

#endif

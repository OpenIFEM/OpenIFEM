#ifndef UTILITIES
#define UTILITIES

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

namespace Utils
{
  using namespace dealii;
  /** \brief This class manages simulation time and output frequency.
   *
   */
  class Time
  {
  public:
    Time(const double time_end,
         const double delta_t,
         const double output_interval)
      : timestep(0),
        time_current(0.0),
        time_end(time_end),
        delta_t(delta_t),
        output_interval(output_interval)
    {
    }
    double current() const { return time_current; }
    double end() const { return time_end; }
    double get_delta_t() const { return delta_t; }
    unsigned int get_timestep() const { return timestep; }
    bool time_to_output() const
    {
      return (timestep % static_cast<unsigned int>(output_interval / delta_t) ==
              0);
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
    const double output_interval;
  };

  /** \brief A helper class to generate triangulations which are not implemented
   * as
   *         functions in dealii.
   *
   *  dealii::GridGenerator can be used to generate a few standard grids such as
   * hyperrectangle,
   *  sphere and so on. This class is based on dealii::GridGenerator to generate
   * more complicated
   *  grids that are useful to us.
   */
  class GridCreator
  {
  public:
    /** \brief Generate triangulation for the flow around cylinder benchmark
     * case.
     *
     *  The geometry and benchmark results are reported in Turek (1996), or
     * [this webpage]
     *  (http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html)
     *  For both 2d and 3d cases, the right boundary is marked with boundary id
     * 2,
     *  the cylindrical surface is marked with boundary id 1,
     *  and all the rest of surfaces are marked with id 0.
     */
    static void flow_around_cylinder(Triangulation<2> &);
    static void flow_around_cylinder(Triangulation<3> &);

  private:
    /** \brief A helper function used by flow_around_cylinder.
     */
    static void flow_around_cylinder_2d(Triangulation<2> &,
                                        bool compute_in_2d = true);
  };
}

#endif

/**
 * This program tests serial Slightly Compressible solver with an
 * acoustic wave in 2D duct case.
 * A Gaussian pulse is used as the time dependent BC with max velocity
 * equal to 6cm/s.
 * This test takes about 770s.
 */
#include "mpi_scnsim.h"
#include "parameters.h"
#include "utilities.h"
#include "mpi_shared_hypo_elasticity.h"
#include "mpi_fsi.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Solid::MPI::SharedHypoElasticity<2>;
extern template class MPI::FSI<2>;

using namespace dealii;

template <int dim>
class TimeDependentBoundaryValues : public Function<dim>
{
public:
  TimeDependentBoundaryValues() : Function<dim>(dim + 1) {}
  TimeDependentBoundaryValues(double t, double dt)
    : Function<dim>(dim + 1, t), dt(dt)
  {
  }
  virtual double value(const Point<dim> &p, const unsigned int component) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;

private:
  double time_value(const Point<dim> &p,
                    const unsigned int component,
                    const double t) const;
  double dt;
};

template <int dim>
double
TimeDependentBoundaryValues<dim>::value(const Point<dim> &p,
                                        const unsigned int component) const
{
  return time_value(p, component, this->get_time()) -
         time_value(p, component, this->get_time() - dt);
}

template <int dim>
void TimeDependentBoundaryValues<dim>::vector_value(
  const Point<dim> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = TimeDependentBoundaryValues::value(p, c);
}

template <int dim>
double TimeDependentBoundaryValues<dim>::time_value(
  const Point<dim> &p, const unsigned int component, const double t) const
{
  Assert(component < this->n_components,
         ExcIndexRange(component, 0, this->n_components));
  if (component == 0 && std::abs(p[0]) < 1e-10)
    {
      // Gaussian wave
      return 60 * exp(-0.5 * pow((t - 0.5e-4) / 0.15e-4, 2));
    }
  return 0;
}

int main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);

      double L = 4, H = 1, l = 0.125, h = l / 8, hdx = 1.3;

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {8, 2}, Point<2>(0, 0), Point<2>(L + 1, H), true);
          // initialize the shared pointer the time dependent BC
          std::shared_ptr<TimeDependentBoundaryValues<2>> ptr =
            std::make_shared<TimeDependentBoundaryValues<2>>(
              TimeDependentBoundaryValues<2>(params.time_step,
                                             params.time_step));
          Fluid::MPI::SCnsIM<2> fluid(tria, params, ptr);

          Triangulation<2> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {static_cast<unsigned int>(l / h),
             static_cast<unsigned int>(H / h)},
            Point<2>(L - l, 0),
            Point<2>(L, H),
            true);
          Solid::MPI::SharedHypoElasticity<2> solid(solid_tria, params,
                                                    h, hdx);

          MPI::FSI<2> fsi(fluid, solid, params);
          fsi.run();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}

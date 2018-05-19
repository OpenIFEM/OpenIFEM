/**
 * This program tests serial Slightly Compressible solver with a PML
 * absorbing boundary condition.
 * A Gaussian pulse is used as the time dependent BC with max velocity
 * equal to 6cm/s.
 * The PML boundary condition (1cm long) is applied to the right boundary.
 * This test takes about 400s.
 */
#include "mpi_scnsim.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Fluid::MPI::SCnsIM<3>;

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
class SigmaPMLField : public Function<dim>
{
public:
  SigmaPMLField(double sig, double l)
    : Function<dim>(), SigmaPMLMax(sig), PMLLength(l)
  {
  }
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const;

private:
  double SigmaPMLMax;
  double PMLLength;
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
      return 6.0 * exp(-0.5 * pow((t - 0.5e-6) / 0.15e-6, 2));
    }
  return 0;
}

template <int dim>
double SigmaPMLField<dim>::value(const Point<dim> &p,
                                 const unsigned int component) const
{
  (void)component;
  (void)p;
  double SigmaPML = 0.0;
  double boundary = 1.4;
  // For tube acoustics
  if (p[0] > boundary - PMLLength)
    // A quadratic increasing function from boundary-PMLlength to the boundary
    SigmaPML = SigmaPMLMax * pow((p[0] + PMLLength - boundary) / PMLLength, 4);
  return SigmaPML;
}

template <int dim>
void SigmaPMLField<dim>::value_list(const std::vector<Point<dim>> &points,
                                    std::vector<double> &values,
                                    const unsigned int component) const
{
  (void)component;
  for (unsigned int i = 0; i < points.size(); ++i)
    values[i] = this->value(points[i]);
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

      double L = 1.4, H = 0.4;
      double PMLlength = 1.2, SigmaMax = 340000;

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            tria, {7, 2}, Point<2>(0, 0), Point<2>(L, H), true);
          // initialize the shared pointer the time dependent BC
          std::shared_ptr<TimeDependentBoundaryValues<2>> ptr =
            std::make_shared<TimeDependentBoundaryValues<2>>(
              TimeDependentBoundaryValues<2>(params.time_step,
                                             params.time_step));
          // initialize the pml field
          auto pml = std::make_shared<SigmaPMLField<2>>(
            SigmaPMLField<2>(SigmaMax, PMLlength));
          Fluid::MPI::SCnsIM<2> flow(tria, params, ptr, pml);
          flow.run();
          auto solution = flow.get_current_solution();
          // The wave is absorbed at last, so the solution should be zero.
          auto v = solution.block(0);
          double vmax = v.max();
          double verror = std::abs(vmax);
          AssertThrow(verror < 5e-2,
                      ExcMessage("Maximum velocity is incorrect!"));
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

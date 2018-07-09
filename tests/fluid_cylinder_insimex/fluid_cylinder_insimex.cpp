/**
 * This program tests serial InsIMEX solver with a 2D flow around cylinder
 * case.
 * Hard-coded parabolic velocity input is used, and Re = 20.
 * To save time, the global mesh refinement level is set to 1.
 * For real application, 2 should be used.
 * This test takes about 72.
 */
#include "insimex.h"
#include "parameters.h"
#include "utilities.h"

extern template class Fluid::InsIMEX<2>;
extern template class Fluid::InsIMEX<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues() : Function<dim>(dim + 1) {}
  virtual double value(const Point<dim> &p, const unsigned int component) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
};

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const
{
  double left_boundary = (dim == 2 ? 0.0 : -0.3);
  if (component == 0 && std::abs(p[0] - left_boundary) < 1e-10)
    {
      // For a parabolic velocity profile, Uavg = 2/3 * Umax in 2D,
      // and 4/9 * Umax in 3D. If nu = 0.001, D = 0.1,
      // then Re = 100 * Uavg
      double Uavg = 0.2;
      double Umax = (dim == 2 ? 3 * Uavg / 2 : 9 * Uavg / 4);
      double value = 4 * Umax * p[1] * (0.41 - p[1]) / (0.41 * 0.41);
      if (dim == 3)
        {
          value *= 4 * p[2] * (0.41 - p[2]) / (0.41 * 0.41);
        }
      return value;
    }
  return 0;
}

template <int dim>
void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                       Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues::value(p, c);
}

int main(int argc, char *argv[])
{
  try
    {
      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);

      if (params.dimension == 2)
        {
          Triangulation<2> tria;
          Utils::GridCreator<2>::flow_around_cylinder(tria);
          auto ptr = std::make_shared<BoundaryValues<2>>(BoundaryValues<2>());
          Fluid::InsIMEX<2> flow(tria, params, ptr);
          flow.run();
          auto solution = flow.get_current_solution();
          // Check the max values of velocity and pressure
          auto v = solution.block(0), p = solution.block(1);
          double vmax = *std::max_element(v.begin(), v.end());
          double pmax = *std::max_element(p.begin(), p.end());
          double verror = std::abs(vmax - 0.4081072) / 0.4081072;
          double perror = std::abs(pmax - 0.1539) / 0.1539;
          AssertThrow(verror < 1e-3 && perror < 1e-3,
                      ExcMessage("Maximum velocity or pressure is incorrect!"));
        }
      else if (params.dimension == 3)
        {
          Triangulation<3> tria;
          Utils::GridCreator<3>::flow_around_cylinder(tria);
          auto ptr = std::make_shared<BoundaryValues<3>>(BoundaryValues<3>());
          Fluid::InsIMEX<3> flow(tria, params, ptr);
          flow.run();
        }
      else
        {
          AssertThrow(false, ExcMessage("This test should be run in 2D!"));
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

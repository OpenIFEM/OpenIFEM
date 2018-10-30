/**
 * 2D leaflet case with serial incompressible fluid solver and hyperelastic
 * solver.
 */
#include "mpi_fsi.h"
#include "mpi_insimex.h"
#include "mpi_scnsim.h"
#include "mpi_shared_hyper_elasticity.h"

extern template class Fluid::MPI::SCnsIM<2>;
extern template class Fluid::MPI::InsIMEX<3>;
extern template class Solid::MPI::SharedHyperElasticity<2>;
extern template class Solid::MPI::SharedHyperElasticity<3>;
extern template class MPI::FSI<2>;
extern template class MPI::FSI<3>;

const double L = 4, H = 1, a = 0.1, b = 0.4, h = 0.05, U = 1.5;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues() : Function<dim>(dim + 1) {}
  virtual double value(const Point<dim> &p, const unsigned int component) const;

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;
};

template <>
double BoundaryValues<2>::value(const Point<2> &p,
                                const unsigned int component) const
{
  if (component == 0 && std::abs(p[0]) < 1e-10 && std::abs(p[1]) > 1e-10)
    {
      return U;
    }
  return 0;
}

template <>
double BoundaryValues<3>::value(const Point<3> &p,
                                const unsigned int component) const
{
  if (component == 2 && std::abs(p[2]) < 1e-10 && std::abs(p[0]) > 1e-10 &&
      std::abs(p[1]) > 1e-10)
    {
      return U;
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

      if (params.dimension == 2)
        {
          parallel::distributed::Triangulation<2> fluid_tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(L / h),
             static_cast<unsigned int>(H / h)},
            Point<2>(0, 0),
            Point<2>(L, H),
            true);
          // Refine the middle part
          for (auto cell : fluid_tria.active_cell_iterators())
            {
              auto center = cell->center();
              if (center[0] >= L / 4 - 2 * a && center[0] <= L / 4 + 3 * a &&
                  cell->is_locally_owned())
                {
                  cell->set_refine_flag();
                }
            }
          fluid_tria.execute_coarsening_and_refinement();

          auto ptr = std::make_shared<BoundaryValues<2>>(BoundaryValues<2>());
          Fluid::MPI::SCnsIM<2> fluid(fluid_tria, params, ptr);

          Triangulation<2> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {static_cast<unsigned int>(a / h),
             static_cast<unsigned int>(b / h)},
            Point<2>(L / 4, 0),
            Point<2>(a + L / 4, b),
            true);
          Solid::MPI::SharedHyperElasticity<2> solid(solid_tria, params);

          MPI::FSI<2> fsi(fluid, solid, params);
          fsi.run();
        }
      else
        {
          parallel::distributed::Triangulation<3> fluid_tria(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(H / (2 * h)),
             static_cast<unsigned int>(H / (2 * h)),
             static_cast<unsigned int>(L / (2 * h))},
            Point<3>(0, 0, 0),
            Point<3>(H, H, L),
            true);
          auto ptr = std::make_shared<BoundaryValues<3>>(BoundaryValues<3>());
          Fluid::MPI::InsIMEX<3> fluid(fluid_tria, params, ptr);

          Triangulation<3> solid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            solid_tria,
            {static_cast<unsigned int>(b / (1 * h)),
             static_cast<unsigned int>(a / (1 * h)),
             static_cast<unsigned int>(a / (1 * h))},
            Point<3>(0, (H - a) / 2, L / 4),
            Point<3>(b, (H + a) / 2, a + L / 4),
            true);
          Solid::MPI::SharedHyperElasticity<3> solid(solid_tria, params);

          MPI::FSI<3> fsi(fluid, solid, params);
          fsi.run();
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

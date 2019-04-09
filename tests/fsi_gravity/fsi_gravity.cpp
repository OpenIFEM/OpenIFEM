#include "fsi.h"
#include "hyper_elasticity.h"
#include "insim.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;
extern template class Solid::HyperElasticity<2>;
extern template class Solid::HyperElasticity<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

extern template class FSI<2>;
extern template class FSI<3>;

int main(int argc, char *argv[])
{
  using namespace dealii;

  try
    {
      std::string infile("parameters.prm");
      if (argc > 1)
        {
          infile = argv[1];
        }
      Parameters::AllParameters params(infile);

      double L = 1, W = 2, H = 5, R = 0.125, h = 0.25;

      if (params.dimension == 2)
        {
          Triangulation<2> fluid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(W / h),
             static_cast<unsigned int>(H / h)},
            Point<2>(0, 0),
            Point<2>(W, -H),
            true);
          // Refine the middle part
          for (auto cell : fluid_tria.active_cell_iterators())
            {
              auto center = cell->center();
              if (center[0] >= W / 2 - 2 * R && center[0] <= W / 2 + 2 * R)
                {
                  cell->set_refine_flag();
                }
            }
          fluid_tria.execute_coarsening_and_refinement();
          Fluid::InsIM<2> fluid(fluid_tria, params);

          Triangulation<2> solid_tria;
          Point<2> center(L, -L);
          Utils::GridCreator<2>::sphere(solid_tria, center, R);
          Solid::HyperElasticity<2> solid(solid_tria, params);

          FSI<2> fsi(fluid, solid, params, true);
          fsi.run();
        }
      else
        {
          Triangulation<3> fluid_tria;
          dealii::GridGenerator::subdivided_hyper_rectangle(
            fluid_tria,
            {static_cast<unsigned int>(W / h),
             static_cast<unsigned int>(W / h),
             static_cast<unsigned int>(H / h)},
            Point<3>(0, 0, 0),
            Point<3>(W, W, -H),
            true);
          Fluid::InsIM<3> fluid(fluid_tria, params);

          Triangulation<3> solid_tria;
          Point<3> center(L, L, -L);
          Utils::GridCreator<3>::sphere(solid_tria, center, R);
          Solid::HyperElasticity<3> solid(solid_tria, params);

          FSI<3> fsi(fluid, solid, params, true);
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

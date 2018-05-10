#include "fsi.h"
#include "hyperelasticSolver.h"
#include "insim.h"

extern template class Fluid::InsIM<2>;
extern template class Fluid::InsIM<3>;
extern template class Solid::HyperelasticSolver<2>;
extern template class Solid::HyperelasticSolver<3>;
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

      double L = 1, W = 2, H = 5, R = 0.25, h = 0.25;

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
          Fluid::InsIM<2> fluid(fluid_tria, params);

          Triangulation<2> solid_tria;
          Point<2> center(L, -L);
          Utils::GridCreator<2>::sphere(solid_tria, center, R);
          Solid::HyperelasticSolver<2> solid(solid_tria, params);

          FSI<2> fsi(fluid, solid, params);
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

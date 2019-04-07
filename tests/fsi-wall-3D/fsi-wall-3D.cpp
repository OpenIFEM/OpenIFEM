#include "mpi_fsi.h"
#include "mpi_scnsim.h"
#include "mpi_shared_hypo_elasticity.h"
#include "parameters.h"
#include "utilities.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

extern template class Fluid::MPI::SCnsIM<3>;
extern template class Solid::MPI::SharedHypoElasticity<3>;
extern template class MPI::FSI<3>;

using namespace dealii;

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

      if (params.dimension == 3)
        {
          // Read solid mesh
          Triangulation<3> tria_solid;
          dealii::GridGenerator::subdivided_hyper_rectangle(tria_solid,
                                                            {20u, 20u, 5u},
                                                            Point<3>(0, 0, 0),
                                                            Point<3>(2, 2, 0.5),
                                                            true);

          // Read fluid mesh
          parallel::distributed::Triangulation<3> tria_fluid(MPI_COMM_WORLD);
          dealii::GridGenerator::subdivided_hyper_rectangle(tria_fluid,
                                                            {20u, 20u, 50u},
                                                            Point<3>(0, 0, 0),
                                                            Point<3>(2, 2, 5),
                                                            true);
          for (auto cell : tria_fluid.active_cell_iterators())
            {
              auto center = cell->center();
              if (center[2] >= 1.5 && center[2] <= 3)
                cell->set_refine_flag();
            }
          tria_fluid.execute_coarsening_and_refinement();

          // Translate solid mesh
          Tensor<1, 3> offset({0, 0, 2});
          GridTools::shift(offset, tria_solid);

          Fluid::MPI::SCnsIM<3> fluid(tria_fluid, params);
          Solid::MPI::SharedHypoElasticity<3> solid(
            tria_solid, params, 0.05, 1.3);
          MPI::FSI<3> fsi(fluid, solid, params);
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

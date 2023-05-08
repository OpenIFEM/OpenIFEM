// This program tests Lagrangian-Eulerian solid-solid interaction between
// OpenIFEM hyperleastic solid solver and Sable The generated executable needs
// to be run with Sable executable
#include "mpi_fsi.h"
#include "mpi_shared_linear_elasticity.h"
#include "mpi_openifem_sable_fsi.h"
#include "parameters.h"
#include "mpi_sable_wrapper.h"
#include "utilities.h"
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

extern template class Fluid::MPI::SableWrap<2>;
extern template class Fluid::MPI::SableWrap<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;
extern template class Solid::MPI::SharedLinearElasticity<2>;
extern template class Solid::MPI::SharedLinearElasticity<3>;

using namespace dealii;

int main(int argc, char *argv[])
{
  int *app_number;
  int flag;
  int global_sum;
  int mstring;
  MPI_Init(NULL, NULL);
  MultithreadInfo::set_thread_limit(1);

  std::string my_name("OpenIFEM");
  std::map<std::string,std::vector<int> > maps_of_other_apps;

  MPI_Comm_get_attr(MPI_COMM_WORLD,MPI_APPNUM,& app_number,&flag);
   
  MPI_Allreduce(app_number,&global_sum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  global_sum++; 
  if(global_sum != 1){

    MPI_Comm_split(MPI_COMM_WORLD,*app_number,0,  &PETSC_COMM_WORLD);

  }
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int my_world_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &my_world_size);

  int my_world_rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &my_world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Finalize the MPI environment.

  int num_apps = global_sum;

  /*build associative array based on app name that holds vectors of processors
   * in other app's communicators*/
  if (num_apps != 1)
    {

      int max_string_size = my_name.size() + 1;
      MPI_Allreduce(
        &max_string_size, &mstring, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      char **strings = new char *[num_apps];
      char **instrings = new char *[num_apps];

      for (int ict = 0; ict < num_apps; ict++)
        {
          strings[ict] = new char[mstring];
          strings[ict][0] = '\0';
          instrings[ict] = new char[mstring];
          instrings[ict][0] = '\0';
          memset(instrings[ict], 0, mstring * sizeof(char));
        }
      strncpy(instrings[*app_number], my_name.c_str(), my_name.size());

      for (int ict = 0; ict < num_apps; ict++)
        {
          MPI_Allreduce(instrings[ict],
                        strings[ict],
                        mstring,
                        MPI_CHAR,
                        MPI_MAX,
                        MPI_COMM_WORLD);
        }
      int *all_ranks_map_in = new int[world_size];
      int *all_ranks_map = new int[world_size];

      memset(all_ranks_map_in, 0, world_size * sizeof(int));
      all_ranks_map_in[world_rank] = *app_number;
      MPI_Allreduce(all_ranks_map_in,
                    all_ranks_map,
                    world_size,
                    MPI_INT,
                    MPI_MAX,
                    MPI_COMM_WORLD);

      for (int ict = 0; ict < num_apps; ict++)
        {
          std::vector<int> avec;
          for (int pct = 0; pct < world_size; pct++)
            {
              if (all_ranks_map[pct] == ict)
                {
                  avec.push_back(pct);
                }
            }
          maps_of_other_apps[std::string(strings[ict])] = avec;
        }
      for (int ict = 0; ict < num_apps; ict++)
        {
          delete[] instrings[ict];
          delete[] strings[ict];
        }
      delete[] instrings;
      delete[] strings;
    }

  if (my_world_rank == 0)
    {
      std::map<std::string, std::vector<int>>::iterator it;
      for (it = maps_of_other_apps.begin(); it != maps_of_other_apps.end();
           it++)
        {
          // std::cout  << my_name << " " << (*it).first << std::endl;
          std::vector<int>::iterator vit;
          for (vit = (*it).second.begin(); vit != (*it).second.end(); vit++)
            {
              // std::cout  << my_name << " " << "proc identity " << *vit <<
              // std::endl;
            }
        }
    }

  std::map<std::string, std::vector<int>>::iterator mit;
  mit = maps_of_other_apps.find(std::string("SABLE"));
  std::vector<int> sable_ids = mit->second;

  std::string infile("parameters.prm");
  if (argc > 1)
    {
      infile = argv[1];
    }
  // Create mesh
  Parameters::AllParameters params(infile);

  // Recieve Eulerian grid information from SABLE
  // NOTE: The code works for square grid only
  // receive coordinates for the lower left corner
  // initialize coordinates to a large -ve value
  double x = -1e10;
  double y = -1e10;
  double temp_x = x;
  double temp_y = y;
  MPI_Allreduce(&x, &temp_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&y, &temp_y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  x = temp_x;
  y = temp_y;
  // recieve grid size
  double dx = 0;
  double temp_dx = 0;
  MPI_Allreduce(&dx, &temp_dx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  dx = temp_dx;
  // receive no. of cells
  int nx = 0;
  int temp_nx = 0;
  MPI_Allreduce(&nx, &temp_nx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  nx = temp_nx;

  // Eulerian solid domain length
  // double Lf = 20.0;
  // grid size
  // double h = 0.2;
  if (params.dimension == 2)
    {
      // Create mesh for Eulerian solid
      Point<2> lower_corner(x, y);
      Point<2> upper_corner(x + dx * nx, y + dx * nx);
      parallel::distributed::Triangulation<2> eul_tria(PETSC_COMM_WORLD);
      dealii::GridGenerator::subdivided_hyper_rectangle(
        eul_tria,
        {static_cast<unsigned int>(nx), static_cast<unsigned int>(nx)},
        lower_corner,
        upper_corner,
        true);
      // Create Eulerian solid object with Sable wrapper
      Fluid::MPI::SableWrap<2> fluid(eul_tria, params, sable_ids);
      // Read mesh for Lagrangian solid
      GridIn<2> grid_in;
      Triangulation<2> lag_tria;
      std::ifstream input_solid("ogival_proj.abaqus");
      grid_in.attach_triangulation(lag_tria);
      grid_in.read_abaqus(input_solid);

      // scal or offset the Lagrangian mesh
      Tensor<1, 2> offset({0.0, 0.0});
      GridTools::shift(offset, lag_tria);
      double scale = 1.0;
      GridTools::scale(scale, lag_tria);

      // Create Lagrangian solid object with linear elastic material
      Solid::MPI::SharedLinearElasticity<2> solid(lag_tria, params);
      // Constrain three points in the Lagrangian solid in x direction
      Point<2> p1(4.21, -0.000811369);
      Point<2> p2(8.6804, -0.000454367);
      Point<2> p3(14.37, 0);
      std::vector<Point<2>> points = {p1, p2, p3};
      std::vector<unsigned int> directions = {1, 1, 1};
      solid.constrain_points(points, directions);
      // Create FSI object
      MPI::OpenIFEM_Sable_FSI<2> fsi(fluid, solid, params, false);
      fsi.run();
    }
  else
    {
      AssertThrow(false, ExcMessage("This test should be run in 2D!"));
    }

  // return 0;
  MPI_Barrier(MPI_COMM_WORLD);
  PetscFinalize();  
  MPI_Finalize();
  return 0;
}
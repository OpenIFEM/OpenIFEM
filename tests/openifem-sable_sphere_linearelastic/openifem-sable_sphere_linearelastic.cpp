// This program tests Lagrangian-Eulerian solid-solid interaction between
// OpenIFEM hyperleastic solid solver and Sable The generated executable needs
// to be run with Sable executable
#include "fsi.h"
#include "linear_elasticity.h"
#include "openifem_sable_fsi.h"
#include "parameters.h"
#include "sable_wrapper.h"
#include "utilities.h"

extern template class Fluid::SableWrap<2>;
extern template class Fluid::SableWrap<3>;
extern template class Utils::GridCreator<2>;
extern template class Utils::GridCreator<3>;

using namespace dealii;

int main(int argc, char *argv[])
{
  int *app_number;
  int flag;
  int global_sum;
  int mstring;
  MPI_Init(NULL, NULL);
  MPI_Comm new_comm;
  std::string my_name("OpenIFEM");
  std::map<std::string, std::vector<int>> maps_of_other_apps;

  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &app_number, &flag);

  new_comm = MPI_COMM_WORLD;
  MPI_Allreduce(app_number, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  global_sum++;
  if (global_sum != 1)
    {

      MPI_Comm_split(MPI_COMM_WORLD, *app_number, 0, &new_comm);
    }
  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int my_world_size;
  MPI_Comm_size(new_comm, &my_world_size);

  int my_world_rank;
  MPI_Comm_rank(new_comm, &my_world_rank);

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
  double z = -1e10;
  double temp_x = x;
  double temp_y = y;
  double temp_z = z;
  MPI_Allreduce(&x, &temp_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&y, &temp_y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&z, &temp_z, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  x = temp_x;
  y = temp_y;
  z = temp_z;
  // recieve grid size
  double h = 0;
  double temp_h = 0;
  MPI_Allreduce(&h, &temp_h, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  h = temp_h;
  // receive no. of cells
  int n = 0;
  int temp_n = 0;
  MPI_Allreduce(&n, &temp_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  n = temp_n;

  if (params.dimension == 3)
    {
      // Lagrangian sphere radius
      double R = 1.0;
      // Lagrangian sphere center
      Point<3> center(0, 0, 1);
      // Create mesh for Eulerian solid
      Point<3> lower_corner(x, y, z);
      Point<3> upper_corner(x + h * n, y + h * n, z + h * n);
      Triangulation<3> eul_tria;
      dealii::GridGenerator::subdivided_hyper_rectangle(
        eul_tria,
        {static_cast<unsigned int>(n),
         static_cast<unsigned int>(n),
         static_cast<unsigned int>(n)},
        lower_corner,
        upper_corner,
        true);
      // Create Eulerian solid object with Sable wrapper
      Fluid::SableWrap<3> fluid(eul_tria, params, sable_ids);
      // Create mesh for Lagrangian solid
      Triangulation<3> lag_tria;
      Utils::GridCreator<3>::sphere(lag_tria, center, R);
      // Create Lagrangian solid object with hyper elastic material
      Solid::LinearElasticity<3> solid(lag_tria, params);
      // Create FSI object
      OpenIFEM_Sable_FSI<3> fsi(fluid, solid, params, false);
      fsi.run();
    }
  else
    {
      AssertThrow(false, ExcMessage("This test should be run in 2D!"));
    }

  // return 0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
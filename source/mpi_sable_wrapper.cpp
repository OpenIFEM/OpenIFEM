#include "mpi_sable_wrapper.h"

namespace Fluid
{

  namespace MPI
  {

    template <int dim>
    SableWrap<dim>::SableWrap(parallel::distributed::Triangulation<dim> &tria,
                              const Parameters::AllParameters &parameters,
                              std::vector<int> &sable_ids,
                              MPI_Comm m)
      : FluidSolver<dim>(tria, parameters, m), sable_ids(sable_ids)
    {
      AssertThrow(
        parameters.fluid_velocity_degree == parameters.fluid_pressure_degree ==
          1,
        ExcMessage("Use 1st order elements for both pressure and velocity!"));
    }

    template <int dim>
    void SableWrap<dim>::initialize_system()
    {
      FluidSolver<dim>::initialize_system();
      fsi_force.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);
      fsi_velocity.reinit(
        owned_partitioning, relevant_partitioning, mpi_communicator);

      // Setup cell wise stress, vf, density received from SABLE
      int stress_size = (dim == 2 ? 3 : 6);
      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (!cell->is_artificial())
            {
              sable_cell_data.initialize(cell, 1);
              const std::vector<std::shared_ptr<SableCellData>> p =
                sable_cell_data.get_data(cell);
              p[0]->cell_stress.resize(stress_size, 0);
              p[0]->cell_stress_no_bgmat.resize(stress_size, 0);
              p[0]->material_vf = 0;
              p[0]->eulerian_density = 0;
              p[0]->modulus = 0;
            }
        }
    }

    template <int dim>
    void SableWrap<dim>::run()
    {

      setup_dofs();
      initialize_system();

      while (is_comm_active)
        {
          if (time.current() == 0)
            run_one_step();
          get_dt_sable();
          run_one_step();
        }
    }

    template <int dim>
    void SableWrap<dim>::run_one_step(bool apply_nonzero_constraints,
                                      bool assemble_system)
    {

      if (time.get_timestep() == 0)
        {

          sable_no_nodes_one_dir = 0;
          sable_no_ele = 0;
          sable_no_nodes = 0;
          Max(sable_no_nodes_one_dir);
          Max(sable_no_ele);
          Max(sable_no_nodes);

          find_ghost_nodes();
          create_dof_map();

          rec_stress(sable_no_ele);
          rec_velocity(sable_no_nodes);

          output_results(0);
        }
      else
        {
          // when simulation_type == FSI, send fsi force and indicator through
          // openifem-sable fsi solver otherwise send 0 force and indicator
          if (parameters.simulation_type != "FSI")
            {
              send_fsi_force(sable_no_nodes);
            }

          // Recieve no. of nodes and elements from Sable
          sable_no_nodes_one_dir = 0;
          sable_no_ele = 0;
          sable_no_nodes = 0;
          Max(sable_no_nodes_one_dir);
          Max(sable_no_ele);
          Max(sable_no_nodes);

          rec_stress(sable_no_ele);
          rec_velocity(sable_no_nodes);

          is_comm_active = All(is_comm_active);
          std::cout << std::string(96, '*') << std::endl
                    << "Received solution from Sable at time step = "
                    << time.get_timestep() << ", at t = " << std::scientific
                    << time.current() << std::endl;
          // Output
          if ((int(time.get_timestep()) % int(parameters.output_interval)) == 0)
            {
              output_results(time.get_timestep());
            }
        }
    }

    template <int dim>
    bool SableWrap<dim>::All(bool my_b)
    {
      int ib = (my_b == true ? 0 : 1);
      int result = 0;
      MPI_Allreduce(&ib, &result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      return (result == 0);
    }

    template <int dim>
    void SableWrap<dim>::get_dt_sable()
    {
      double dt = 0;
      Max(dt);
      time.set_delta_t(dt);
      time.increment();
    }

    template <int dim>
    void SableWrap<dim>::Max(int &send_buffer)
    {
      int temp;
      MPI_Allreduce(&send_buffer, &temp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      send_buffer = temp;
    }

    template <int dim>
    void SableWrap<dim>::Max(double &send_buffer)
    {
      double temp;
      MPI_Allreduce(
        &send_buffer, &temp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      send_buffer = temp;
    }

    template <int dim>
    void SableWrap<dim>::rec_data(double **rec_buffer,
                                  const std::vector<int> &cmapp,
                                  const std::vector<int> &cmapp_sizes,
                                  int data_size)
    {
      std::vector<MPI_Request> handles;
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          MPI_Request req;
          MPI_Irecv(rec_buffer[ict],
                    cmapp_sizes[ict],
                    MPI_DOUBLE,
                    cmapp[ict],
                    1,
                    MPI_COMM_WORLD,
                    &req);
          handles.push_back(req);
        }
      std::vector<MPI_Request>::iterator hit;
      for (hit = handles.begin(); hit != handles.end(); hit++)
        {
          MPI_Status stat;
          MPI_Wait(&(*hit), &stat);
        }
    }

    template <int dim>
    void SableWrap<dim>::send_data(double **send_buffer,
                                   const std::vector<int> &cmapp,
                                   const std::vector<int> &cmapp_sizes)
    {
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          int status = MPI_Send(send_buffer[ict],
                                cmapp_sizes[ict],
                                MPI_DOUBLE,
                                cmapp[ict],
                                1,
                                MPI_COMM_WORLD);
          assert(status == MPI_SUCCESS);
        }
    }

    template <int dim>
    void SableWrap<dim>::find_ghost_nodes()
    {

      int node_z =
        int(sable_no_nodes / (sable_no_nodes_one_dir * sable_no_nodes_one_dir));
      int node_z_begin = 0;
      int node_z_end = node_z;

      int sable_no_el_one_dir = (dim == 2 ? int(std::sqrt(sable_no_ele))
                                          : int(std::cbrt(sable_no_ele)));

      int ele_z =
        int(sable_no_ele / (sable_no_el_one_dir * sable_no_el_one_dir));
      int ele_z_begin = 0;
      int ele_z_end = ele_z;

      if (dim == 3)
        {
          node_z_begin = 1;
          node_z_end = node_z - 1;

          ele_z_begin = 1;
          ele_z_end = ele_z - 1;
        }

      for (int l = node_z_begin; l < node_z_end; l++)
        {
          int cornder_node_id =
            l * sable_no_nodes_one_dir * sable_no_nodes_one_dir +
            sable_no_nodes_one_dir + 1;
          for (int i = 0; i < sable_no_nodes_one_dir - 2; i++)
            {
              for (int j = 0; j < sable_no_nodes_one_dir - 2; j++)
                {
                  int n = cornder_node_id + j + i * (sable_no_nodes_one_dir);
                  non_ghost_nodes.push_back(n);
                }
            }
        }

      for (int l = ele_z_begin; l < ele_z_end; l++)
        {
          int cornder_el_id = l * sable_no_el_one_dir * sable_no_el_one_dir +
                              sable_no_el_one_dir + 1;
          for (int i = 0; i < sable_no_el_one_dir - 2; i++)
            {
              for (int j = 0; j < sable_no_el_one_dir - 2; j++)
                {
                  int n = cornder_el_id + j + i * (sable_no_el_one_dir);
                  non_ghost_cells.push_back(n);
                }
            }
        }

      assert(non_ghost_nodes.size() == triangulation.n_vertices());
      assert(non_ghost_cells.size() == triangulation.n_cells());
    }

    template <int dim>
    void SableWrap<dim>::create_dof_map()
    {
      // Create SABLE to OpenIFEM DOF map
      // Used when data is recieved from SABLE
      // -1: dof is not locally owned
      sable_openifem_dof_map.resize(dof_handler.n_dofs(), -1);

      // initialize OpenIFEM to SABLE dof map
      PETScWrappers::MPI::Vector tmp_dof_map;
      tmp_dof_map.reinit(owned_partitioning[0], mpi_communicator);

      std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   ++v)
                {
                  if (!vertex_touched[cell->vertex_index(v)])
                    {
                      vertex_touched[cell->vertex_index(v)] = true;
                      for (unsigned int i = 0; i < dim; i++)
                        {
                          // Sable vertex indexing is same as deal.ii
                          int sable_index = cell->vertex_index(v) * dim + i;
                          int openifem_index = cell->vertex_dof_index(v, i);
                          sable_openifem_dof_map[openifem_index] = sable_index;
                          tmp_dof_map[sable_index] = openifem_index;
                        }
                    }
                }
            }
        }

      tmp_dof_map.compress(VectorOperation::insert);
      openifem_sable_dof_map = tmp_dof_map;
    }

    template <int dim>
    void SableWrap<dim>::rec_velocity(const int &sable_n_nodes)
    {
      // Recieve solution
      int sable_sol_size = sable_n_nodes * dim;
      std::vector<int> cmapp = sable_ids;
      std::vector<int> cmapp_sizes(sable_ids.size(), sable_sol_size);

      // create rec buffer
      double **nv_rec_buffer = new double *[cmapp.size()];
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          nv_rec_buffer[ict] = new double[cmapp_sizes[ict]];
        }
      // recieve data
      rec_data(nv_rec_buffer, cmapp, cmapp_sizes, sable_sol_size);

      // remove solution from ghost layers of Sable mesh
      std::vector<double> sable_solution;
      PETScWrappers::MPI::BlockVector tmp;
      tmp.reinit(owned_partitioning, mpi_communicator);

      for (unsigned int n = 0; n < triangulation.n_vertices(); n++)
        {
          int non_ghost_node_id = non_ghost_nodes[n];
          int index = non_ghost_node_id * dim;
          for (unsigned int i = 0; i < dim; i++)
            {
              sable_solution.push_back(nv_rec_buffer[0][index + i]);
            }
        }

      for (unsigned int i = 0; i < sable_openifem_dof_map.size(); i++)
        {
          // skip dofs which are not locally owned
          if (sable_openifem_dof_map[i] != -1)
            tmp[i] = sable_solution[sable_openifem_dof_map[i]];
        }

      tmp.compress(VectorOperation::insert);
      present_solution = tmp;

      // delete solution
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          delete[] nv_rec_buffer[ict];
        }
      delete[] nv_rec_buffer;
    }

    template <int dim>
    void SableWrap<dim>::send_fsi_force(const int &sable_n_nodes)
    {

      Vector<double> localized_fsi_force(fsi_force.block(0));
      Vector<double> localized_fsi_velocity(fsi_velocity.block(0));

      int sable_force_size = sable_n_nodes * dim;
      std::vector<int> cmapp = sable_ids;
      std::vector<int> cmapp_sizes(cmapp.size(), sable_force_size);

      // create send buffer
      double **nv_send_buffer_force = new double *[cmapp.size()];
      double **nv_send_buffer_vel = new double *[cmapp.size()];

      for (unsigned int ict = 0; ict < cmapp.size(); ict++)
        {
          nv_send_buffer_force[ict] = new double[cmapp_sizes[ict]];
          nv_send_buffer_vel[ict] = new double[cmapp_sizes[ict]];
          for (int jct = 0; jct < cmapp_sizes[ict]; jct++)
            {
              nv_send_buffer_force[ict][jct] = 0;
              nv_send_buffer_vel[ict][jct] = 0;
            }
        }

      // create send buffer
      // add zero nodal forces corresponding to ghost nodes
      for (unsigned int n = 0; n < triangulation.n_vertices(); n++)
        {
          int non_ghost_node_id = non_ghost_nodes[n];
          int index = non_ghost_node_id * dim;
          for (unsigned int i = 0; i < dim; i++)
            {
              nv_send_buffer_force[0][index + i] =
                localized_fsi_force[openifem_sable_dof_map[n * dim + i]];
              nv_send_buffer_vel[0][index + i] =
                localized_fsi_velocity[openifem_sable_dof_map[n * dim + i]];
            }
        }
      // send fsi force
      send_data(nv_send_buffer_force, cmapp, cmapp_sizes);
      // send Dirichlet bc values for the artificial fluid
      send_data(nv_send_buffer_vel, cmapp, cmapp_sizes);

      // delete solution
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          delete[] nv_send_buffer_force[ict];
          delete[] nv_send_buffer_vel[ict];
        }
      delete[] nv_send_buffer_force;
      delete[] nv_send_buffer_vel;
    }

    template <int dim>
    void SableWrap<dim>::rec_stress(const int &sable_n_elements)
    {
      int sable_stress_size = sable_n_elements * dim * 2;
      int sable_stress_per_ele_size = dim * 2;
      std::vector<int> cmapp = sable_ids;
      std::vector<int> cmapp_sizes(cmapp.size(), sable_stress_size);
      
      // create rec buffer
      double **nv_rec_buffer_1 = new double *[cmapp.size()];
      double **nv_rec_buffer_2 = new double *[cmapp.size()];
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          nv_rec_buffer_1[ict] = new double[cmapp_sizes[ict]];
          nv_rec_buffer_2[ict] = new double[cmapp_sizes[ict]];
        }
      // recieve data
      // cell-wise stress for all background materials with VF average
      rec_data(nv_rec_buffer_1, cmapp, cmapp_sizes, sable_stress_size);
      // cell-wise stress for only selected background material without VF
      // average
      rec_data(nv_rec_buffer_2, cmapp, cmapp_sizes, sable_stress_size);

      // remove solution from ghost layers of Sable mesh
      std::vector<double> sable_stress;
      std::vector<double> sable_stress_no_bgmat;

      for (unsigned int n = 0; n < triangulation.n_cells(); n++)
        {
          int non_ghost_cell_id = non_ghost_cells[n];
          int index = non_ghost_cell_id * sable_stress_per_ele_size;
          for (int i = 0; i < sable_stress_per_ele_size; i++)
            {
              sable_stress.push_back(nv_rec_buffer_1[0][index + i]);
              sable_stress_no_bgmat.push_back(nv_rec_buffer_2[0][index + i]);
            }
        }

      assert(sable_stress.size() ==
             triangulation.n_cells() * sable_stress_per_ele_size);
      assert(sable_stress_no_bgmat.size() ==
             triangulation.n_cells() * sable_stress_per_ele_size);

      // Sable stress tensor in 2D: xx yy zz xy
      // Sable stress tensor in 3D: xx yy zz xy yz xz
      std::vector<int> stress_sequence;
      // create stress sequence according to dimension
      if (dim == 2)
        stress_sequence = {0, 3, 1};
      else
        stress_sequence = {0, 3, 1, 5, 4, 2};

      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              int count = 0;
              auto ptr = sable_cell_data.get_data(cell);
              for (auto j : stress_sequence)
                {
                  ptr[0]->cell_stress[count] =
                    sable_stress[cell->index() * sable_stress_per_ele_size + j];
                  ptr[0]->cell_stress_no_bgmat[count] =
                    sable_stress_no_bgmat[cell->index() *
                                            sable_stress_per_ele_size +
                                          j];
                  count = count + 1;
                }
            }
        }

      // delete buffer
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          delete[] nv_rec_buffer_1[ict];
          delete[] nv_rec_buffer_2[ict];
        }
      delete[] nv_rec_buffer_1;
      delete[] nv_rec_buffer_2;
    }

    template class SableWrap<2>;
    template class SableWrap<3>;
  } // namespace MPI
} // namespace Fluid

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
      (void)apply_nonzero_constraints;
      (void)assemble_system;

      std::cout.precision(6);
      std::cout.width(12);

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
          rec_vf(sable_no_ele);
          update_nodal_mass();
          rec_velocity(sable_no_nodes);

          output_results(0);
          std::cout << "Received inital solution from Sable" << std::endl;
        }
      else
        {
          // when simulation_type == FSI, send fsi force and indicator through
          // openifem-sable fsi solver otherwise send 0 force and indicator
          if (parameters.simulation_type != "FSI")
            {
              send_fsi_force(sable_no_nodes);
              send_indicator(sable_no_ele, sable_no_nodes);
            }

          // Recieve no. of nodes and elements from Sable
          sable_no_nodes_one_dir = 0;
          sable_no_ele = 0;
          sable_no_nodes = 0;
          Max(sable_no_nodes_one_dir);
          Max(sable_no_ele);
          Max(sable_no_nodes);

          rec_stress(sable_no_ele);
          rec_vf(sable_no_ele);
          update_nodal_mass();
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

    template <int dim>
    void SableWrap<dim>::rec_vf(const int &sable_n_elements)
    {

      std::vector<int> cmapp = sable_ids;
      std::vector<int> cmapp_sizes(sable_ids.size(), sable_n_elements);

      // create rec buffer
      double **nv_rec_buffer_1 = new double *[cmapp.size()];
      double **nv_rec_buffer_2 = new double *[cmapp.size()];
      double **nv_rec_buffer_3 = new double *[cmapp.size()];
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          nv_rec_buffer_1[ict] = new double[cmapp_sizes[ict]];
          nv_rec_buffer_2[ict] = new double[cmapp_sizes[ict]];
          nv_rec_buffer_3[ict] = new double[cmapp_sizes[ict]];
        }
      // recieve volume fraction for the material from SABLE
      rec_data(nv_rec_buffer_1, cmapp, cmapp_sizes, sable_n_elements);
      // recieve element density for the material from SABLE
      rec_data(nv_rec_buffer_2, cmapp, cmapp_sizes, sable_n_elements);
      // recieve element shear modulus for the material from SABLE
      rec_data(nv_rec_buffer_3, cmapp, cmapp_sizes, sable_n_elements);

      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              int non_ghost_cell_id = non_ghost_cells[cell->index()];
              auto ptr = sable_cell_data.get_data(cell);
              ptr[0]->material_vf = nv_rec_buffer_1[0][non_ghost_cell_id];
              ptr[0]->eulerian_density = nv_rec_buffer_2[0][non_ghost_cell_id];
              ptr[0]->modulus = nv_rec_buffer_3[0][non_ghost_cell_id];
            }
        }

      // delete buffer
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          delete[] nv_rec_buffer_1[ict];
          delete[] nv_rec_buffer_2[ict];
          delete[] nv_rec_buffer_3[ict];
        }
      delete[] nv_rec_buffer_1;
      delete[] nv_rec_buffer_2;
      delete[] nv_rec_buffer_3;
    }

    template <int dim>
    void SableWrap<dim>::send_indicator(const int &sable_n_elements,
                                        const int &sable_n_nodes)
    {
      std::vector<int> cmapp = sable_ids;
      std::vector<int> cmapp_sizes_element(sable_ids.size(), sable_n_elements);
      std::vector<int> cmapp_sizes_nodal(sable_ids.size(), sable_n_nodes);

      double *indicator_buffer = new double[sable_n_elements];
      double *exact_indicator_buffer = new double[sable_n_elements];
      double *lag_density_buffer = new double[sable_n_elements];
      double *nodal_indicator_buffer = new double[sable_n_nodes];

      for (int i = 0; i < sable_n_elements; i++)
        {
          indicator_buffer[i] = 0;
          exact_indicator_buffer[i] = 0;
          lag_density_buffer[i] = 0;
        }

      for (int i = 0; i < sable_n_nodes; i++)
        nodal_indicator_buffer[i] = 0;

      std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto ptr = cell_property.get_data(cell);
              auto s = sable_cell_data.get_data(cell);
              int non_ghost_cell_id =
                non_ghost_cells[cell->active_cell_index()];
              // get indicator
              indicator_buffer[non_ghost_cell_id] = ptr[0]->indicator;
              exact_indicator_buffer[non_ghost_cell_id] =
                ptr[0]->exact_indicator;
              // get nodal indicator flag
              if (ptr[0]->indicator != 0)
                {
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       ++v)
                    {
                      if (!vertex_touched[cell->vertex_index(v)])
                        {
                          vertex_touched[cell->vertex_index(v)] = true;
                          int non_ghost_node_id =
                            non_ghost_nodes[cell->vertex_index(v)];
                          nodal_indicator_buffer[non_ghost_node_id] = 1.0;
                        }
                    }
                }
              // get lagrangian density
              if (ptr[0]->exact_indicator != 0)
                {
                  lag_density_buffer[non_ghost_cell_id] =
                    ptr[0]->exact_indicator * parameters.solid_rho;

                  // change mass matrix for implicit Eulerian penalty
                  if (ptr[0]->exact_indicator == 1)
                    lag_density_buffer[non_ghost_cell_id] +=
                      parameters.solid_rho * parameters.penalty_scale_factor[1];
                }
            }
        }

      // create send buffer
      double **nv_send_buffer_ind = new double *[cmapp.size()];
      double **nv_send_buffer_exact_ind = new double *[cmapp.size()];
      double **nv_send_buffer_density = new double *[cmapp.size()];
      double **nv_send_buffer_nodal_ind = new double *[cmapp.size()];
      for (unsigned int ict = 0; ict < cmapp.size(); ict++)
        {
          nv_send_buffer_ind[ict] = new double[cmapp_sizes_element[ict]];
          nv_send_buffer_exact_ind[ict] = new double[cmapp_sizes_element[ict]];
          nv_send_buffer_density[ict] = new double[cmapp_sizes_element[ict]];
          nv_send_buffer_nodal_ind[ict] = new double[cmapp_sizes_nodal[ict]];
        }

      // get data from all the OpenIFEM processors
      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          MPI_Allreduce(indicator_buffer,
                        nv_send_buffer_ind[ict],
                        cmapp_sizes_element[ict],
                        MPI_DOUBLE,
                        MPI_MAX,
                        mpi_communicator);
          MPI_Allreduce(exact_indicator_buffer,
                        nv_send_buffer_exact_ind[ict],
                        cmapp_sizes_element[ict],
                        MPI_DOUBLE,
                        MPI_MAX,
                        mpi_communicator);
          MPI_Allreduce(lag_density_buffer,
                        nv_send_buffer_density[ict],
                        cmapp_sizes_element[ict],
                        MPI_DOUBLE,
                        MPI_MAX,
                        mpi_communicator);
          MPI_Allreduce(nodal_indicator_buffer,
                        nv_send_buffer_nodal_ind[ict],
                        cmapp_sizes_nodal[ict],
                        MPI_DOUBLE,
                        MPI_MAX,
                        mpi_communicator);
        }

      // send indicator field
      send_data(nv_send_buffer_ind, cmapp, cmapp_sizes_element);
      send_data(nv_send_buffer_exact_ind, cmapp, cmapp_sizes_element);
      // send modified Lagrangian density
      send_data(nv_send_buffer_density, cmapp, cmapp_sizes_element);
      // send nodal indicator field
      send_data(nv_send_buffer_nodal_ind, cmapp, cmapp_sizes_nodal);

      for (unsigned ict = 0; ict < cmapp.size(); ict++)
        {
          delete[] nv_send_buffer_ind[ict];
          delete[] nv_send_buffer_exact_ind[ict];
          delete[] nv_send_buffer_density[ict];
          delete[] nv_send_buffer_nodal_ind[ict];
        }

      delete[] indicator_buffer;
      delete[] nodal_indicator_buffer;
      delete[] nv_send_buffer_ind;
      delete[] nv_send_buffer_exact_ind;
      delete[] nv_send_buffer_density;
      delete[] nv_send_buffer_nodal_ind;
    }

    template <int dim>
    void SableWrap<dim>::update_nodal_mass()
    {
      nodal_mass.reinit(locally_owned_scalar_dofs, mpi_communicator);

      FEValues<dim> scalar_fe_values(scalar_fe,
                                     volume_quad_formula,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values);

      const unsigned int dofs_per_cell = scalar_fe.dofs_per_cell;
      const unsigned int n_q_points = volume_quad_formula.size();

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (auto cell = scalar_dof_handler.begin_active();
           cell != scalar_dof_handler.end();
           ++cell)
        {

          if (cell->is_locally_owned())
            {
              scalar_fe_values.reinit(cell);
              auto p = sable_cell_data.get_data(cell);
              auto eul_density = p[0]->eulerian_density;
              cell->get_dof_indices(local_dof_indices);

              local_matrix = 0;
              scalar_fe_values.reinit(cell);

              // Loop over quadrature points
              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  // Loop over the dofs again, to assemble
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          local_matrix[i][j] +=
                            eul_density * scalar_fe_values.shape_value(i, q) *
                            scalar_fe_values.shape_value(j, q) *
                            scalar_fe_values.JxW(q);
                        }
                    }
                }
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  double sum = 0;
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      sum = sum + local_matrix[i][j];
                    }
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      if (i == j)
                        {
                          local_matrix[i][j] = sum;
                        }
                      else
                        {
                          local_matrix[i][j] = 0;
                        }
                    }
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  auto index = local_dof_indices[i];
                  nodal_mass[index] += local_matrix[i][i];
                }
            }
        }

      nodal_mass.compress(VectorOperation::add);
    }

    template <int dim>
    void SableWrap<dim>::output_results(const unsigned int output_index) const
    {

      TimerOutput::Scope timer_section(timer, "Output results");

      pcout << "Writing results..." << std::endl;
      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.push_back("pressure");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      // vector to be output must be ghosted
      data_out.add_data_vector(present_solution,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      std::vector<std::string> fsi_force_names(dim, "fsi_force");
      fsi_force_names.push_back("dummy_fsi_force");

      data_out.add_data_vector(fsi_force,
                               fsi_force_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      // Partition
      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
          subdomain(i) = triangulation.locally_owned_subdomain();
        }
      data_out.add_data_vector(subdomain, "subdomain");

      // Indicator and cell-wise shear modulus, stress
      Vector<double> ind(triangulation.n_active_cells());
      Vector<double> exact_ind(triangulation.n_active_cells());
      Vector<double> shear_modulus(triangulation.n_active_cells());
      unsigned int stress_size = (dim == 2 ? 3 : 6);
      std::vector<Vector<double>> sable_stress;
      std::vector<Vector<double>> sable_stress_no_bgmat;
      sable_stress = std::vector<Vector<double>>(
        stress_size, Vector<double>(triangulation.n_cells()));
      sable_stress_no_bgmat = std::vector<Vector<double>>(
        stress_size, Vector<double>(triangulation.n_cells()));

      for (auto cell = triangulation.begin_active();
           cell != triangulation.end();
           ++cell)
        {
          if (cell->is_locally_owned())
            {
              auto p = cell_property.get_data(cell);
              auto c = sable_cell_data.get_data(cell);
              ind[cell->active_cell_index()] = p[0]->indicator;
              exact_ind[cell->active_cell_index()] = p[0]->exact_indicator;
              shear_modulus[cell->active_cell_index()] = c[0]->modulus;
              for (unsigned int j = 0; j < stress_size; j++)
                {
                  sable_stress[j][cell->active_cell_index()] =
                    c[0]->cell_stress[j];
                  sable_stress_no_bgmat[j][cell->active_cell_index()] =
                    c[0]->cell_stress_no_bgmat[j];
                }
            }
        }
      data_out.add_data_vector(ind, "Indicator");
      data_out.add_data_vector(exact_ind, "exact_indicator");
      data_out.add_data_vector(shear_modulus, "shear_modulus");
      data_out.add_data_vector(sable_stress[0], "cell_stress_xx");
      data_out.add_data_vector(sable_stress[1], "cell_stress_xy");
      data_out.add_data_vector(sable_stress[2], "cell_stress_yy");
      data_out.add_data_vector(sable_stress_no_bgmat[0],
                               "cell_stress_no_bgmat_xx");
      data_out.add_data_vector(sable_stress_no_bgmat[1],
                               "cell_stress_no_bgmat_xy");
      data_out.add_data_vector(sable_stress_no_bgmat[2],
                               "cell_stress_no_bgmat_yy");

      if (dim == 3)
        {
          data_out.add_data_vector(sable_stress[3], "cell_stress_xz");
          data_out.add_data_vector(sable_stress[4], "cell_stress_yz");
          data_out.add_data_vector(sable_stress[5], "cell_stress_zz");
          data_out.add_data_vector(sable_stress_no_bgmat[3],
                                   "cell_stress_no_bgmat_xz");
          data_out.add_data_vector(sable_stress_no_bgmat[4],
                                   "cell_stress_no_bgmat_yz");
          data_out.add_data_vector(sable_stress_no_bgmat[5],
                                   "cell_stress_no_bgmat_zz");
        }

      data_out.build_patches(parameters.fluid_pressure_degree);

      std::string basename =
        "fluid" + Utilities::int_to_string(output_index, 6) + "-";

      std::string filename =
        basename +
        Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
        ".vtu";

      std::ofstream output(filename);
      data_out.write_vtu(output);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          for (unsigned int i = 0;
               i < Utilities::MPI::n_mpi_processes(mpi_communicator);
               ++i)
            {
              times_and_names.push_back(
                {time.current(),
                 basename + Utilities::int_to_string(i, 4) + ".vtu"});
            }
          std::ofstream pvd_output("fluid.pvd");
          DataOutBase::write_pvd_record(pvd_output, times_and_names);
        }
    }

    template class SableWrap<2>;
    template class SableWrap<3>;
  } // namespace MPI
} // namespace Fluid

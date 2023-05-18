#include "sable_wrapper.h"

namespace Fluid
{

  template <int dim>
  SableWrap<dim>::SableWrap(Triangulation<dim> &tria,
                            const Parameters::AllParameters &parameters,
                            std::vector<int> &sable_ids,
                            std::shared_ptr<Function<dim>> bc)
    : FluidSolver<dim>(tria, parameters, bc),
      fe_vector_output(FE_Q<dim>(parameters.fluid_velocity_degree), dim),
      dof_handler_vector_output(triangulation),
      sable_ids(sable_ids)
  {
    AssertThrow(
      parameters.fluid_velocity_degree == parameters.fluid_pressure_degree == 1,
      ExcMessage("Use 1st order elements for both pressure and velocity!"));
  }

  template <int dim>
  void SableWrap<dim>::initialize_system()
  {
    FluidSolver<dim>::initialize_system();
    fsi_acceleration.reinit(dofs_per_block);
    fsi_velocity.reinit(dofs_per_block);
    fsi_vel_diff_eul.reinit(dofs_per_block);
    fsi_penalty_acceleration.reinit(dofs_per_block);
    fsi_penalty_force.reinit(dofs_per_block);
    int stress_vec_size = dim + dim * (dim - 1) * 0.5;
    fsi_stress = std::vector<Vector<double>>(
      stress_vec_size, Vector<double>(scalar_dof_handler.n_dofs()));

    // Setup and distribute dofs for the DofHandler which is only used for
    // outputting vector quantities
    dof_handler_vector_output.distribute_dofs(fe_vector_output);
    DoFRenumbering::Cuthill_McKee(dof_handler_vector_output);

    // Setup cell wise stress
    int stress_size = (dim == 2 ? 3 : 6);
    cell_wise_stress.initialize(
      triangulation.begin_active(), triangulation.end(), 1);
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<CellStress>> p =
          cell_wise_stress.get_data(cell);
        p[0]->cell_stress.resize(stress_size, 0);
        p[0]->cell_stress_no_bgmat.resize(stress_size, 0);
        p[0]->material_vf = 0;
        p[0]->eulerian_density = 0;
        p[0]->modulus = 0;
      }
  }

  template <int dim>
  void SableWrap<dim>::assemble_force()
  {
    TimerOutput::Scope timer_section(timer, "Assemble force");

    Tensor<1, dim> gravity;
    for (unsigned int i = 0; i < dim; ++i)
      gravity[i] = parameters.gravity[i];

    system_rhs = 0;
    fsi_force = 0;
    fsi_force_acceleration_part = 0;
    fsi_force_stress_part = 0;
    fsi_penalty_force = 0;

    FEValues<dim> fe_values(fe,
                            volume_quad_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEValues<dim> scalar_fe_values(scalar_fe,
                                   volume_quad_formula,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quad_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int u_dofs = fe.base_element(0).dofs_per_cell;
    const unsigned int p_dofs = fe.base_element(1).dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();

    AssertThrow(u_dofs * dim + p_dofs == dofs_per_cell,
                ExcMessage("Wrong partitioning of dofs!"));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    Vector<double> local_rhs(dofs_per_cell);
    Vector<double> local_rhs_acceleration_part(dofs_per_cell);
    Vector<double> local_rhs_stress_part(dofs_per_cell);
    Vector<double> local_penalty_force(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);
    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<Tensor<1, dim>> fsi_acc_values(n_q_points);
    std::vector<double> fsi_stress_value(n_q_points);
    std::vector<std::vector<double>> fsi_cell_stress =
      std::vector<std::vector<double>>(fsi_stress.size(),
                                       std::vector<double>(n_q_points));
    std::vector<Tensor<1, dim>> fsi_penalty_values(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    auto scalar_cell = scalar_dof_handler.begin_active();
    for (auto cell = dof_handler.begin_active();
         scalar_cell != scalar_dof_handler.end(), cell != dof_handler.end();
         ++cell, ++scalar_cell)
      {
        auto p = cell_property.get_data(cell);
        const double ind = p[0]->indicator;
        const double ind_exact = p[0]->exact_indicator;
        auto s = cell_wise_stress.get_data(cell);
        /*const double rho_bar =
          parameters.solid_rho * ind + s[0]->eulerian_density * (1 - ind);*/
        const double rho_bar = parameters.solid_rho * ind_exact +
                               s[0]->eulerian_density * (1 - ind_exact);

        fe_values.reinit(cell);
        scalar_fe_values.reinit(scalar_cell);

        local_rhs = 0;
        local_rhs_acceleration_part = 0;
        local_rhs_stress_part = 0;
        local_penalty_force = 0;

        SymmetricTensor<2, dim> sable_cell_stress;
        if (ind != 0)
          {
            // Get cell-wise SABLE stress
            int count = 0;
            for (unsigned int k = 0; k < dim; k++)
              {
                for (unsigned int m = 0; m < k + 1; m++)
                  {
                    sable_cell_stress[k][m] = s[0]->cell_stress[count];
                    count++;
                  }
              }

            fe_values[velocities].get_function_values(present_solution,
                                                      current_velocity_values);

            fe_values[velocities].get_function_gradients(
              present_solution, current_velocity_gradients);

            fe_values[pressure].get_function_values(present_solution,
                                                    current_pressure_values);

            fe_values[velocities].get_function_values(present_solution,
                                                      present_velocity_values);

            fe_values[velocities].get_function_values(fsi_acceleration,
                                                      fsi_acc_values);

            fe_values[velocities].get_function_values(fsi_penalty_acceleration,
                                                      fsi_penalty_values);

            for (unsigned int i = 0; i < fsi_stress.size(); i++)
              {
                scalar_fe_values.get_function_values(fsi_stress[i],
                                                     fsi_stress_value);
                fsi_cell_stress[i] = fsi_stress_value;
              }
          }

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            SymmetricTensor<2, dim> fsi_stress_tensor;
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k] = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k] = fe_values[velocities].value(k, q);
                phi_p[k] = fe_values[pressure].value(k, q);
              }

            if (ind != 0)
              {
                int stress_index = 0;
                for (unsigned int k = 0; k < dim; k++)
                  {
                    for (unsigned int m = 0; m < k + 1; m++)
                      {
                        fsi_stress_tensor[k][m] =
                          fsi_cell_stress[stress_index][q];
                        stress_index++;
                      }
                  }
                // When cell-wise SABLE stress is used
                if (parameters.fsi_force_calculation_option == "CellBased")
                  fsi_stress_tensor += sable_cell_stress;
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (ind != 0)
                  {
                    local_rhs(i) +=
                      (scalar_product(grad_phi_u[i], fsi_stress_tensor) +
                       rho_bar * fsi_acc_values[q] * phi_u[i]) *
                      fe_values.JxW(q);
                    local_rhs_acceleration_part(i) +=
                      (rho_bar * fsi_acc_values[q] * phi_u[i]) *
                      fe_values.JxW(q);
                    local_rhs_stress_part(i) +=
                      (scalar_product(grad_phi_u[i], fsi_stress_tensor)) *
                      fe_values.JxW(q);
                    local_penalty_force(i) +=
                      (rho_bar * fsi_penalty_values[q] * phi_u[i]) *
                      fe_values.JxW(q) / time.get_delta_t();
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          {
            system_rhs[local_dof_indices[i]] += local_rhs(i);
            fsi_force[local_dof_indices[i]] += local_rhs(i);
            fsi_force_acceleration_part[local_dof_indices[i]] +=
              local_rhs_acceleration_part(i);
            fsi_force_stress_part[local_dof_indices[i]] +=
              local_rhs_stress_part(i);
            fsi_penalty_force[local_dof_indices[i]] += local_penalty_force(i);
          }
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

        rec_stress(sable_no_ele);
        rec_vf(sable_no_ele);
        update_nodal_mass();
        rec_velocity(sable_no_nodes);
        output_results(0);
        std::cout << "Received inital solution from Sable" << std::endl;
        // All(active);
      }
    else
      {
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
        check_no_slip_bc();
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
  void SableWrap<dim>::run()
  {
    triangulation.refine_global(parameters.global_refinements[0]);
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
        MPI_Send(send_buffer[ict],
                 cmapp_sizes[ict],
                 MPI_DOUBLE,
                 cmapp[ict],
                 1,
                 MPI_COMM_WORLD);
      }
  }

  template <int dim>
  void SableWrap<dim>::find_ghost_nodes()
  {

    int node_z =
      int(sable_no_nodes / (sable_no_nodes_one_dir * sable_no_nodes_one_dir));
    int node_z_begin = 0;
    int node_z_end = node_z;

    int sable_no_el_one_dir =
      (dim == 2 ? int(std::sqrt(sable_no_ele)) : int(std::cbrt(sable_no_ele)));

    int ele_z = int(sable_no_ele / (sable_no_el_one_dir * sable_no_el_one_dir));
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
  void SableWrap<dim>::rec_velocity(const int &sable_n_nodes)
  {
    // Recieve solution
    int sable_sol_size = sable_n_nodes * dim;
    unsigned int vel_size = triangulation.n_vertices() * dim;
    std::vector<int> cmapp = sable_ids;
    std::vector<int> cmapp_sizes;
    cmapp_sizes.push_back(sable_sol_size);
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
    for (unsigned int n = 0; n < triangulation.n_vertices(); n++)
      {
        int non_ghost_node_id = non_ghost_nodes[n];
        int index = non_ghost_node_id * dim;
        for (unsigned int i = 0; i < dim; i++)
          {
            sable_solution.push_back(nv_rec_buffer[0][index + i]);
          }
      }

    assert(sable_solution.size() == vel_size);

    // synchronize solution
    present_solution.reinit(dofs_per_block);

    // Syncronize Sable and OpenIFEM solution
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                for (unsigned int i = 0; i < dim; i++)
                  {
                    // Sable vertex indexing is same as deal.ii
                    int sable_sol_index = cell->vertex_index(v) * dim + i;
                    int openifem_sol_index = cell->vertex_dof_index(v, i);
                    present_solution[openifem_sol_index] =
                      sable_solution[sable_sol_index];
                  }
              }
          }
      }

    // delete solution
    for (unsigned ict = 0; ict < cmapp.size(); ict++)
      {
        delete[] nv_rec_buffer[ict];
      }
    delete[] nv_rec_buffer;
  }

  template <int dim>
  void SableWrap<dim>::rec_stress(const int &sable_n_elements)
  {
    // initialize stress vector
    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
          {
            stress[i][j] = 0.0;
          }
      }

    int sable_stress_size = sable_n_elements * dim * 2;
    int sable_stress_per_ele_size = dim * 2;
    std::vector<int> cmapp = sable_ids;
    std::vector<int> cmapp_sizes;
    cmapp_sizes.push_back(sable_stress_size);

    int openifem_stress_per_ele_size = (dim == 2 ? 3 : 6);

    int openifem_stress_size =
      triangulation.n_cells() * openifem_stress_per_ele_size;

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
    // cell-wise stress for only selected background material without VF average
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

    std::vector<double> openifem_stress(openifem_stress_size, 0);

    // Sable stress tensor in 2D: xx yy zz xy
    // Sable stress tensor in 3D: xx yy zz xy yz xz
    std::vector<int> stress_sequence;
    // create stress sequence according to dimension
    if (dim == 2)
      stress_sequence = {0, 3, 1};
    else
      stress_sequence = {0, 3, 1, 5, 4, 2};

    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        int count = 0;
        auto ptr = cell_wise_stress.get_data(cell);
        for (auto j : stress_sequence)
          {
            openifem_stress[cell->index() * openifem_stress_per_ele_size +
                            count] =
              sable_stress[cell->index() * sable_stress_per_ele_size + j];
            ptr[0]->cell_stress[count] =
              sable_stress[cell->index() * sable_stress_per_ele_size + j];
            ptr[0]->cell_stress_no_bgmat[count] =
              sable_stress_no_bgmat[cell->index() * sable_stress_per_ele_size +
                                    j];
            count = count + 1;
          }
      }

    // syncronize solution
    auto cell = dof_handler.begin_active();
    auto scalar_cell = scalar_dof_handler.begin_active();
    std::vector<types::global_dof_index> dof_indices(scalar_fe.dofs_per_cell);
    std::vector<int> surrounding_cells(scalar_dof_handler.n_dofs(), 0);
    for (; cell != dof_handler.end(); ++cell, ++scalar_cell)
      {
        scalar_cell->get_dof_indices(dof_indices);
        int index = cell->active_cell_index() * openifem_stress_per_ele_size;
        int count = 0;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j <= i; ++j)
              {
                for (unsigned int k = 0; k < scalar_fe.dofs_per_cell; ++k)
                  {
                    stress[i][j][dof_indices[k]] +=
                      openifem_stress[index + count];
                    if (i == 0 && j == 0)
                      surrounding_cells[dof_indices[k]]++;
                  }
                count++;
              }
          }
      }

    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j <= i; ++j)
          {
            for (unsigned int k = 0; k < scalar_dof_handler.n_dofs(); ++k)
              {
                stress[i][j][k] /= surrounding_cells[k];
              }
          }
      }

    for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j <= i; ++j)
          {
            for (unsigned int k = 0; k < scalar_dof_handler.n_dofs(); ++k)
              {
                stress[j][i][k] = stress[i][j][k];
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

    int sable_vf_size = sable_n_elements;
    std::vector<int> cmapp = sable_ids;
    std::vector<int> cmapp_sizes;
    cmapp_sizes.push_back(sable_vf_size);

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
    rec_data(nv_rec_buffer_1, cmapp, cmapp_sizes, sable_vf_size);
    // recieve element density for the material from SABLE
    rec_data(nv_rec_buffer_2, cmapp, cmapp_sizes, sable_vf_size);
    // recieve element shear modulus for the material from SABLE
    rec_data(nv_rec_buffer_3, cmapp, cmapp_sizes, sable_vf_size);

    // remove solution from ghost layers of Sable mesh
    std::vector<double> vf_vector;
    std::vector<double> density;
    std::vector<double> modulus;

    for (unsigned int n = 0; n < triangulation.n_cells(); n++)
      {
        int index = non_ghost_cells[n];
        vf_vector.push_back(nv_rec_buffer_1[0][index]);
        density.push_back(nv_rec_buffer_2[0][index]);
        modulus.push_back(nv_rec_buffer_3[0][index]);
      }

    assert(vf_vector.size() == triangulation.n_cells());
    assert(density.size() == triangulation.n_cells());
    assert(modulus.size() == triangulation.n_cells());

    int count = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        auto ptr = cell_wise_stress.get_data(cell);
        ptr[0]->material_vf = vf_vector[count];
        ptr[0]->eulerian_density = density[count];
        ptr[0]->modulus = modulus[count];
        count += 1;
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
  void SableWrap<dim>::send_fsi_force(const int &sable_n_nodes)
  {
    // For quadrature points based indicator and FSI force calculation forces
    // are assembled in find_fluid_bc_qpoints For nodes based indicator and FSI
    // foce calculation, assemble FSI forces here
    if (parameters.fsi_force_criteria == "Nodes")
      assemble_force();

    int sable_force_size = sable_n_nodes * dim;
    std::vector<int> cmapp = sable_ids;
    std::vector<int> cmapp_sizes;
    cmapp_sizes.push_back(sable_force_size);

    // Syncronize Sable and OpenIFEM solution
    std::vector<double> sable_fsi_force(triangulation.n_vertices() * dim, 0);
    std::vector<double> sable_fsi_velocity(triangulation.n_vertices() * dim, 0);
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                for (unsigned int i = 0; i < dim; i++)
                  {
                    // Sable vertex indexing is same as deal.ii
                    int sable_force_index = cell->vertex_index(v) * dim + i;
                    int openifem_force_index = cell->vertex_dof_index(v, i);
                    sable_fsi_force[sable_force_index] =
                      fsi_force[openifem_force_index];
                    sable_fsi_velocity[sable_force_index] =
                      fsi_velocity[openifem_force_index];
                  }
              }
          }
      }

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

    // add zero nodal forces corresponding to ghost nodes
    for (unsigned int n = 0; n < triangulation.n_vertices(); n++)
      {
        int non_ghost_node_id = non_ghost_nodes[n];
        int index = non_ghost_node_id * dim;
        for (unsigned int i = 0; i < dim; i++)
          {
            nv_send_buffer_force[0][index + i] = sable_fsi_force[n * dim + i];
            nv_send_buffer_vel[0][index + i] = sable_fsi_velocity[n * dim + i];
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
  void SableWrap<dim>::send_indicator(const int &sable_n_elements,
                                      const int &sable_n_nodes)
  {

    int sable_indicator_field_size = sable_n_elements;
    std::vector<int> cmapp = sable_ids;
    std::vector<int> cmapp_sizes_element;
    cmapp_sizes_element.push_back(sable_indicator_field_size);
    std::vector<int> cmapp_sizes_nodal;
    cmapp_sizes_nodal.push_back(sable_n_nodes);

    // create vector of indicator field
    std::vector<double> indicator_field(triangulation.n_cells(), 0);
    std::vector<double> indicator_field_exact(triangulation.n_cells(), 0);
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    // create vector of nodal indicator flags
    std::vector<double> nodal_indicator_field(triangulation.n_vertices(), 0);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        auto ptr = cell_property.get_data(cell);
        auto s = cell_wise_stress.get_data(cell);

        indicator_field[cell->active_cell_index()] = ptr[0]->indicator;
        indicator_field_exact[cell->active_cell_index()] =
          ptr[0]->exact_indicator;
        if (ptr[0]->indicator != 0)
          {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                if (!vertex_touched[cell->vertex_index(v)])
                  {
                    vertex_touched[cell->vertex_index(v)] = true;
                    nodal_indicator_field[cell->vertex_index(v)] = 1.0;
                  }
              }
          }
      }

    std::vector<double> sable_indicator_field(sable_indicator_field_size, 0);
    std::vector<double> sable_indicator_field_exact(sable_indicator_field_size,
                                                    0);
    std::vector<double> sable_lag_density(sable_indicator_field_size, 0);

    for (unsigned int n = 0; n < triangulation.n_cells(); n++)
      {
        int non_ghost_cell_id = non_ghost_cells[n];
        sable_indicator_field[non_ghost_cell_id] = indicator_field[n];
        sable_indicator_field_exact[non_ghost_cell_id] =
          indicator_field_exact[n];
        /*if (indicator_field[n] != 0)
          sable_lag_density[non_ghost_cell_id] =
            indicator_field[n] * parameters.solid_rho;*/
        if (indicator_field_exact[n] != 0)
          {
            sable_lag_density[non_ghost_cell_id] =
              indicator_field_exact[n] * parameters.solid_rho;

            // change mass matrix for implicit Eulerian penalty
            if (indicator_field[n] == 1)
              sable_lag_density[non_ghost_cell_id] +=
                parameters.solid_rho * parameters.penalty_scale_factor[1];
          }
      }

    // create send buffer
    double **nv_send_buffer = new double *[cmapp.size()];
    double **nv_send_buffer_exact = new double *[cmapp.size()];
    double **nv_send_buffer_density = new double *[cmapp.size()];
    for (unsigned int ict = 0; ict < cmapp.size(); ict++)
      {
        nv_send_buffer[ict] = new double[cmapp_sizes_element[ict]];
        nv_send_buffer_exact[ict] = new double[cmapp_sizes_element[ict]];
        nv_send_buffer_density[ict] = new double[cmapp_sizes_element[ict]];
        for (int jct = 0; jct < cmapp_sizes_element[ict]; jct++)
          {
            nv_send_buffer[ict][jct] = sable_indicator_field[jct];
            nv_send_buffer_exact[ict][jct] = sable_indicator_field_exact[jct];
            nv_send_buffer_density[ict][jct] = sable_lag_density[jct];
          }
      }

    // send indicator field
    send_data(nv_send_buffer, cmapp, cmapp_sizes_element);
    send_data(nv_send_buffer_exact, cmapp, cmapp_sizes_element);
    // send modified Lagrangian density
    send_data(nv_send_buffer_density, cmapp, cmapp_sizes_element);
    for (unsigned ict = 0; ict < cmapp.size(); ict++)
      {
        delete[] nv_send_buffer[ict];
        delete[] nv_send_buffer_exact[ict];
        delete[] nv_send_buffer_density[ict];
      }
    // create send buffer
    nv_send_buffer = new double *[cmapp.size()];
    for (unsigned int ict = 0; ict < cmapp.size(); ict++)
      {
        nv_send_buffer[ict] = new double[cmapp_sizes_nodal[ict]];
        for (int jct = 0; jct < cmapp_sizes_nodal[ict]; jct++)
          {
            nv_send_buffer[ict][jct] = 0;
          }
      }

    // add zero nodal indicator corresponding to ghost nodes
    for (unsigned int n = 0; n < triangulation.n_vertices(); n++)
      {
        int non_ghost_node_id = non_ghost_nodes[n];
        nv_send_buffer[0][non_ghost_node_id] = nodal_indicator_field[n];
      }

    // send data
    send_data(nv_send_buffer, cmapp, cmapp_sizes_nodal);

    // delete solution
    for (unsigned ict = 0; ict < cmapp.size(); ict++)
      {
        delete[] nv_send_buffer[ict];
      }
    delete[] nv_send_buffer;
    delete[] nv_send_buffer_exact;
    delete[] nv_send_buffer_density;
  }

  template <int dim>
  void SableWrap<dim>::update_nodal_mass()
  {

    nodal_mass.reinit(scalar_dof_handler.n_dofs());

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
        scalar_fe_values.reinit(cell);
        auto p = cell_wise_stress.get_data(cell);
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
                    local_matrix[i][j] += eul_density *
                                          scalar_fe_values.shape_value(i, q) *
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

  template <int dim>
  void SableWrap<dim>::check_no_slip_bc()
  {
    // initialize blockvector
    fsi_vel_diff_eul.reinit(dofs_per_block);
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        auto ptr = cell_property.get_data(cell);
        if (ptr[0]->indicator != 0)
          {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                for (unsigned int i = 0; i < dim; i++)
                  {
                    int index = cell->vertex_dof_index(v, i);
                    fsi_vel_diff_eul[index] =
                      fsi_velocity[index] - present_solution[index];
                  }
              }
          }
      }
  }

  template <int dim>
  void SableWrap<dim>::output_results(const unsigned int output_index) const
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    std::cout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    // fsi force output
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_force(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    Vector<double> fsi_force_output;
    fsi_force_output.reinit(dofs_per_block[0]);
    fsi_force_output = fsi_force.block(0);
    solution_names = std::vector<std::string>(dim, "fsi_force");
    data_out.add_data_vector(dof_handler_vector_output,
                             fsi_force_output,
                             solution_names,
                             data_component_interpretation_force);

    Vector<double> fsi_acceleration_output;
    fsi_acceleration_output.reinit(dofs_per_block[0]);
    fsi_acceleration_output = fsi_force_acceleration_part.block(0);
    solution_names = std::vector<std::string>(dim, "fsi_acceleration");
    data_out.add_data_vector(dof_handler_vector_output,
                             fsi_acceleration_output,
                             solution_names,
                             data_component_interpretation_force);

    Vector<double> fsi_stress_output;
    fsi_stress_output.reinit(dofs_per_block[0]);
    fsi_stress_output = fsi_force_stress_part.block(0);
    solution_names = std::vector<std::string>(dim, "fsi_stress");
    data_out.add_data_vector(dof_handler_vector_output,
                             fsi_stress_output,
                             solution_names,
                             data_component_interpretation_force);

    // output penalty force
    Vector<double> fsi_penalty_force_output;
    fsi_penalty_force_output.reinit(dofs_per_block[0]);
    fsi_penalty_force_output = fsi_penalty_force.block(0);
    solution_names = std::vector<std::string>(dim, "fsi_penalty_force");
    data_out.add_data_vector(dof_handler_vector_output,
                             fsi_penalty_force_output,
                             solution_names,
                             data_component_interpretation_force);

    // output difference between solid and artificial fluid velocities (no-slip
    // bc)
    Vector<double> fsi_velocity_output;
    fsi_velocity_output.reinit(dofs_per_block[0]);
    fsi_velocity_output = fsi_vel_diff_eul.block(0);
    solution_names = std::vector<std::string>(dim, "fsi_velocity_difference");
    data_out.add_data_vector(dof_handler_vector_output,
                             fsi_velocity_output,
                             solution_names,
                             data_component_interpretation_force);

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

    int i = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        auto p = cell_property.get_data(cell);
        auto c = cell_wise_stress.get_data(cell);
        ind[i] = p[0]->indicator;
        exact_ind[i] = p[0]->exact_indicator;
        shear_modulus[i] = c[0]->modulus;
        for (unsigned int j = 0; j < stress_size; j++)
          {
            sable_stress[j][i] = c[0]->cell_stress[j];
            sable_stress_no_bgmat[j][i] = c[0]->cell_stress_no_bgmat[j];
          }
        i++;
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

    // viscous stress
    data_out.add_data_vector(scalar_dof_handler, stress[0][0], "Txx");
    data_out.add_data_vector(scalar_dof_handler, stress[0][1], "Txy");
    data_out.add_data_vector(scalar_dof_handler, stress[1][1], "Tyy");
    if (dim == 3)
      {
        data_out.add_data_vector(scalar_dof_handler, stress[0][2], "Txz");
        data_out.add_data_vector(scalar_dof_handler, stress[1][2], "Tyz");
        data_out.add_data_vector(scalar_dof_handler, stress[2][2], "Tzz");

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

    std::string basename = "fluid";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
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
    double dt = std::numeric_limits<double>::max();
    Min(dt);
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
    MPI_Allreduce(&send_buffer, &temp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    send_buffer = temp;
  }

  template <int dim>
  void SableWrap<dim>::Min(int &send_buffer)
  {
    int temp;
    MPI_Allreduce(&send_buffer, &temp, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    send_buffer = temp;
  }

  template <int dim>
  void SableWrap<dim>::Min(double &send_buffer)
  {
    double temp;
    MPI_Allreduce(&send_buffer, &temp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    send_buffer = temp;
  }

  template class SableWrap<2>;
  template class SableWrap<3>;
} // namespace Fluid

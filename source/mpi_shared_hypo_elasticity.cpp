#include "mpi_shared_hypo_elasticity.h"

namespace
{
  std::vector<unsigned int> dirichlet_boundary_x, dirichlet_boundary_y;
  std::vector<unsigned int> neumann_boundary_x, neumann_boundary_y;
  template <typename U>
  static void dirichlet_boundary_function_x(U *p)
  {
    p->x = p->X;
    p->vx = 0.;
    p->previous_vx = 0.;
    p->ax = 0.;
    p->vx_t = 0.;
  }

  template <typename U>
  static void dirichlet_boundary_function_y(U *p)
  {
    p->y = p->Y;
    p->vy = 0.;
    p->previous_vy = 0.;
    p->ay = 0.;
    p->vy_t = 0.;
  }

  template <typename U>
  std::function<void(U *)> neumann_boundary_function(double tx, double ty)
  {
    std::function<void(U *)> f = [=](U *p) {
      p->tx = tx;
      p->ty = ty;
    };
    return f;
  }
} // namespace

namespace Solid
{
  namespace MPI
  {
    using namespace dealii;

    template <int dim>
    SharedHypoElasticity<dim>::SharedHypoElasticity(
      Triangulation<dim> &tria,
      const Parameters::AllParameters &params,
      double dx,
      double hdx)
      : SharedSolidSolver<dim>(tria, params), dx(dx), hdx(hdx)
    {
    }

    template <int dim>
    void SharedHypoElasticity<dim>::run_one_step(bool first_step)
    {
      // Not parallelized yet, only rank 0 does work!
      if (first_step)
        {
          if (this_mpi_process == 0)
            {
              construct_particles();
              vtk_writer_write(m_body, time.get_timestep());
            }
          this->output_results(time.get_timestep());
        }
      time.increment();
      pcout << std::endl
            << "Timestep " << time.get_timestep() << " @ " << time.current()
            << "s" << std::endl;
      if (this_mpi_process == 0)
        {
          m_body.step();
          m_body.update_boundary();
          synchronize();
        }
      // distribute
      Vector<double> displacement(dof_handler.n_dofs());
      Vector<double> velocity(dof_handler.n_dofs());
      Utilities::MPI::sum(
        serialized_displacement, mpi_communicator, displacement);
      Utilities::MPI::sum(serialized_velocity, mpi_communicator, velocity);
      current_displacement = displacement;
      current_velocity = velocity;
      if (time.time_to_output())
        {
          this->output_results(time.get_timestep());
          if (this_mpi_process == 0)
            {
              vtk_writer_write(m_body, time.get_timestep());
            }
        }
      if (parameters.simulation_type == "Solid" && time.time_to_save())
        {
          this->save_checkpoint(time.get_timestep());
        }
    }

    template <int dim>
    void SharedHypoElasticity<dim>::initialize_system()
    {
      SharedSolidSolver<dim>::initialize_system();
      serialized_displacement = Vector<double>(dof_handler.n_dofs());
      serialized_velocity = Vector<double>(dof_handler.n_dofs());
    }

    template <int dim>
    void SharedHypoElasticity<dim>::assemble_system(bool initial_step)
    {
      // do nothing
      (void)initial_step;
    }

    template <int dim>
    void SharedHypoElasticity<dim>::update_strain_and_stress()
    {
      // do nothing
    }

    template <int dim>
    void SharedHypoElasticity<dim>::synchronize()
    {
      std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
      unsigned int n_face_q_points = face_quad_formula.size();
      unsigned int face_quad_point_id = 0;
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              if (!vertex_touched[cell->vertex_index(v)])
                {
                  vertex_touched[cell->vertex_index(v)] = true;
                  int id = vertex_mapping[cell->vertex_index(v)];
                  double ux = m_body.get_cur_particles()[id]->x -
                              m_body.get_cur_particles()[id]->X;
                  double uy = m_body.get_cur_particles()[id]->y -
                              m_body.get_cur_particles()[id]->Y;
                  double vx = m_body.get_cur_particles()[id]->vx;
                  double vy = m_body.get_cur_particles()[id]->vy;
                  serialized_displacement(cell->vertex_dof_index(v, 0)) = ux;
                  serialized_displacement(cell->vertex_dof_index(v, 1)) = uy;
                  serialized_velocity(cell->vertex_dof_index(v, 0)) = vx;
                  serialized_velocity(cell->vertex_dof_index(v, 1)) = vy;
                }
            }
          auto ptr = cell_property.get_data(cell);
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      auto traction =
                        ptr[f * n_face_q_points + q]->fsi_traction;
                      m_body.get_face_quad_points()[face_quad_point_id]->tx =
                        traction[0];
                      m_body.get_face_quad_points()[face_quad_point_id]->ty =
                        traction[1];
                      face_quad_point_id++;
                    }
                }
            }
        }
    }

    template <int dim>
    void SharedHypoElasticity<dim>::construct_particles()
    {
      FEValues<dim> fe_values(
        fe, volume_quad_formula, update_quadrature_points | update_JxW_values);
      unsigned int n_q_points = volume_quad_formula.size();
      // Particles
      unsigned int n_particles = triangulation.n_vertices();
      particle_tl_weak **particles = new particle_tl_weak *[n_particles];
      unsigned int particle_id = 0;
      vertex_mapping = std::vector<int>(triangulation.n_vertices(), -1);
      // Volume quadrature points, assuming 2nd order integration
      unsigned int n_vol_quad = 4 * triangulation.n_active_cells();
      particle_tl_weak **vol_quad_points = new particle_tl_weak *[n_vol_quad];
      unsigned int vol_quad_point_id = 0;
      // Face quadrature points, assuming 2nd order integration
      unsigned int n_face_q_points = face_quad_formula.size();
      unsigned int n_face_quad = 0;
      // Boundary conditions
      boundary_conditions<particle_tl_weak> bc;
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          // Vertex
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              if (vertex_mapping[cell->vertex_index(v)] == -1)
                {
                  particles[particle_id] = new particle_tl_weak(particle_id);
                  particles[particle_id]->X = cell->vertex(v)[0];
                  particles[particle_id]->Y = cell->vertex(v)[1];
                  particles[particle_id]->x = cell->vertex(v)[0];
                  particles[particle_id]->y = cell->vertex(v)[1];
                  particles[particle_id]->rho = parameters.solid_rho;
                  particles[particle_id]->h = hdx * dx;
                  particles[particle_id]->m =
                    cell->measure() * parameters.solid_rho;
                  particles[particle_id]->quad_weight =
                    particles[particle_id]->m / particles[particle_id]->rho;
                  vertex_mapping[cell->vertex_index(v)] = particle_id++;
                }
            }
          // Volume quad point
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              Point<dim> q_point = fe_values.quadrature_point(q);
              vol_quad_points[vol_quad_point_id] =
                new particle_tl_weak(vol_quad_point_id);
              vol_quad_points[vol_quad_point_id]->X = q_point[0];
              vol_quad_points[vol_quad_point_id]->Y = q_point[1];
              vol_quad_points[vol_quad_point_id]->x = q_point[0];
              vol_quad_points[vol_quad_point_id]->y = q_point[1];
              vol_quad_points[vol_quad_point_id]->quad_weight =
                fe_values.JxW(q);
              vol_quad_points[vol_quad_point_id]->h = hdx * dx;
              vol_quad_points[vol_quad_point_id]->rho = parameters.solid_rho;
              vol_quad_point_id++;
            }
          // Count face quad points first
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  n_face_quad += n_face_q_points;
                }
            }
        }
      AssertThrow(triangulation.n_vertices() == particle_id,
                  ExcMessage("Vertices do not match!"));
      AssertThrow(n_vol_quad == vol_quad_point_id,
                  ExcMessage("Volume quadrature points do not match!"));
      particle_tl_weak **face_quad_points = new particle_tl_weak *[n_face_quad];
      FEFaceValues<dim> fe_face_values(
        fe, face_quad_formula, update_quadrature_points | update_JxW_values);
      unsigned int face_quad_point_id = 0;
      for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
           ++cell)
        {
          // Then set face quad points
          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  std::vector<unsigned int> dirichlet_ids;
                  std::vector<unsigned int> neumann_ids;
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_face;
                       ++v)
                    {
                      dirichlet_ids.push_back(
                        vertex_mapping[cell->face(f)->vertex_index(v)]);
                    }
                  fe_face_values.reinit(cell, f);
                  for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      Point<dim> q_point = fe_face_values.quadrature_point(q);
                      face_quad_points[face_quad_point_id] =
                        new particle_tl_weak(face_quad_point_id);
                      face_quad_points[face_quad_point_id]->X = q_point[0];
                      face_quad_points[face_quad_point_id]->Y = q_point[1];
                      face_quad_points[face_quad_point_id]->x = q_point[0];
                      face_quad_points[face_quad_point_id]->y = q_point[1];
                      face_quad_points[face_quad_point_id]->quad_weight =
                        fe_face_values.JxW(q);
                      face_quad_points[face_quad_point_id]->h = hdx * dx;
                      face_quad_points[face_quad_point_id]->rho =
                        parameters.solid_rho;
                      neumann_ids.push_back(face_quad_point_id);
                      face_quad_point_id++;
                    }
                  // bc
                  auto ptr1 = parameters.solid_dirichlet_bcs.find(
                    cell->face(f)->boundary_id());
                  if (ptr1 != parameters.solid_dirichlet_bcs.end())
                    {
                      auto flag = ptr1->second;
                      if (flag == 1 || flag == 3 || flag == 5 || flag == 7)
                        {
                          bc.add_boundary_condition(
                            dirichlet_ids, &dirichlet_boundary_function_x);
                        }
                      if (flag == 2 || flag == 3 || flag == 6 || flag == 7)
                        {
                          bc.add_boundary_condition(
                            dirichlet_ids, &dirichlet_boundary_function_y);
                        }
                    }
                  auto ptr2 = parameters.solid_neumann_bcs.find(
                    cell->face(f)->boundary_id());
                  if (ptr2 != parameters.solid_neumann_bcs.end() &&
                      parameters.simulation_type != "FSI")
                    {
                      auto traction = ptr2->second;
                      auto f = neumann_boundary_function<particle_tl_weak>(
                        traction[0], traction[1]);
                      bc.add_neumann_boundary_condition(neumann_ids, f);
                    }
                }
            }
        }
      AssertThrow(n_face_quad == face_quad_point_id,
                  ExcMessage("Face quadrature points do not match!"));

      find_neighbors(particles, n_particles, hdx * dx, distance_euclidian);
      find_neighbors(particles,
                     n_particles,
                     vol_quad_points,
                     n_vol_quad,
                     hdx * dx,
                     distance_euclidian);
      find_neighbors(particles,
                     n_particles,
                     face_quad_points,
                     n_face_quad,
                     hdx * dx,
                     distance_euclidian);
      precomp_rkpm<particle_tl_weak>(particles, vol_quad_points, n_vol_quad);
      precomp_rkpm<particle_tl_weak>(particles, face_quad_points, n_face_quad);
      precomp_rkpm<particle_tl_weak>(particles, n_particles);
      physical_constants physical_constants(
        parameters.nu[0], parameters.E[0], parameters.solid_rho);
      simulation_data sim_data(
        physical_constants,
        correction_constants(
          constants_monaghan(), constants_artificial_viscosity(), 0, true));
      m_body = body<particle_tl_weak>(particles,
                                      n_particles,
                                      sim_data,
                                      parameters.time_step,
                                      bc,
                                      vol_quad_points,
                                      n_vol_quad,
                                      0,
                                      face_quad_points,
                                      n_face_quad,
                                      parameters.damping);
      m_body.add_action(&derive_quad_coordinates);
      m_body.add_action(&derive_displacement_weak);
      m_body.add_action(&derive_face_quad_coordinates);
      m_body.add_action(&derive_face_displacement);
      m_body.add_action(&derive_velocity_tl_weak);
      m_body.add_action(&contmech_velocity_gradient_tl_gp);
      m_body.add_action(&contmech_continuity_gp);
      m_body.add_action(&material_eos_gp);
      m_body.add_action(&material_stress_rate_jaumann_gp);
      m_body.add_action(&contmech_cauchy_to_nominal_gp);
      m_body.add_action(&derive_stress_weak_tl);
      m_body.add_action(&derive_particle_stress);
      m_body.add_action(&derive_traction_weak_tl);
      m_body.add_action(&contmech_momentum_tl_weak);
      m_body.add_action(&contmech_advection);
    }

    template class SharedHypoElasticity<2>;
  } // namespace MPI
} // namespace Solid

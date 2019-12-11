#include "hypo_elasticity.h"
#include <iostream>

namespace
{
  std::vector<std::vector<unsigned int>> dirichlet_boundaries;
  std::vector<std::vector<unsigned int>> neumann_boundaries;

  template <int dim>
  std::function<void(particle<dim> *)>
  dirichlet_boundary_function(unsigned int n)
  {
    std::function<void(particle<dim> *)> f = [=](particle<dim> *p) {
      p->x[n] = p->X[n];
      p->v[n] = 0.;
      p->previous_v[n] = 0.;
      p->a[n] = 0.;
      p->v_t[n] = 0.;
    };
    return f;
  }

  template <int dim>
  std::function<void(particle<dim> *)>
  neumann_boundary_function(std::vector<double> v)
  {
    std::function<void(particle<dim> *)> f = [=](particle<dim> *p) {
      for (unsigned int n = 0; n < dim; ++n)
        p->t[n] = v[n];
    };
    return f;
  }
} // namespace

namespace Solid
{
  using namespace dealii;

  template <int dim>
  HypoElasticity<dim>::HypoElasticity(Triangulation<dim> &tria,
                                      const Parameters::AllParameters &params,
                                      double dx,
                                      double hdx)
    : SolidSolver<dim>(tria, params), dx(dx), hdx(hdx)
  {
  }

  template <int dim>
  void HypoElasticity<dim>::run_one_step(bool first_step)
  {
    if (first_step)
      {
        construct_particles();
        this->output_results(time.get_timestep());
        utilities<dim>::vtk_write_particle(m_body->get_particles(),
                                           m_body->get_num_part(),
                                           time.get_timestep(),
                                           "particles");
      }
    time.increment();
    std::cout << std::endl
              << "Timestep " << time.get_timestep() << " @ " << time.current()
              << "s" << std::endl;
    m_body->step();
    synchronize();
    if (time.time_to_output())
      {
        this->output_results(time.get_timestep());
        utilities<dim>::vtk_write_particle(m_body->get_particles(),
                                           m_body->get_num_part(),
                                           time.get_timestep(),
                                           "particles");
      }
  }

  template <int dim>
  void HypoElasticity<dim>::initialize_system()
  {
    SolidSolver<dim>::initialize_system();
  }

  template <int dim>
  void HypoElasticity<dim>::assemble_system(bool initial_step)
  {
    // do nothing
    (void)initial_step;
  }

  template <int dim>
  void HypoElasticity<dim>::update_strain_and_stress()
  {
    // do nothing
  }

  template <int dim>
  void HypoElasticity<dim>::synchronize()
  {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    unsigned int n_face_q_points = face_quad_formula.size();
    unsigned int face_quad_point_id = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                int id = vertex_mapping[cell->vertex_index(v)];
                auto disp = m_body->get_particles()[id]->x -
                            m_body->get_particles()[id]->X;
                auto vel = m_body->get_particles()[id]->v;
                auto acc = m_body->get_particles()[id]->a;
                for (unsigned int n = 0; n < dim; ++n)
                  {
                    current_displacement(cell->vertex_dof_index(v, n)) =
                      disp[n];
                    current_velocity(cell->vertex_dof_index(v, n)) = vel[n];
                    current_acceleration(cell->vertex_dof_index(v, n)) = acc[n];
                  }
              }
          }
        auto ptr = cell_property.get_data(cell);
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                for (unsigned int q = 0; q < n_face_q_points; ++q)
                  {
                    auto traction = ptr[f]->fsi_traction;
                    for (unsigned int n = 0; n < dim; ++n)
                      {
                        m_body->get_face_quad_points()[face_quad_point_id]
                          ->t[n] = traction[n];
                      }
                    face_quad_point_id++;
                  }
              }
          }
      }
  }

  template <int dim>
  void HypoElasticity<dim>::construct_particles()
  {
    FEValues<dim> fe_values(
      fe, volume_quad_formula, update_quadrature_points | update_JxW_values);
    unsigned int n_q_points = volume_quad_formula.size();
    // Particles
    unsigned int n_particles = triangulation.n_vertices();
    particle<dim> **particles = new particle<dim> *[n_particles];
    unsigned int particle_id = 0;
    vertex_mapping = std::vector<int>(triangulation.n_vertices(), -1);
    // Volume quadrature points, assuming 2nd order integration
    unsigned int n_vol_quad =
      volume_quad_formula.size() * triangulation.n_active_cells();
    particle<dim> **vol_quad_points = new particle<dim> *[n_vol_quad];
    unsigned int vol_quad_point_id = 0;
    // Face quadrature points, assuming 2nd order integration
    unsigned int n_face_q_points = face_quad_formula.size();
    unsigned int n_face_quad = 0;
    // Boundary conditions
    boundary_conditions<dim> bc;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        // Vertex
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if (vertex_mapping[cell->vertex_index(v)] == -1)
              {
                particles[particle_id] = new particle<dim>(particle_id);
                for (unsigned int n = 0; n < dim; ++n)
                  {
                    particles[particle_id]->X[n] = cell->vertex(v)[n];
                    particles[particle_id]->x[n] = cell->vertex(v)[n];
                  }
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
              new particle<dim>(vol_quad_point_id);
            for (unsigned int n = 0; n < dim; ++n)
              {
                vol_quad_points[vol_quad_point_id]->X[n] = q_point[n];
                vol_quad_points[vol_quad_point_id]->x[n] = q_point[n];
              }
            vol_quad_points[vol_quad_point_id]->quad_weight = fe_values.JxW(q);
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
    particle<dim> **face_quad_points = new particle<dim> *[n_face_quad];
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
                      new particle<dim>(face_quad_point_id);
                    for (unsigned int n = 0; n < dim; ++n)
                      {
                        face_quad_points[face_quad_point_id]->X[n] = q_point[n];
                        face_quad_points[face_quad_point_id]->x[n] = q_point[n];
                      }
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
                        auto f = dirichlet_boundary_function<dim>(0);
                        bc.add_boundary_condition(dirichlet_ids, f);
                      }
                    if (flag == 2 || flag == 3 || flag == 6 || flag == 7)
                      {
                        auto f = dirichlet_boundary_function<dim>(1);
                        bc.add_boundary_condition(dirichlet_ids, f);
                      }
                    if (flag == 4 || flag == 5 || flag == 6 || flag == 7)
                      {
                        auto f = dirichlet_boundary_function<dim>(2);
                        bc.add_boundary_condition(dirichlet_ids, f);
                      }
                  }
                auto ptr2 = parameters.solid_neumann_bcs.find(
                  cell->face(f)->boundary_id());
                if (ptr2 != parameters.solid_neumann_bcs.end() &&
                    parameters.simulation_type != "FSI")
                  {
                    auto f = neumann_boundary_function<dim>(ptr2->second);
                    bc.add_neumann_boundary_condition(neumann_ids, f);
                  }
              }
          }
      }
    AssertThrow(n_face_quad == face_quad_point_id,
                ExcMessage("Face quadrature points do not match!"));

    utilities<dim>::find_neighbors(particles, n_particles, hdx * dx);
    utilities<dim>::find_neighbors(
      particles, n_particles, vol_quad_points, n_vol_quad, hdx * dx);
    utilities<dim>::find_neighbors(
      particles, n_particles, face_quad_points, n_face_quad, hdx * dx);
    utilities<dim>::precomp_rkpm(particles, vol_quad_points, n_vol_quad);
    utilities<dim>::precomp_rkpm(particles, face_quad_points, n_face_quad);
    utilities<dim>::precomp_rkpm(particles, n_particles);
    physical_constants physical_constants(
      parameters.nu[0], parameters.E[0], parameters.solid_rho);
    simulation_data sim_data(
      physical_constants,
      correction_constants(
        constants_monaghan(), constants_artificial_viscosity(), 0, true));
    m_body = std::make_unique<body<dim>>(particles,
                                         n_particles,
                                         sim_data,
                                         parameters.time_step,
                                         bc,
                                         vol_quad_points,
                                         n_vol_quad,
                                         face_quad_points,
                                         n_face_quad,
                                         parameters.damping);
  }

  template class HypoElasticity<2>;
  template class HypoElasticity<3>;
} // namespace Solid

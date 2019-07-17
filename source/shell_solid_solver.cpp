#include "shell_solid_solver.h"
#include <deal.II/grid/grid_out.h>
#include <fstream>

namespace Solid
{
  using namespace dealii;

  ShellSolidSolver::ShellSolidSolver(Triangulation<2, 3> &tria,
                                     const Parameters::AllParameters &params,
                                     libMesh::LibMeshInit *libmesh_init)
    : SolidSolver<2, 3>(tria, params),
      libmesh_init(libmesh_init),
      m_mesh(libmesh_init->comm(), 2)
  {
    shell_params.debug = false;
    shell_params.nu = parameters.nu[0];
    shell_params.em = parameters.E[0];
    shell_params.thickness = 0.1;
    shell_params.isOutfileSet = false;
  }

  void ShellSolidSolver::get_forcing_file(const std::string &file)
  {
    this->shell_params.force_filename = file;
  }

  void ShellSolidSolver::run()
  {
    setup_dofs();
    initialize_system();
    m_mesh.write("out.msh");
    this->m_shell->run();
    this->m_shell->writeOutput();
  }

  void ShellSolidSolver::initialize_system() {}

  void ShellSolidSolver::setup_dofs()
  {
    SolidSolver<2, 3>::setup_dofs();
    construct_mesh();
    // Construct BC map for internal solver
    std::map<libMesh::boundary_id_type, unsigned int> dirichlet_bcs;
    for (const auto &bc : parameters.solid_dirichlet_bcs)
      {
        dirichlet_bcs[static_cast<libMesh::boundary_id_type>(bc.first)] =
          bc.second;
      }
    this->m_shell->make_constraints(dirichlet_bcs);
  }

  void ShellSolidSolver::update_strain_and_stress() {}

  void ShellSolidSolver::assemble_system(bool initial_step)
  {
    // Do nothing. We don't assemble in the wrapper
    (void)initial_step;
  }

  void ShellSolidSolver::run_one_step(bool first_step) { (void)first_step; }

  void ShellSolidSolver::synchronize() {}

  void ShellSolidSolver::construct_mesh()
  {
    m_mesh.allow_renumbering(false);
    // Add nodes
    auto vertices = this->triangulation.get_vertices();
    m_mesh.reserve_nodes(static_cast<unsigned int>(vertices.size()));
    for (int i = 0; i < static_cast<int>(vertices.size()); ++i)
      {
        auto &v = vertices[i];
        libMesh::Real x(v(0)), y(v(1)), z(v(2));
        m_mesh.add_point(libMesh::Point(x, y, z), i);
      }
    // Add elements
    m_mesh.reserve_elem(triangulation.n_active_cells());
    for (auto cell = triangulation.begin_active(); cell != triangulation.end();
         ++cell)
      {
        libMesh::Elem *elem = libMesh::Elem::build(libMesh::QUAD4).release();
        elem->set_id(cell->index());
        elem->subdomain_id() =
          static_cast<libMesh::subdomain_id_type>(cell->material_id());
        elem = m_mesh.add_elem(elem);
        // Vertices order in dealii is different from libMesh
        std::vector<unsigned int> vertices{0, 1, 3, 2};
        unsigned int v_lm = 0;
        for (auto &&v : vertices)
          {
            elem->set_node(v_lm++) = m_mesh.node_ptr(cell->vertex_index(v));
          }
        // Add boundary ids
        std::vector<unsigned int> sides{3, 1, 0, 2};
        if (cell->at_boundary())
          {
            for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
              {
                if (cell->face(f)->at_boundary())
                  {
                    m_mesh.get_boundary_info().add_side(
                      elem, sides[f], cell->face(f)->boundary_id());
                  }
              }
          }
      }
    m_mesh.set_mesh_dimension(2);
    m_mesh.set_spatial_dimension(3);
    m_mesh.prepare_for_use();
    // Create internal shell solid solver
    this->m_shell =
      std::make_unique<ShellSolid::shellsolid>(m_mesh, this->shell_params);
  }
} // namespace Solid
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
    this->m_shell->run();
    grab_solution();
    grab_stress();
    this->m_shell->writeOutput();
    output_results(0);
  }

  void ShellSolidSolver::initialize_system()
  {
    current_displacement.reinit(dof_handler.n_dofs());
    current_drilling.reinit(dof_handler.n_dofs());
    strain = std::vector<std::vector<Vector<double>>>(
      3,
      std::vector<Vector<double>>(3,
                                  Vector<double>(scalar_dof_handler.n_dofs())));
    stress = std::vector<std::vector<Vector<double>>>(
      3,
      std::vector<Vector<double>>(3,
                                  Vector<double>(scalar_dof_handler.n_dofs())));

    // Set up cell property, which contains the FSI traction required in FSI
    // simulation
    cell_property.initialize(triangulation.begin_active(),
                             triangulation.end(),
                             GeometryInfo<2>::faces_per_cell);
  }

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

  void ShellSolidSolver::synchronize()
  {
    grab_solution();
    grab_stress();
  }

  void ShellSolidSolver::grab_solution()
  {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    const std::vector<libMesh::Number> &solution(m_shell->get_solution());
    AssertThrow(solution.size() == current_displacement.size() * 5,
                ExcMessage("Inconsistent solution size!"));
    // Copy the solutions
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                for (unsigned int n : {0, 1, 2})
                  {
                    current_displacement(cell->vertex_dof_index(v, n)) =
                      solution[15 * cell->vertex_index(v) + n];
                    current_drilling(cell->vertex_dof_index(v, n)) =
                      solution[15 * cell->vertex_index(v) + 3 + n];
                  }
              }
          }
      }
  }

  void ShellSolidSolver::push_solution()
  {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    std::vector<libMesh::Number> solution(current_displacement.size() * 2);
    // Copy the solutions
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                for (unsigned int n : {0, 1, 2})
                  {
                    solution[6 * cell->vertex_index(v) + n] =
                      current_displacement(cell->vertex_dof_index(v, n));
                  }
              }
          }
      }
    this->m_shell->set_solution(solution);
  }

  void ShellSolidSolver::grab_stress()
  {
    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    const std::vector<libMesh::Number> &solution(m_shell->get_solution());
    AssertThrow(solution.size() == current_displacement.size() * 5,
                ExcMessage("Inconsistent solution size!"));
    // Copy the solutions
    for (auto cell = scalar_dof_handler.begin_active();
         cell != scalar_dof_handler.end();
         ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
          {
            if (!vertex_touched[cell->vertex_index(v)])
              {
                vertex_touched[cell->vertex_index(v)] = true;
                for (unsigned int i : {0, 1, 2})
                  {
                    for (unsigned int j : {0, 1, 2})
                      stress[i][j](cell->vertex_dof_index(v, 0)) =
                        solution[15 * cell->vertex_index(v) + 3 * i + j];
                  }
              }
          }
      }
  }

  void ShellSolidSolver::output_results(const unsigned int output_index)
  {
    TimerOutput::Scope timer_section(timer, "Output results");

    std::vector<std::string> solution_names(3, "displacements");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        3, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<2, DoFHandler<2, 3>> data_out;
    data_out.attach_dof_handler(dof_handler);

    // displacements
    data_out.add_data_vector(dof_handler,
                             current_displacement,
                             solution_names,
                             data_component_interpretation);
    // velocity
    solution_names = std::vector<std::string>(3, "drillings");
    data_out.add_data_vector(dof_handler,
                             current_drilling,
                             solution_names,
                             data_component_interpretation);

    // strain and stress
    data_out.add_data_vector(scalar_dof_handler, stress[0][0], "Sxx");
    data_out.add_data_vector(scalar_dof_handler, stress[0][1], "Sxy");
    data_out.add_data_vector(scalar_dof_handler, stress[1][1], "Syy");
    data_out.add_data_vector(scalar_dof_handler, stress[0][2], "Sxz");
    data_out.add_data_vector(scalar_dof_handler, stress[1][2], "Syz");
    data_out.add_data_vector(scalar_dof_handler, stress[2][2], "Szz");

    data_out.build_patches();

    std::string basename = "solid";
    std::string filename =
      basename + "-" + Utilities::int_to_string(output_index, 6) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.push_back({time.current(), filename});
    std::ofstream pvd_output(basename + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
} // namespace Solid
#include "utilities.h"
#include <bitset>

namespace Utils
{
  bool Time::time_to_output() const
  {
    auto delta = static_cast<unsigned int>(output_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

  bool Time::time_to_refine() const
  {
    auto delta = static_cast<unsigned int>(refinement_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

  bool Time::time_to_save() const
  {
    auto delta = static_cast<unsigned int>(save_interval / delta_t);
    return (timestep >= delta && timestep % delta == 0);
  }

  void Time::increment()
  {
    time_current += delta_t;
    ++timestep;
  }

  void Time::decrement()
  {
    time_current -= delta_t;
    --timestep;
  }

  void Time::set_delta_t(double delta) { delta_t = delta; }

  template <int dim, typename VectorType>
  SPHInterpolator<dim, VectorType>::SPHInterpolator(
    const DoFHandler<dim> &dof_handler, const Point<dim> &point)
    : dof_handler(dof_handler), target(point)
  {
    for (auto cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;
        Point<dim> center = cell->center();
        double h = cell->diameter();
        double kernel_value = cubic_spline(center, target, h);
        if (kernel_value > 1e-12)
          {
            sources.push_back({cell, kernel_value});
          }
      }
  }

  template <int dim, typename VectorType>
  double SPHInterpolator<dim, VectorType>::cubic_spline(const Point<dim> &pi,
                                                        const Point<dim> &pj,
                                                        double h)
  {
    double w(0.);
    double q = pi.distance(pj) / h;
    // M_1_PI = 1 / pi
    double coef = (dim == 2 ? 10 * M_1_PI / (7 * h * h) : M_1_PI / (h * h * h));
    if (q < 2.)
      {
        if (q >= 1.)
          {
            w = coef * (0.25 * (2 - q) * (2 - q) * (2 - q));
          }
        else
          {
            w = coef * (1 - 1.5 * q * q + 0.75 * q * q * q);
          }
      }
    return w;
  }

  template <int dim, typename VectorType>
  void SPHInterpolator<dim, VectorType>::point_value(
    const VectorType &fe_function,
    Vector<typename VectorType::value_type> &value)
  {
    typedef typename VectorType::value_type Number;
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(value.size() == fe.n_components(),
           ExcDimensionMismatch(value.size(), fe.n_components()));

    // Cell center in unit coordinate system
    Point<dim> unit_center;
    for (unsigned int i = 0; i < dim; ++i)
      unit_center[i] = 0.5;
    Quadrature<dim> quad(unit_center);
    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping, fe, quad, update_values);
    value = 0;
    for (auto p : sources)
      {
        auto cell = p.first;
        fe_values.reinit(cell);
        std::vector<Vector<Number>> u_value(1,
                                            Vector<Number>(fe.n_components()));
        fe_values.get_function_values(fe_function, u_value);
        // In SPH interpolation, volume must be multiplied because kernel
        // function has a unit of LENGTH^{-dim}
        u_value[0] *= p.second * cell->measure();
        value += u_value[0];
      }
  }

  template <int dim, typename VectorType>
  void SPHInterpolator<dim, VectorType>::point_gradient(
    const VectorType &fe_function,
    std::vector<Tensor<1, dim, typename VectorType::value_type>> &gradient)
  {
    typedef typename VectorType::value_type Number;
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(gradient.size() == fe.n_components(),
           ExcDimensionMismatch(gradient.size(), fe.n_components()));
    // Cell center in unit coordinate system
    Point<dim> unit_center;
    for (unsigned int i = 0; i < dim; ++i)
      unit_center[i] = 0.5;
    Quadrature<dim> quad(unit_center);
    MappingQGeneric<dim> mapping(1);
    FEValues<dim> fe_values(mapping, fe, quad, update_gradients);
    for (unsigned int i = 0; i < gradient.size(); ++i)
      gradient[i] = 0;
    for (auto p : sources)
      {
        auto cell = p.first;
        fe_values.reinit(cell);
        std::vector<std::vector<Tensor<1, dim, Number>>> u_gradient(
          1, std::vector<Tensor<1, dim, Number>>(fe.n_components()));
        fe_values.get_function_gradients(fe_function, u_gradient);
        for (unsigned int i = 0; i < gradient.size(); ++i)
          {
            // In SPH interpolation, volume must be multiplied because kernel
            // function has a unit of LENGTH^{-dim}
            u_gradient[0][i] *= p.second * cell->measure();
            gradient[i] += u_gradient[0][i];
          }
      }
  }

  template <int dim, typename VectorType>
  GridInterpolator<dim, VectorType>::GridInterpolator(
    const DoFHandler<dim> &dof_handler,
    const Point<dim> &point,
    const std::vector<bool> &mask,
    const typename DoFHandler<dim>::active_cell_iterator &cell)
    : dof_handler(dof_handler), point(point), cell_found(true)
  {
    // If the cell is valid we just use the cell
    if (cell.state() == IteratorState::IteratorStates::valid)
      {
        cell_point.first = cell;
        cell_point.second = mapping.transform_real_to_unit_cell(cell, point);
        return;
      }
    // This function throws an exception of GridTools::ExcPointNotFound if the
    // point
    // does not lie in any cell. In this case, we set the cell pointer to null.
    try
      {
        cell_point = GridTools::find_active_cell_around_point(
          mapping, dof_handler, point, mask);
      }
    catch (GridTools::ExcPointNotFound<dim> &e)
      {
        cell_point.first = dof_handler.end();
        cell_point.second = point;
        cell_found = false;
      }
  }

  template <int dim, typename VectorType>
  void GridInterpolator<dim, VectorType>::point_value(
    const VectorType &fe_function,
    Vector<typename VectorType::value_type> &value)
  {
    typedef typename VectorType::value_type Number;
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(value.size() == fe.n_components(),
           ExcDimensionMismatch(value.size(), fe.n_components()));
    // If for some reason, the point is not found in any cell,
    // or it is on a cell that is not locally owned, return 0.
    if (cell_point.first == dof_handler.end() ||
        !cell_point.first->is_locally_owned())
      {
        value = 0;
        return;
      }
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,
           ExcInternalError());

    const Quadrature<dim> quadrature(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);
    std::vector<Vector<Number>> u_value(1, Vector<Number>(fe.n_components()));
    fe_values.get_function_values(fe_function, u_value);
    value = u_value[0];
  }

  template <int dim, typename VectorType>
  void GridInterpolator<dim, VectorType>::point_gradient(
    const VectorType &fe_function,
    std::vector<Tensor<1, dim, typename VectorType::value_type>> &gradient)
  {
    typedef typename VectorType::value_type Number;
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    Assert(gradient.size() == fe.n_components(),
           ExcDimensionMismatch(gradient.size(), fe.n_components()));
    // If for some reason, the point is not found in any cell,
    // or it is on a cell that is not locally owned, return 0.
    if (cell_point.first == dof_handler.end() ||
        !cell_point.first->is_locally_owned())
      {
        for (auto &v : gradient)
          {
            v = 0;
          }
        return;
      }
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,
           ExcInternalError());

    const Quadrature<dim> quadrature(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
    FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients);
    fe_values.reinit(cell_point.first);
    std::vector<std::vector<Tensor<1, dim, Number>>> u_gradient(
      1, std::vector<Tensor<1, dim, Number>>(fe.n_components()));
    fe_values.get_function_gradients(fe_function, u_gradient);
    gradient = u_gradient[0];
  }

  template <int dim, typename VectorType>
  const typename DoFHandler<dim>::active_cell_iterator
  GridInterpolator<dim, VectorType>::get_cell() const
  {
    return cell_point.first;
  }

  template <int dim, typename MeshType>
  CellLocator<dim, MeshType>::CellLocator(
    DoFHandler<dim> &dh,
    const Point<dim> &p,
    const typename MeshType::active_cell_iterator &hint)
    : dof_handler(dh), point(p), hint(hint), cell_found(true)
  {
  }

  template <int dim, typename MeshType>
  const typename MeshType::active_cell_iterator
  CellLocator<dim, MeshType>::search()
  {
    // If the hint is the begin iterator we do not use BFS.
    if (hint == dof_handler.begin_active())
      {
        MappingQ1<dim> mapping;
        return (GridTools::find_active_cell_around_point(
                  mapping, dof_handler, point))
          .first;
      }
    // Create an unordered set to store the flagged cells
    std::unordered_set<std::string> flag_table;
    std::queue<typename MeshType::active_cell_iterator> cell_queue;
    // Start with the hint cell
    cell_queue.push(hint);
    while (!cell_queue.empty())
      {
        auto current_cell = cell_queue.front();
        flag_table.insert(current_cell->id().to_string());
        cell_queue.pop();
        // If the point is inside current cell then we are done.
        if (current_cell->point_inside(point))
          {
            return current_cell;
          }
        std::vector<typename MeshType::active_cell_iterator> neightbors;
        // Get all the active neighbors!
        GridTools::get_active_neighbors<MeshType>(current_cell, neightbors);
        for (unsigned int i = 0; i < neightbors.size(); ++i)
          {
            // Push all the unflagged cells into the queue
            if (flag_table.find(neightbors[i]->id().to_string()) ==
                flag_table.end())
              {
                cell_queue.push(neightbors[i]);
                flag_table.insert(neightbors[i]->id().to_string());
              }
          }
      }
    // If the queue is already empty, we failed in finding the cell
    cell_found = false;
    // Return an invalid iterator
    typename MeshType::active_cell_iterator invalid_itr;
    return invalid_itr;
  }

  // Written by Davis Wells on dealii mailing list.
  template <int dim>
  void GridCreator<dim>::flow_around_cylinder_2d(Triangulation<2> &tria,
                                                 bool compute_in_2d)
  {
    double left = compute_in_2d ? 0.0 : -0.3;

    // set up the bulk triangulation
    Triangulation<2> bulk_triangulation;
    GridGenerator::subdivided_hyper_rectangle(bulk_triangulation,
                                              {compute_in_2d ? 22u : 25u, 4u},
                                              Point<2>(left, 0.0),
                                              Point<2>(2.2, 0.41));
    std::set<Triangulation<2>::active_cell_iterator> cells_to_remove;
    Tensor<1, 2> cylinder_triangulation_offset;
    for (const auto cell : bulk_triangulation.active_cell_iterators())
      {
        if ((cell->center() - Point<2>(0.2, 0.2)).norm() < 0.15)
          cells_to_remove.insert(cell);

        if (cylinder_triangulation_offset == Point<2>())
          {
            for (unsigned int vertex_n = 0;
                 vertex_n < GeometryInfo<2>::vertices_per_cell;
                 ++vertex_n)
              if (cell->vertex(vertex_n) == Point<2>(left, 0.0))
                {
                  // skip two cells in the bottom left corner
                  cylinder_triangulation_offset =
                    2.0 * (cell->vertex(3) - Point<2>(left, 0.0));
                  break;
                }
          }
      }
    Triangulation<2> result_1;
    GridGenerator::create_triangulation_with_removed_cells(
      bulk_triangulation, cells_to_remove, result_1);

    // set up the cylinder triangulation
    Triangulation<2> cylinder_triangulation;
    GridGenerator::hyper_cube_with_cylindrical_hole(
      cylinder_triangulation, 0.05, 0.41 / 4.0);
    GridTools::shift(cylinder_triangulation_offset, cylinder_triangulation);
    // dumb hack
    for (const auto cell : cylinder_triangulation.active_cell_iterators())
      cell->set_material_id(2);

    // merge them together
    auto minimal_line_length = [](const Triangulation<2> &tria) -> double {
      double min_line_length = 1000.0;

      for (const auto cell : tria.active_cell_iterators())
        {

          min_line_length = std::min(
            min_line_length, (cell->vertex(0) - cell->vertex(1)).norm());
          min_line_length = std::min(
            min_line_length, (cell->vertex(0) - cell->vertex(2)).norm());
          min_line_length = std::min(
            min_line_length, (cell->vertex(1) - cell->vertex(3)).norm());
          min_line_length = std::min(
            min_line_length, (cell->vertex(2) - cell->vertex(3)).norm());
        }
      return min_line_length;
    };

    // the cylindrical triangulation might not match the Cartesian grid: as a
    // result the vertices might not be lined up. Get around this by deleting
    // duplicated vertices with a very low numerical tolerance.
    const double tolerance =
      std::min(minimal_line_length(result_1),
               minimal_line_length(cylinder_triangulation)) /
      2.0;

    GridGenerator::merge_triangulations(
      result_1, cylinder_triangulation, tria, tolerance);

    const types::manifold_id tfi_id = 1;

    const types::manifold_id polar_id = 0;
    for (const auto cell : tria.active_cell_iterators())

      {
        // set all non-boundary manifold ids to the new TFI manifold id.

        if (cell->material_id() == 2)
          {

            cell->set_manifold_id(tfi_id);
            for (unsigned int face_n = 0;
                 face_n < GeometryInfo<2>::faces_per_cell;
                 ++face_n)
              {

                if (cell->face(face_n)->at_boundary())
                  cell->face(face_n)->set_manifold_id(polar_id);
                else
                  cell->face(face_n)->set_manifold_id(tfi_id);
              }
          }
      }

    PolarManifold<2> polar_manifold(Point<2>(0.2, 0.2));
    tria.set_manifold(polar_id, polar_manifold);
    TransfiniteInterpolationManifold<2> inner_manifold;
    inner_manifold.initialize(tria);
    tria.set_manifold(tfi_id, inner_manifold);

    std::vector<Point<2> *> inner_pointers;
    for (const auto cell : tria.active_cell_iterators())
      {

        for (unsigned int face_n = 0; face_n < GeometryInfo<2>::faces_per_cell;
             ++face_n)
          {

            if (cell->face(face_n)->manifold_id() == polar_id)
              {
                inner_pointers.push_back(&cell->face(face_n)->vertex(0));
                inner_pointers.push_back(&cell->face(face_n)->vertex(1));
              }
          }
      }
    // de-duplicate
    std::sort(inner_pointers.begin(), inner_pointers.end());
    inner_pointers.erase(
      std::unique(inner_pointers.begin(), inner_pointers.end()),
      inner_pointers.end());

    // find the current center...
    Point<2> center;
    for (const Point<2> *const ptr : inner_pointers)
      center += *ptr / double(inner_pointers.size());

    // and recenter at (0.2, 0.2)
    for (Point<2> *const ptr : inner_pointers)
      *ptr += Point<2>(0.2, 0.2) - center;

    Point<2> center2;
    for (const Point<2> *const ptr : inner_pointers)
      center2 += *ptr / double(inner_pointers.size());
  }

  // Create 2D triangulation:
  template <>
  void GridCreator<2>::flow_around_cylinder(Triangulation<2> &tria)
  {
    flow_around_cylinder_2d(tria);
    // Set the left boundary (inflow) to 0, the right boundary (outflow) to 1,
    // upper to 2, lower to 3 and the cylindrical surface to 4.
    for (Triangulation<2>::active_cell_iterator cell = tria.begin();
         cell != tria.end();
         ++cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                if (std::abs(cell->face(f)->center()[0] - 2.2) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(1);
                  }
                else if (std::abs(cell->face(f)->center()[0] - 0.0) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(0);
                  }
                else if (std::abs(cell->face(f)->center()[1] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(3);
                  }
                else if (std::abs(cell->face(f)->center()[1]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(2);
                  }
                else
                  {
                    cell->face(f)->set_all_boundary_ids(4);
                  }
              }
          }
      }
  }

  // Create 3D triangulation:
  template <>
  void GridCreator<3>::flow_around_cylinder(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    flow_around_cylinder_2d(tria_2d, false);
    GridGenerator::extrude_triangulation(tria_2d, 9, 0.41, tria);
    // Set boundaries in x direction to 0 and 1; y direction to 2 and 3;
    // z direction to 4 and 5; the cylindrical surface 6.
    for (Triangulation<3>::active_cell_iterator cell = tria.begin();
         cell != tria.end();
         ++cell)
      {
        for (unsigned int f = 0; f < GeometryInfo<3>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary())
              {
                if (std::abs(cell->face(f)->center()[0] - 2.2) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(1);
                  }
                else if (std::abs(cell->face(f)->center()[0] + 0.3) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(0);
                  }
                else if (std::abs(cell->face(f)->center()[1] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(3);
                  }
                else if (std::abs(cell->face(f)->center()[1]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(2);
                  }
                else if (std::abs(cell->face(f)->center()[2] - 0.41) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(5);
                  }
                else if (std::abs(cell->face(f)->center()[2]) < 1e-12)
                  {
                    cell->face(f)->set_all_boundary_ids(4);
                  }
                else
                  {
                    cell->face(f)->set_all_boundary_ids(6);
                  }
              }
          }
      }
  }

  template <int dim>
  void GridCreator<dim>::sphere(Triangulation<dim> &tria,
                                const Point<dim> &center,
                                double radius)
  {
    SphericalManifold<dim> spherical_manifold(center);
    TransfiniteInterpolationManifold<dim> inner_manifold;
    GridGenerator::hyper_ball(tria, center, radius);
    tria.set_all_manifold_ids(1);
    tria.set_all_manifold_ids_on_boundary(0);
    tria.set_manifold(0, spherical_manifold);
    inner_manifold.initialize(tria);
    tria.set_manifold(1, inner_manifold);
  }

  template <>
  void GridCreator<2>::cylinder(Triangulation<2> &tria,
                                const double radius,
                                const double length)
  {
    (void)tria;
    (void)radius;
    (void)length;
    AssertThrow(false, ExcNotImplemented());
  }

  template <>
  void GridCreator<3>::cylinder(Triangulation<3> &tria,
                                const double radius,
                                const double length)
  {
    Triangulation<2> tria2d;
    Point<2> center(0, 0);
    GridCreator<2>::sphere(tria2d, center, radius);
    GridGenerator::extrude_triangulation(
      tria2d, static_cast<unsigned int>(length / (4 * radius)), length, tria);
    tria.set_all_manifold_ids_on_boundary(0);
    for (auto cell = tria.begin(); cell != tria.end(); ++cell)
      {
        for (unsigned int i = 0; i < GeometryInfo<3>::faces_per_cell; ++i)
          {
            if (cell->at_boundary(i))
              {
                if (std::abs(cell->face(i)->center()(2)) < 1e-10)
                  {
                    cell->face(i)->set_boundary_id(1);
                    cell->face(i)->set_manifold_id(numbers::flat_manifold_id);
                  }
                else if (std::abs(cell->face(i)->center()(2) - length) < 1e-10)
                  {
                    cell->face(i)->set_boundary_id(2);
                    cell->face(i)->set_manifold_id(numbers::flat_manifold_id);
                  }
              }
          }
      }
    tria.set_manifold(0, CylindricalManifold<3>(2));
  }

  template class GridCreator<2>;
  template class GridCreator<3>;
  template class GridInterpolator<2, Vector<double>>;
  template class GridInterpolator<3, Vector<double>>;
  template class GridInterpolator<2, BlockVector<double>>;
  template class GridInterpolator<3, BlockVector<double>>;
  template class GridInterpolator<2, PETScWrappers::MPI::BlockVector>;
  template class GridInterpolator<3, PETScWrappers::MPI::BlockVector>;
  template class SPHInterpolator<2, Vector<double>>;
  template class SPHInterpolator<3, Vector<double>>;
  template class SPHInterpolator<2, PETScWrappers::MPI::BlockVector>;
  template class SPHInterpolator<3, PETScWrappers::MPI::BlockVector>;
  template class Utils::CellLocator<2, DoFHandler<2, 2>>;
  template class Utils::CellLocator<3, DoFHandler<3, 3>>;
} // namespace Utils

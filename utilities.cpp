#include "utilities.h"

namespace Utils
{
  // The code to create triangulation is copied from [Martin Kronbichler's code]
  // (https://github.com/kronbichler/adaflo/blob/master/tests/flow_past_cylinder.cc)
  // with very few modifications.
  // Helper function used in both 2d and 3d:
  void Utils::GridCreator::flow_around_cylinder_2d(Triangulation<2> &tria, bool compute_in_2d)
  {
    SphericalManifold<2> boundary(Point<2>(0.5, 0.2));
    Triangulation<2> left, middle, right, tmp, tmp2;
    GridGenerator::subdivided_hyper_rectangle(
      left,
      std::vector<unsigned int>({3U, 4U}),
      Point<2>(),
      Point<2>(0.3, 0.41),
      false);
    GridGenerator::subdivided_hyper_rectangle(
      right,
      std::vector<unsigned int>({18U, 4U}),
      Point<2>(0.7, 0),
      Point<2>(2.5, 0.41),
      false);

    // Create middle part first as a hyper shell.
    GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
    middle.set_manifold(0, boundary);
    middle.refine_global(1);

    // Then move the vertices to the points where we want them to be to create a
    // slightly asymmetric cube with a hole
    for (Triangulation<2>::cell_iterator cell = middle.begin();
         cell != middle.end();
         ++cell)
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
          Point<2> &vertex = cell->vertex(v);
          if (std::abs(vertex[0] - 0.7) < 1e-10 &&
              std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.7, 0.205);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.7, 0.41);
          else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.7, 0);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.4) < 1e-10)
            vertex = Point<2>(0.5, 0.41);
          else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                   std::abs(vertex[1] - 0.0) < 1e-10)
            vertex = Point<2>(0.5, 0.0);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.3) < 1e-10)
            vertex = Point<2>(0.3, 0.41);
          else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                   std::abs(vertex[1] - 0.1) < 1e-10)
            vertex = Point<2>(0.3, 0);
          else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
                   std::abs(vertex[1] - 0.2) < 1e-10)
            vertex = Point<2>(0.3, 0.205);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.59, 0.11);
          else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.59, 0.29);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.13621) < 1e-4)
            vertex = Point<2>(0.41, 0.11);
          else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                   std::abs(vertex[1] - 0.26379) < 1e-4)
            vertex = Point<2>(0.41, 0.29);
        }

    // Refine once to create the same level of refinement as in the
    // neighboring domains:
    middle.refine_global(1);

    // Must copy the triangulation because we cannot merge triangulations with
    // refinement:
    GridGenerator::flatten_triangulation(middle, tmp2);

    // Left domain is requred in 3d only.
    if (compute_in_2d)
    {
      GridGenerator::merge_triangulations(tmp2, right, tria);
    }
    else
    {
      GridGenerator::merge_triangulations(left, tmp2, tmp);
      GridGenerator::merge_triangulations(tmp, right, tria);
    }
  }

  // Create 2D triangulation:
  void GridCreator::flow_around_cylinder(Triangulation<2> &tria)
  {
    flow_around_cylinder_2d(tria);
    // Set the cylinder boundary to 1, the right boundary (outflow) to 2, the rest to 0.
    for (Triangulation<2>::active_cell_iterator cell = tria.begin();
         cell != tria.end();
         ++cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary())
          {
            if (std::abs(cell->face(f)->center()[0] - 2.5) < 1e-12)
            {
              cell->face(f)->set_all_boundary_ids(2);
            }
            else if (Point<2>(0.5, 0.2).distance(cell->face(f)->center()) <= 0.05)
              {
                cell->face(f)->set_all_manifold_ids(10);
                cell->face(f)->set_all_boundary_ids(1);
              }
            else
            {
              cell->face(f)->set_all_boundary_ids(0);
            }
          }
      }
    }
  }

  // Create 3D triangulation:
  void GridCreator::flow_around_cylinder(Triangulation<3> &tria)
  {
    Triangulation<2> tria_2d;
    flow_around_cylinder_2d(tria_2d, false);
    GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);
    // Set the cylinder boundary to 1, the right boundary (outflow) to 2, the rest to 0.
    for (Triangulation<3>::active_cell_iterator cell = tria.begin();
        cell != tria.end(); ++cell)
    {
      for (unsigned int f = 0; f<GeometryInfo<3>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
          {
            cell->face(f)->set_all_boundary_ids(2);
          }
          else if (Point<3>(0.5, 0.2, cell->face(f)->center()[2]).distance
            (cell->face(f)->center()) <= 0.05)
          {
            cell->face(f)->set_all_manifold_ids(10);
            cell->face(f)->set_all_boundary_ids(1);
          }
          else
          {
            cell->face(f)->set_all_boundary_ids(0);
          }
        }
      }
    }
  }
}

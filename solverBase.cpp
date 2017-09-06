#include "solverBase.h"

namespace IFEM
{
  using namespace dealii;
  using namespace std;

  template<>
  void SolverBase<2>::generateMesh()
  {
    GridGenerator::subdivided_hyper_rectangle(this->tria, std::vector<unsigned int>{128, 16},
      Point<2>(0.,0.), Point<2>(8.,1.), true);
    std::cout << "Mesh info: " << endl << "dimension: 2" 
      << " number of cells: " << this->tria.n_active_cells() << std::endl;
  }

  template<>
  void SolverBase<3>::generateMesh()
  {
    GridGenerator::subdivided_hyper_rectangle(this->tria, std::vector<unsigned int>{32, 4, 4},
      Point<3>(0.,0.,0.), Point<3>(8.,1.,1.), true);
    std::cout << "Mesh info: " << endl << "dimension: 3"
      << " number of cells: " << this->tria.n_active_cells() << std::endl;
  }

  template<int dim>
  void SolverBase<dim>::readMesh(const std::string& fileName)
  {
    GridIn<dim> gridIn;
    gridIn.attach_triangulation(this->tria);
    ifstream file(fileName);
    gridIn.read_abaqus(file);

    cout << "Mesh info: " << endl << "dimension: " << dim
      << " number of cells: " << this->tria.n_active_cells() << endl;
    map<unsigned int, unsigned int> boundaryCount;
    for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      for (unsigned int face = 0; face<GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if (cell->face(face)->at_boundary())
        {
          boundaryCount[cell->face(face)->boundary_id()]++;
        }
      }
    }
    cout << "Boundary indicators: ";
    for (auto itr = boundaryCount.begin(); itr != boundaryCount.end(); ++itr)
    {
      cout << itr->first << " (" << itr->second << " times) ";
    }
    cout << endl;
  }

  template<int dim>
  void SolverBase<dim>::readBC(const std::string& fileName)
  {
    using boost::property_tree::ptree;
    std::ifstream jsonFile(fileName);
    ptree root;
    read_json(jsonFile, root);

    for (auto itr = root.begin(); itr != root.end(); ++itr)
    {
      auto item = itr->second;
      string type = item.get<string>("type"); // must have
      if (type == "traction" || type == "gravity")
      {
        // expecting an array of double
        vector<double> components;
        BOOST_FOREACH(ptree::value_type& v, item.get_child("value."))
        {
          stringstream ss;
          double val;
          ss << v.second.data();
          ss >> val;
          components.push_back(val);
        }
        if (components.size() < static_cast<unsigned int>(dim))
        {
          throw runtime_error("Traction/gravity has insufficient number of components!");
        }
        Tensor<1, dim> temp;
        for (unsigned int i = 0; i < dim; ++i)
        {
          temp[i] = components[i];
        }
        if (type == "traction")
        {
          unsigned int boundaryId = item.get<unsigned int>("boundary_id");
          this->bc.traction[boundaryId] = temp;
        }
        else
        {
          this->bc.gravity = temp;
        }
      }
      else if (type == "pressure")
      {
        unsigned int boundaryId = item.get<unsigned int>("boundary_id");
        // expecting a double
        this->bc.pressure[boundaryId] = item.get<double>("value");
      }
      else if (type == "displacement")
      {
        unsigned int boundaryId = item.get<unsigned int>("boundary_id");
        // expecting an array of unsigned int and an array of double
        vector<bool> dofs;
        vector<double> vals;
        BOOST_FOREACH(ptree::value_type& v, item.get_child("dof."))
        {
          stringstream ss;
          int dof;
          ss << v.second.data();
          ss >> dof;
          dofs.push_back(static_cast<bool>(dof));
        }
        BOOST_FOREACH(ptree::value_type& v, item.get_child("value."))
        {
          stringstream ss;
          double val;
          ss << v.second.data();
          ss >> val;
          vals.push_back(val);
        }
        if (dofs.size() != vals.size())
        {
          throw runtime_error("Displacement bc has different numbers of dofs and values!");
        }
        if (static_cast<int>(dofs.size()) > dim)
        {
          dofs.resize(dim);
          vals.resize(dim);
        }
        this->bc.displacement[boundaryId] = {dofs, vals};
      }
      else
      {
        throw runtime_error("Unknown type of boundary condition!");
      }
    }
    // print BC
    cout << "BC info:" << endl;
    cout << "gravity = ";
    for (unsigned int i = 0; i < dim; ++i)
    {
      cout << this->bc.gravity[i] << " ";
    }
    cout << endl;
    for (auto itr = this->bc.traction.begin(); itr != this->bc.traction.end(); ++itr)
    {
      cout << "boundary_id = " << itr->first << " traction = ";
      for (unsigned int i = 0; i < dim; ++i)
      {
        cout << itr->second[i] << " ";
      }
      cout << endl;
    }
    for (auto itr = this->bc.pressure.begin(); itr != this->bc.pressure.end(); ++itr)
    {
      cout << "boundary_id = " << itr->first << " pressure = " << itr->second << endl;
    }
    for (auto itr = this->bc.displacement.begin(); itr != this->bc.displacement.end(); ++itr)
    {
      cout << "boundary_id = " << itr->first << " dofs = ";
      for (auto i : itr->second.first)
      {
        cout << i << " ";
      }
      cout << "displacements = ";
      for (auto i : itr->second.second)
      {
        cout << i << " ";
      }
      cout << endl;
    }
  }

  template<int dim>
  void SolverBase<dim>::setup()
  {
    this->dofHandler.distribute_dofs(this->fe);
    DoFRenumbering::Cuthill_McKee(this->dofHandler);
    DynamicSparsityPattern dsp(this->dofHandler.n_dofs(), this->dofHandler.n_dofs());
    DoFTools::make_sparsity_pattern(this->dofHandler, dsp);
    this->pattern.copy_from(dsp);
    this->tangentStiffness.reinit(this->pattern);
    this->mass.reinit(this->pattern);
    this->sysMatrix.reinit(this->pattern);
    this->solution.reinit(this->dofHandler.n_dofs());
    this->sysRhs.reinit(this->dofHandler.n_dofs());
  }

  template<int dim>
  void SolverBase<dim>::applyBC(SparseMatrix<double>& A, Vector<double>& x, Vector<double>& b)
  {
    map<types::global_dof_index, double> boundaryValues;
    for (auto itr = this->bc.displacement.begin(); itr != this->bc.displacement.end(); ++itr)
    {
      unsigned int id = itr->first;
      // If deal.II version is 9.0, use Functions::ConstantFunction instead
      ConstantFunction<dim> function(itr->second.second); // value of the BC
      ComponentMask mask(itr->second.first);
      VectorTools::interpolate_boundary_values(this->dofHandler, id,
        function, boundaryValues, mask);
    }
    MatrixTools::apply_boundary_values(boundaryValues, A, x, b);
  }

  template<int dim>
  unsigned int SolverBase<dim>::solve(const SparseMatrix<double>& A,
    Vector<double>& x, const Vector<double>& b)
  {
    SolverControl control(5000, 1e-12);
    SolverCG<> cg(control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(this->tangentStiffness, 1.2);
    cg.solve(A, x, b, preconditioner);
    return control.last_step();
  }

  template class SolverBase<2>;
  template class SolverBase<3>;
}

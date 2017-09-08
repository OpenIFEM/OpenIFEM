#include "linearElasticSolver.h"

namespace
{
  using namespace dealii;
  /**
   * Helper function that returns the contribution to strain epsilon(i, j) at the q-th
   * quadrature point made by the a-th shape function where a is in the range of dofs_per_cell.
   * Note that shape_grad_component(a, q, i) returns the full gradient of the i-th component
   * of the a-th shape function at the q-th quadrature point...
   * Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_18.html
   */
  template<int dim>
  inline
  SymmetricTensor<2, dim> getStrain(const FEValues<dim>& feValues,
    const unsigned int a, const unsigned int q)
  {
    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
    {
      strain[i][i] = feValues.shape_grad_component(a, q, i)[i];
    }
    for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = i; j < dim; ++j)
      {
        strain[i][j] = (feValues.shape_grad_component(a, q, i)[j] +
          feValues.shape_grad_component(a, q, j)[i])/2;
      }
    }
    return strain;
  }

  /**
   * Helper function to compute the strain epsilon(i, j) at a quadrature point.
   * FEValues::get_function_gradients extracts the gradients of each component of the
   * solution field at a quadrature point, which is [u_{i, j}, v_{i, j}, w_{i, j}].
   * We simply take this vector to compute the strain in a straight-forward way.
   */
  template<int dim>
  inline
  SymmetricTensor<2, dim> getStrain(const std::vector<Tensor<1, dim>>& grad)
  {
    Assert(grad.size() == dim, ExcInternalError());
    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i)
    {
      strain[i][i] = grad[i][i];
    }
    for (unsigned int i = 0; i < dim; ++i)
    {
      for (unsigned int j = i; j < dim; ++j)
      {
        strain[i][j] = (grad[i][j] + grad[j][i])/2;
      }
    }
    return strain;
  }

  template<int dim>
  class StrainPostprocessor : public DataPostprocessorTensor<dim>
  {
  public:
    StrainPostprocessor() : DataPostprocessorTensor<dim>("strain", update_gradients) {}
    virtual void evaluate_vector_field(
      const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      // ensure that there really are as many output slots
      // as there are points at which DataOut provides the
      // gradients:
      AssertDimension(input_data.solution_gradients.size(),
        computed_quantities.size());
      for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
      {
        // ensure that each output slot has exactly 'dim*dim'
        // components (as should be expected, given that we
        // want to create tensor-valued outputs), and copy the
        // gradients of the solution at the evaluation points
        // into the output slots:
        AssertDimension (computed_quantities[p].size(),
                          (Tensor<2,dim>::n_independent_components));
        for (unsigned int d=0; d<dim; ++d)
        {
          for (unsigned int e=0; e<dim; ++e)
          {
            computed_quantities[p][Tensor<2,dim>::component_to_unrolled_index
              (TableIndices<2>(d,e))] = (input_data.solution_gradients[p][d][e] +
              input_data.solution_gradients[p][e][d])/2;
          }
        }
      }
    }
  };

  template<int dim>
  class StressPostprocessor : public DataPostprocessorTensor<dim>
  {
  public:
    StressPostprocessor(const dealii::SymmetricTensor<4, dim>& C)
      : DataPostprocessorTensor<dim>("stress", update_gradients), elasticity(C) {}
    virtual void evaluate_vector_field(
      const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override
    {
      AssertDimension(input_data.solution_gradients.size(),
        computed_quantities.size());
      std::vector<SymmetricTensor<2, dim>> strain(computed_quantities.size());
      std::vector<SymmetricTensor<2, dim>> stress(computed_quantities.size());
      for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
      {
        AssertDimension (computed_quantities[p].size(),
                          (Tensor<2,dim>::n_independent_components));
        for (unsigned int d=0; d<dim; ++d)
        {
          for (unsigned int e=0; e<dim; ++e)
          {
            strain[p][d][e] = (input_data.solution_gradients[p][d][e] +
              input_data.solution_gradients[p][e][d])/2;
          }
        }
        stress[p] = elasticity*strain[p];
        // computed_quantities has dim*dim components
        // so we have to convert SymmetricTensor to general tensor
        Tensor<2, dim> temp = stress[p];
        temp.unroll(computed_quantities[p]);
      }
    }
  private:
    SymmetricTensor<4, dim> elasticity;
  };
}

namespace IFEM
{
  using namespace dealii;

  template<int dim>
  LinearElasticSolver<dim>::AssemblyScratchData::AssemblyScratchData(
    const FiniteElement<dim> &finite_element, const QGauss<dim> &quad,
    const QGauss<dim-1> &face_quad) : fe_values(finite_element, quad,
    update_values | update_gradients | update_quadrature_points | update_JxW_values),
    fe_face_values(finite_element, face_quad,
      update_values | update_quadrature_points | update_normal_vectors | update_JxW_values) {}

  template<int dim>
  LinearElasticSolver<dim>::AssemblyScratchData::AssemblyScratchData(
    const AssemblyScratchData &scratch) :
    fe_values(scratch.fe_values.get_fe(), scratch.fe_values.get_quadrature(),
      scratch.fe_values.get_update_flags()),
    fe_face_values(scratch.fe_face_values.get_fe(), scratch.fe_face_values.get_quadrature(),
      scratch.fe_face_values.get_update_flags()) {}

  template<int dim>
  void LinearElasticSolver<dim>::localAssemble(const typename
      DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch,
      AssemblyCopyData &data)
  {
    auto mat = std::dynamic_pointer_cast<LinearElasticMaterial<dim>>(this->material);
    Assert(mat, ExcInternalError());
    SymmetricTensor<4, dim> elasticity = mat->getElasticityTensor();
    const unsigned int   dofsPerCell = this->fe.dofs_per_cell;
    const unsigned int   numQuadPts  = this->quadFormula.size();
    const unsigned int numFaceQuadPts = this->faceQuadFormula.size();
    data.cell_matrix.reinit(dofsPerCell, dofsPerCell);
    data.cell_rhs.reinit(dofsPerCell);
    data.dof_indices.resize(dofsPerCell);
    scratch.fe_values.reinit(cell);
    // matrix
    for (unsigned int i = 0; i < dofsPerCell; ++i)
    {
      for (unsigned int j = 0; j < dofsPerCell; ++j)
      {
        for (unsigned int q = 0; q < numQuadPts; ++q)
        {
            data.cell_matrix(i,j) += getStrain(scratch.fe_values, i, q)*elasticity*
              getStrain(scratch.fe_values, j, q)*scratch.fe_values.JxW(q);
        }
      }
    }
    // body force
    double rho = this->material->getDensity();
    for (unsigned int i = 0; i < dofsPerCell; ++i)
    {
      const unsigned int component_i = this->fe.system_to_component_index(i).first;
      for (unsigned int q = 0; q < numQuadPts; ++q)
      {
        data.cell_rhs(i) += scratch.fe_values.shape_value(i, q)*rho*
          this->bc.gravity[component_i]*scratch.fe_values.JxW(q);
      }
    }
    // traction
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    {
      if (cell->face(i)->at_boundary() &&
        this->bc.traction.find(cell->face(i)->boundary_id()) != this->bc.traction.end())
      {
        scratch.fe_face_values.reinit(cell, i);
        Tensor<1,dim> traction = this->bc.traction[cell->face(i)->boundary_id()];
        for (unsigned int q = 0; q < numFaceQuadPts; ++q)
        {
          for (unsigned int j = 0; j < dofsPerCell; ++j)
          {
            const unsigned int component_j = this->fe.system_to_component_index(j).first;
            data.cell_rhs(j) += scratch.fe_face_values.shape_value(j, q)*traction[component_j]
              *scratch.fe_face_values.JxW(q);
          }
        }
      }
    }
    cell->get_dof_indices(data.dof_indices);
  }

  template<int dim>
  void LinearElasticSolver<dim>::scatter(const AssemblyCopyData &data)
  {
    for (unsigned int i = 0; i < data.dof_indices.size(); ++i)
    {
      for (unsigned int j = 0; j < data.dof_indices.size(); ++j)
      {
        this->tangentStiffness.add(data.dof_indices[i], data.dof_indices[j],
          data.cell_matrix(i,j));
      }
      this->sysRhs(data.dof_indices[i]) += data.cell_rhs(i);
    }
  }

  template<int dim>
  void LinearElasticSolver<dim>::globalAssemble()
  {
    /**
     * WorkStream is designed such that the first function runs in parallel and
     * the second function call runs in serial.
     */
    WorkStream::run(this->dofHandler.begin_active(), this->dofHandler.end(),
      *this, &LinearElasticSolver::localAssemble, &LinearElasticSolver::scatter,
      AssemblyScratchData(this->fe, this->quadFormula, this->faceQuadFormula),
      AssemblyCopyData());
  }

  template <int dim>
  void LinearElasticSolver<dim>::output(const unsigned int cycle) const
  {
    // Output the strain, stress and displacements
    std::string fileName = "LinearElastic-" + Utilities::int_to_string(cycle, 4) + ".vtu";
    std::ofstream out(fileName.c_str());
    // Pitfall! StrainPostprocessor must be declared before DataOut because the latter is
    // holding a pointer to the former!
    StrainPostprocessor<dim> strain;
    auto mat = std::dynamic_pointer_cast<LinearElasticMaterial<dim>>(this->material);
    Assert(mat, ExcInternalError());
    StressPostprocessor<dim> stress(mat->getElasticityTensor());
    DataOut<dim> data;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data.add_data_vector(this->dofHandler, this->solution,
      std::vector<std::string>(dim, "displacement"), interpretation);
    data.add_data_vector(this->dofHandler, this->solution, strain);
    data.add_data_vector(this->dofHandler, this->solution, stress);
    data.build_patches();
    data.write_vtu(out);
  }

  template<int dim>
  void LinearElasticSolver<dim>::runStatics(const std::string& fileName)
  {
    if (fileName.empty())
    {
      this->generateMesh();
    }
    else
    {
      this->readMesh(fileName);
    }
    this->readBC();
    this->setup();
    this->globalAssemble();
    this->applyBC(this->tangentStiffness, this->solution, this->sysRhs);
    this->output(0);
    auto n = this->solve(this->tangentStiffness, this->solution, this->sysRhs);
    std::cout << "Matrix solver converged in " << n << " steps" << std::endl;
    this->output(1);
  }

  template<int dim>
  void LinearElasticSolver<dim>::runDynamics(const std::string& fileName)
  {
    const double damping = 1.5;
    const double gamma = 0.5 + damping;
    const double beta = gamma/2;
    const double dt = 1;
    const int nsteps = 200;
    const int nprint = 10;
    int step = 0;

    if (fileName.empty())
    {
      this->generateMesh();
    }
    else
    {
      this->readMesh(fileName);
    }
    this->readBC();
    this->setup();
    this->globalAssemble();
    this->output(0);

    Vector<double> un(this->dofHandler.n_dofs()); // displacement at step n
    Vector<double> un1(this->dofHandler.n_dofs()); // displacement at step n+1
    Vector<double> vn(this->dofHandler.n_dofs()); // velocity at step n
    Vector<double> vn1(this->dofHandler.n_dofs()); // velocity at step n+1
    Vector<double> an(this->dofHandler.n_dofs()); // acceleration at step n
    Vector<double> an1(this->dofHandler.n_dofs()); // acceleration at step n+1

    Vector<double> temp1(this->dofHandler.n_dofs()); // temp1 = un + dt*vn + dt^2*(0.5-beta)*an
    Vector<double> temp2(this->dofHandler.n_dofs()); // temp2 = -K*temp1 + Fn+1

    // a0 = F0/M
    MatrixCreator::create_mass_matrix(this->dofHandler, this->quadFormula, this->mass); // rho=1
    this->mass *= this->material->getDensity();
    this->applyBC(this->mass, an, this->sysRhs);
    this->solve(this->mass, an, this->sysRhs);

    // A = M + beta*dt^2*K
    this->sysMatrix = 0.0;
    this->sysMatrix.add(1.0, this->tangentStiffness);
    this->sysMatrix.add(beta*dt*dt, this->mass);

    while (step < nsteps)
    {
      temp1 = un;
      temp1.add(dt, vn, dt*dt*(0.5-beta), an);
      this->tangentStiffness.vmult(temp2, temp1);
      temp2 *= -1.0;
      temp2 += this->sysRhs;
      this->applyBC(this->sysMatrix, an1, temp2);
      this->solve(this->sysMatrix, an1, temp2);
      // un1 = un + dt*vn + dt^2*((0.5-beta)*an + beta*an1)
      un1 = un;
      un1.add(dt, vn);
      un1.add((0.5-beta)*dt*dt, an, beta*dt*dt, an1);
      // vn1 = vn + dt*((1-gamma)*an + gamma*an1)
      vn1 = vn;
      vn1.add((1-gamma)*dt, an, gamma*dt, an1);
      un = un1;
      vn = vn1;
      an = an1;
      ++step;
      if (step % nprint == 0)
      {
        this->solution = un;
        this->output(step);
      }
    }
  }

  template class LinearElasticSolver<2>;
  template class LinearElasticSolver<3>;
}

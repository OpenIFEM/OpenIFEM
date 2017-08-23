template<int dim>
void LinearElasticSolver<dim>::setMaterial(Material<dim>* mat)
{
  LinearMaterial<dim>* linear = dynamic_cast<LinearMaterial<dim>*>(mat);
  if (!linear)
  {
    throw runtime_error("A linear material is expected in a linear elastic solver!");
  }
  this->material = make_shared<LinearMaterial<dim>>(*linear);
}

template<int dim>
void LinearElasticSolver<dim>::assemble()
{
  shared_ptr<LinearMaterial<dim>> linear =
    dynamic_pointer_cast<LinearMaterial<dim>>(this->material);
  if (!linear)
  {
    throw runtime_error("A linear material is expected in a linear elastic solver!");
  }
  SymmetricTensor<4, dim> elasticity = linear->getElasticityTensor();
  // fe values for volume integration
  FEValues<dim> feValues (this->fe, this->quadFormula,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  // fe values for surface integration
  FEFaceValues<dim> feFaceValues(this->fe, this->faceQuadFormula,
    update_values | update_quadrature_points |
    update_normal_vectors | update_JxW_values);
  const unsigned int   dofsPerCell = this->fe.dofs_per_cell;
  const unsigned int   numQuadPts  = this->quadFormula.size();
  const unsigned int numFaceQuadPts = this->faceQuadFormula.size();
  FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
  Vector<double> cellRhs (dofsPerCell);
  vector<types::global_dof_index> localDofIndices(dofsPerCell);
  // TODO: these should not be hardcoded!
  double lambda = this->material->getLameFirst();
  double mu = this->material->getLameSecond();
  double rho = this->material->getDensity();
  for (auto cell = this->dofHandler.begin_active();
    cell != this->dofHandler.end(); ++cell)
  {
    cellMatrix = 0.;
    cellRhs = 0.;
    feValues.reinit (cell);
    // matrix
    for (unsigned int i=0; i < dofsPerCell; ++i)
    {
      const unsigned int component_i = this->fe.system_to_component_index(i).first;
      for (unsigned int j = 0; j < dofsPerCell; ++j)
      {
        const unsigned int component_j = this->fe.system_to_component_index(j).first;
        for (unsigned int q = 0; q < numQuadPts; ++q)
        {
          cellMatrix(i,j) += (
            feValues.shape_grad(i,q)[component_i] *
            feValues.shape_grad(j,q)[component_j] * lambda +
            feValues.shape_grad(i,q)[component_j] *
            feValues.shape_grad(j,q)[component_i] * mu +
            (component_i == component_j ?
             (feValues.shape_grad(i,q)*feValues.shape_grad(j,q)*mu) : 0)
            ) * feValues.JxW(q);
        }
      }
    }
    // body force
    for (unsigned int i=0; i < dofsPerCell; ++i)
    {
      const unsigned int component_i = this->fe.system_to_component_index(i).first;
      for (unsigned int q = 0; q < numQuadPts; ++q)
      {
        cellRhs(i) += feValues.shape_value(i, q)*rho*
          this->bc.gravity[component_i]*feValues.JxW(q);
      }
    }
    // traction
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    {
      if (cell->face(i)->at_boundary() &&
        this->bc.traction.find(cell->face(i)->boundary_id()) != this->bc.traction.end())
      {
        feFaceValues.reinit(cell, i);
        Tensor<1,dim> traction = this->bc.traction[cell->face(i)->boundary_id()];
        for (unsigned int q = 0; q < numFaceQuadPts; ++q)
        {
          for (unsigned int j = 0; j < dofsPerCell; ++j)
          {
            const unsigned int component_j = this->fe.system_to_component_index(j).first;
            cellRhs(j) += feFaceValues.shape_value(j, q)*traction[component_j]
              *feFaceValues.JxW(q);
          }
        }
      }
    }
    // scatter
    cell->get_dof_indices(localDofIndices);
    for (unsigned int i = 0; i < dofsPerCell; ++i)
    {
      for (unsigned int j = 0; j < dofsPerCell; ++j)
      {
        this->sysMatrix.add(localDofIndices[i], localDofIndices[j], cellMatrix(i,j));
      }
      this->sysRhs(localDofIndices[i]) += cellRhs(i);
    }
  }
}

template<int dim>
void LinearElasticSolver<dim>::evaluateStressStrain()
{
  FEValues<dim> feValues (this->fe, this->quadFormula,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int   numQuadPts  = this->quadFormula.size();
  for (auto cell = this->dofHandler.begin_active(); cell != this->dofHandler.end(); ++cell)
  {
    feValues.reinit(cell);
    vector<vector<Tensor<1, dim>>> quadPointGradients(numQuadPts,
      vector<Tensor<1, dim>>(dim));
    feValues.get_function_gradients(this->solution, quadPointGradients);
    const unsigned int numComponents = (dim == 2? 3 : 6);
    vector<Vector<double>> strain(numQuadPts, Vector<double>(numComponents));
    vector<Vector<double>> stress(numQuadPts, Vector<double>(numComponents));
    FullMatrix<double> coefficient(numComponents, numComponents);
    double lambda = 1.0;
    double mu = 1.0;
    coefficient(0, 0) = lambda + 2*mu;
    coefficient(0, 1) = lambda;
    coefficient(1, 0) = lambda;
    coefficient(1, 1) = lambda + 2*mu;
    coefficient(2, 2) = mu;
    for (unsigned int q = 0; q < numQuadPts; ++q)
    {
      if (dim == 2)
      {
        strain[q](0) = quadPointGradients[q][0][0];
        strain[q](1) = quadPointGradients[q][1][1];
        strain[q](2) = quadPointGradients[q][0][1] + quadPointGradients[q][1][0];
      }
      else
      {
        strain[q](0) = quadPointGradients[q][0][0];
        strain[q](1) = quadPointGradients[q][1][1];
        strain[q](2) = quadPointGradients[q][2][2];
        strain[q](3) = quadPointGradients[q][1][2] + quadPointGradients[q][2][1];
        strain[q](4) = quadPointGradients[q][0][2] + quadPointGradients[q][2][0];
        strain[q](5) = quadPointGradients[q][0][1] + quadPointGradients[q][1][0];
      }
      coefficient.vmult(stress[q], strain[q]);
    }
  }
}

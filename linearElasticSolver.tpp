template<int dim>
void LinearElasticSolver<dim>::assemble()
{
  // quadrature formula for volume integration, 2 Gauss points in each direction
  QGauss<dim>  quadratureFormula(2);
  // quadrature formula for surface integration, 2 Gauss points in each direction
  QGauss<dim-1> faceQuadratureFormula(2);
  // fe values for volume integration
  FEValues<dim> feValues (this->fe, quadratureFormula,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);
  // fe values for surface integration
  FEFaceValues<dim> feFaceValues(this->fe, faceQuadratureFormula,
    update_values | update_quadrature_points |
    update_normal_vectors | update_JxW_values);
  const unsigned int   dofsPerCell = this->fe.dofs_per_cell;
  const unsigned int   numQuadPts  = quadratureFormula.size();
  const unsigned int numFaceQuadPts = faceQuadratureFormula.size();
  FullMatrix<double> cellMatrix(dofsPerCell, dofsPerCell);
  Vector<double> cellRhs (dofsPerCell);
  vector<types::global_dof_index> localDofIndices(dofsPerCell);
  // TODO: these should not be hardcoded!
  double lambda = 1.;
  double mu = 1.;
  double rho = 1.;
  for (auto cell = this->dofHandler.begin_active();
    cell != this->dofHandler.end(); ++cell)
  {
    cellMatrix = 0.;
    cellRhs = 0.;
    feValues.reinit (cell);
    // matrix
    for (unsigned int i=0; i < dofsPerCell; ++i) {
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


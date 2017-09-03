namespace
{
  /** 
   * Helper function that returns the contribution to strain epsilon(i, j) at the q-th
   * quadrature point made by the a-th shape function where a is in the range of dofs_per_cell.
   * Note that shape_grad_component(a, q, i) returns the full gradient of the i-th component
   * of the a-th shape function at the q-th quadrature point...
   * Reference: http://www.dealii.org/8.5.0/doxygen/deal.II/step_18.html
   */
  template<int dim>
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
  SymmetricTensor<2, dim> getStrain(const vector<Tensor<1, dim>>& grad)
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
}

template<int dim>
void LinearElasticSolver<dim>::assemble()
{
  SymmetricTensor<4, dim> elasticity = this->material.getElasticityTensor();
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
  for (auto cell = this->dofHandler.begin_active();
    cell != this->dofHandler.end(); ++cell)
  {
    cellMatrix = 0.;
    cellRhs = 0.;
    feValues.reinit (cell);
    // matrix
    for (unsigned int i=0; i < dofsPerCell; ++i)
    {
      for (unsigned int j = 0; j < dofsPerCell; ++j)
      {
        for (unsigned int q = 0; q < numQuadPts; ++q)
        {
          cellMatrix(i,j) += getStrain(feValues, i, q)*elasticity*
            getStrain(feValues, j, q)*feValues.JxW(q);
        }
      }
    }
    // body force
    double rho = this->material.getDensity();
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
        this->tangentStiffness.add(localDofIndices[i], localDofIndices[j], cellMatrix(i,j));
      }
      this->sysRhs(localDofIndices[i]) += cellRhs(i);
    }
  }
}

template <int dim>
void LinearElasticSolver<dim>::output(const unsigned int cycle) const
{
  /*-------------------------------------------------------------------------------------*/
  // Compute stress and strain
  
  // A new pair of FE and DoFHandler whose nodal dofs are scalar
  FE_Q<dim> scalarFe(1);
  DoFHandler<dim> scalarDofHandler(this->tria);
  scalarDofHandler.distribute_dofs(scalarFe);

  // stress and strain should contain Sxx, Syy, Szz, Sxy, Sxz, Syz
  // (in 2D Szz, Sxz, Syz are zero)
  // Each of these components is a Vector of size n_dofs
  vector<Vector<double>> strainGlobal(6, Vector<double>(scalarDofHandler.n_dofs()));
  vector<Vector<double>> stressGlobal(6, Vector<double>(scalarDofHandler.n_dofs()));

  // Count how many cells share a particular dof
  vector<unsigned int> count(scalarDofHandler.n_dofs(), 0);

  SymmetricTensor<4, dim> elasticity = this->material.getElasticityTensor();

  // Need to re-calculate the gradients
  FEValues<dim> feValues (this->fe, this->quadFormula, update_gradients);
  // Average strain and stress in a cell
  SymmetricTensor<2, dim> cellAvgStrain, cellAvgStress;
  auto cell = this->dofHandler.begin_active();
  auto scalarCell = scalarDofHandler.begin_active();
  for (; cell != this->dofHandler.end(); ++cell, ++scalarCell)
  {
    cellAvgStrain.clear();
    cellAvgStress.clear();
    feValues.reinit(cell);
    // Gradients of all displacement components at all quadrature points
    vector<vector<Tensor<1, dim>>> quadDispGradients(this->quadFormula.size(),
      vector<Tensor<1, dim>>(dim));
    feValues.get_function_gradients(this->solution, quadDispGradients);
    for (unsigned int q = 0; q < this->quadFormula.size(); ++q)
    {
      // strain and stress tensors at a quadrature point
      SymmetricTensor<2, dim> quadStrain = getStrain(quadDispGradients[q]);
      SymmetricTensor<2, dim> quadStress = elasticity*quadStrain;
      cellAvgStrain += quadStrain;
      cellAvgStress += quadStress;
    }
    cellAvgStrain /= this->quadFormula.size();
    cellAvgStress /= this->quadFormula.size();
    // Distribute the cell average strain and stress to global dofs
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      strainGlobal[0][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[0][0]; // Exx
      strainGlobal[1][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[1][1]; // Eyy
      strainGlobal[3][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[0][1]; // Exy
      stressGlobal[0][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[0][0]; // Sxx
      stressGlobal[1][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[1][1]; // Syy
      stressGlobal[3][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[0][1]; // Sxy
      if (dim == 3)
      {
        strainGlobal[2][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[2][2]; // Ezz
        strainGlobal[4][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[0][1]; // Exy
        strainGlobal[5][scalarCell->vertex_dof_index(v, 0)] += cellAvgStrain[1][2]; // Eyz
        stressGlobal[2][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[2][2]; // Szz
        stressGlobal[4][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[0][1]; // Sxy
        stressGlobal[5][scalarCell->vertex_dof_index(v, 0)] += cellAvgStress[1][2]; // Syz
      }
      count[scalarCell->vertex_dof_index(v, 0)]++;
    }
  }
  // Global nodal average
  for (unsigned int i = 0; i < 6; ++i)
  {
    for (types::global_dof_index j = 0; j < scalarDofHandler.n_dofs(); ++j)
    {
      strainGlobal[i][j] /= count[j];
      stressGlobal[i][j] /= count[j];
    }
  }

  /*-------------------------------------------------------------------------------------*/
  // Output the strain, stress and displacements
  std::string fileName = "LinearElastic-";
  char id[50];
  snprintf(id, 50, "%d", cycle);
  fileName += id;
  fileName += ".vtu";
  std::ofstream out(fileName.c_str());
  DataOut<dim> data;

  data.add_data_vector(scalarDofHandler, strainGlobal[0], "Exx");
  data.add_data_vector(scalarDofHandler, strainGlobal[1], "Eyy");
  data.add_data_vector(scalarDofHandler, strainGlobal[3], "Exy");
  data.add_data_vector(scalarDofHandler, stressGlobal[0], "Sxx");
  data.add_data_vector(scalarDofHandler, stressGlobal[1], "Syy");
  data.add_data_vector(scalarDofHandler, stressGlobal[3], "Sxy");

  switch(dim)
  {
    case 2:
      data.add_data_vector(this->dofHandler, this->solution, vector<string>{"Ux", "Uy"});
      break;
    case 3:
      data.add_data_vector(scalarDofHandler, strainGlobal[2], "Ezz");
      data.add_data_vector(scalarDofHandler, strainGlobal[4], "Exy");
      data.add_data_vector(scalarDofHandler, strainGlobal[5], "Eyz");
      data.add_data_vector(scalarDofHandler, stressGlobal[2], "Szz");
      data.add_data_vector(scalarDofHandler, stressGlobal[4], "Sxy");
      data.add_data_vector(scalarDofHandler, stressGlobal[5], "Syz");
      data.add_data_vector(this->dofHandler, this->solution, vector<string>{"Ux", "Uy", "Uz"});
      break;
    default:
      Assert(false, ExcNotImplemented());
  }

  data.build_patches();
  data.write_vtu(out);
}

template<int dim>
void LinearElasticSolver<dim>::runStatics(const string& fileName)
{
  LinearMaterial<dim> steel(1., 1.);
  if (fileName.empty())
  {
    this->generateMesh();
  }
  else
  {
    this->readMesh(fileName);
  }
  this->setMaterial(steel);
  this->readBC();
  this->setup();
  this->assemble();
  this->applyBC(this->tangentStiffness, this->solution, this->sysRhs);
  this->output(0);
  this->solve(this->tangentStiffness, this->solution, this->sysRhs);
  this->output(1);
}

template<int dim>
void LinearElasticSolver<dim>::runDynamics(const string& fileName)
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
  LinearMaterial<dim> steel(1., 1.);
  this->setMaterial(steel);
  this->readBC();
  this->setup();
  this->assemble();
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
  this->mass *= this->material.getDensity();
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

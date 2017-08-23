template<int dim>
double Material<dim>::getLameFirst() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->lambda;
}

template<int dim>
double Material<dim>::getLameSecond() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->mu;
}

template<int dim>
double Material<dim>::getShearModulus() const
{
  return this->getLameSecond();
}

template<int dim>
double Material<dim>::getYoungsModulus() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->mu*(3*this->lambda + 2*this->mu)/(this->lambda + this->mu);
}

template<int dim>
double Material<dim>::getBulkModulus() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->lambda + 2*this->mu/3;
}

template<int dim>
double Material<dim>::getPoissonsRatio() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->lambda/(2*(this->lambda + this->mu));
}

template<int dim>
double Material<dim>::getDensity() const
{
  if (!this->initialized)
  {
    throw std::runtime_error("Material is not initialized!");
  }
  return this->density;
}

template<int dim>
void Material<dim>::print() const
{
  std::cout << "lambda = " << this->lambda << ", mu = " << this->mu
    << ", density = " << this->density << ", initialized = " 
    << this->initialized << std::endl;
}

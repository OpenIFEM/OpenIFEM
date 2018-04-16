#ifndef MOONEY_RIVLIN
#define MOONEY_RIVLIN

#include "hyperelasticMaterial.h"
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>


namespace Solid
{
  extern template class HyperelasticMaterial<2>;
  extern template class HyperelasticMaterial<3>;

  /*! \brief Mooney_Rivlin material.
   *
   * The isotropic part of the strain energy in the Neo-Hookean
   * model is written as \f$ C_1(\bar{I}_1 -3)+C_2(\bar{I}_2-3) \f$.
   */
  template <int dim>
  class MooneyRivlin : public HyperelasticMaterial<dim>
  {
    using ST = typename dealii::Physics::Elasticity::StandardTensors<dim>;

  public:
    MooneyRivlin() : HyperelasticMaterial<dim>(), c1(0.0), c2(0.0) {}
    MooneyRivlin(double param1, double param2, double param3,double rho=0.0)
      : HyperelasticMaterial<dim>(param2, rho), c1(param1), c2(param3)
    {
    }

    virtual dealii::SymmetricTensor<2, dim> get_tau_bar() const override
    {
      
    // Calculating d_psi/d_b_bar
       const dealii::SymmetricTensor<2, dim> temp1 = (this->c1*ST::I + this->c2 *(dealii::trace(this->b_bar)*ST::I - this->b_bar));
    // Multiplication by b_bar	
       dealii::SymmetricTensor<2, dim> temp2;
       
         for (unsigned int i=0; i<dim; ++i)
	 {
    	     for (unsigned int j=0; j<dim; ++j)
	     {
		temp2[i][j]=0;
		for(unsigned int k=0; k<dim; ++k)
		{
			temp2[i][j]+=this->b_bar[i][k]*temp1[k][j];
		}		
	     }
	  }   
     return 2*temp2;
     }

  

    virtual dealii::SymmetricTensor<4, dim> get_cc_bar() const override
    {
      //Calculating d2_psi/d_b_bar^2
        const dealii::SymmetricTensor<4, dim> temp1 = this->c2 * (ST::IxI - ST::S);
      //Calculating b_bar*d2_psi/d_b_bar^2*b_bar
        dealii::SymmetricTensor<4, dim> temp2;
	dealii::SymmetricTensor<4, dim> temp3;
	for (unsigned int i=0; i<dim; ++i)
	 {
    	     for (unsigned int j=0; j<dim; ++j)
	     {
                  for (unsigned int k=0; k<dim; ++k)
	 	  {
    	     	      for (unsigned int l=0; l<dim; ++l)
	     	      {
			  temp2[i][j][k][l]=0;
			  for(unsigned int m=0;m<dim;++m)
			  {
				temp2[i][j][k][l]+=this->b_bar[i][m]*temp1[m][j][k][l];
			  }
		      }
                  }
              }
         }
	 for (unsigned int i=0; i<dim; ++i)
	 {
    	     for (unsigned int j=0; j<dim; ++j)
	     {
                  for (unsigned int k=0; k<dim; ++k)
	 	  {
    	     	      for (unsigned int l=0; l<dim; ++l)
	     	      {
			  temp3[i][j][k][l]=0;
			  for(unsigned int m=0;m<dim;++m)
			  {
				temp3[i][j][k][l]+=temp2[i][j][k][m]*this->b_bar[m][l];
			  }
		      }
                  }
              }
         }
    return 4*temp3;

    }
   
  private:
    double c1;
    double c2;
  };
}

#endif

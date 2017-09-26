#ifndef PARAMETERS
#define PARAMETERS

#include <deal.II/base/parameter_handler.h>
#include <string>
#include <vector>
#include <iostream>

namespace IFEM
{
  namespace Parameters
  {
    struct FESystem
    {
      unsigned int polyDegree;
      unsigned int quadOrder;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct Geometry
    {
      unsigned int globalRefinement;
      double scale;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct LinearSolver
    {
      std::string typeLin;
      double tolLin;
      double maxItrLin;
      std::string typePre;
      double relaxPre;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct NonlinearSolver
    {
      unsigned int maxItrNL;
      double       tolF;
      double       tolU;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct Material
    {
      std::string typeMat;
      double rho;
      double E; // Linear elastic material only
      double nu; // Linear elastic material only
      std::vector<double> C; // Hyperelastic material only
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct Time
    {
      double endTime;
      double deltaTime;
      static void declareParameters(dealii::ParameterHandler &);
      void parseParameters(dealii::ParameterHandler &);
    };

    struct AllParameters : public FESystem, public Geometry,
      public LinearSolver, public NonlinearSolver, public Material, public Time
    {
      AllParameters(const std::string &);
      static void declareParameters(dealii::ParameterHandler &prm);
      void parseParameters(dealii::ParameterHandler &prm);
    };
  }
}

#endif

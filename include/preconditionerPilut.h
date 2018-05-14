#ifndef dealii_petsc_precondition_Pilut
#define dealii_petsc_precondition_Pilut

#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <petscconf.h>
#include <petscpc.h>

using namespace dealii;

class PreconditionPilut : public PETScWrappers::PreconditionerBase
{
public:
  /**
   * Standardized data struct to pipe additional flags to the
   * preconditioner.
   */
  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData(const unsigned int maxiter = 20,
                   const unsigned int factorrowsize = 20,
                   const double tolerance = 0.0001);

    /**
     */
    unsigned int maxiter;

    /**
     */
    unsigned int factorrowsize;

    /**
     */
    double tolerance;
  };

  /**
   * Empty Constructor. You need to call initialize() before using this
   * object.
   */
  PreconditionPilut() = default;

  /**
   * Constructor. Take the matrix which is used to form the preconditioner,
   * and additional flags if there are any.
   */
  PreconditionPilut(const PETScWrappers::MatrixBase &matrix,
                    const AdditionalData &additional_data = AdditionalData());

  /**
   * Initialize the preconditioner object and calculate all data that is
   * necessary for applying it in a solver. This function is automatically
   * called when calling the constructor with the same arguments and is only
   * used if you create the preconditioner without arguments.
   */
  void initialize(const PETScWrappers::MatrixBase &matrix,
                  const AdditionalData &additional_data = AdditionalData());

  friend PETScWrappers::MatrixBase;

private:
  /**
   * Store a copy of the flags for this particular preconditioner.
   */
  AdditionalData additional_data;
};

class PreconditionEuclid : public PETScWrappers::PreconditionerBase
{
public:
  /**
   * Empty Constructor. You need to call initialize() before using this
   * object.
   */
  PreconditionEuclid() = default;

  /**
   * Constructor. Take the matrix which is used to form the preconditioner,
   * and additional flags if there are any.
   */
  PreconditionEuclid(const PETScWrappers::MatrixBase &matrix);

  /**
   * Initialize the preconditioner object and calculate all data that is
   * necessary for applying it in a solver. This function is automatically
   * called when calling the constructor with the same arguments and is only
   * used if you create the preconditioner without arguments.
   */
  void initialize(const PETScWrappers::MatrixBase &matrix);

  friend PETScWrappers::MatrixBase;
};

#endif

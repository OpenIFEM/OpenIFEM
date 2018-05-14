#include "preconditionerPilut.h"
#include <_hypre_parcsr_ls.h>
#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
/* this include is needed ONLY to allow access to the private data inside the
 * Mat object specific to hypre */
#include <petsc/private/matimpl.h>

typedef struct
{
  HYPRE_Solver hsolver;
  Mat hpmat; /* MatHYPRE */

  HYPRE_Int (*destroy)(HYPRE_Solver);
  HYPRE_Int (*solve)(HYPRE_Solver,
                     HYPRE_ParCSRMatrix,
                     HYPRE_ParVector,
                     HYPRE_ParVector);
  HYPRE_Int (*setup)(HYPRE_Solver,
                     HYPRE_ParCSRMatrix,
                     HYPRE_ParVector,
                     HYPRE_ParVector);

  MPI_Comm comm_hypre;
  char *hypre_type;
  PetscBool printstatistics;
} MY_PC_HYPRE;

static PetscErrorCode
PCSetFromOptions_HYPRE_Euclid(PetscOptionItems *PetscOptionsObject, PC pc)
{
  MY_PC_HYPRE *jac = (MY_PC_HYPRE *)pc->data;
  PetscErrorCode ierr;
  PetscBool flag;
  char *args[8], levels[16];
  PetscInt cnt = 0;
  PetscBool bjilu = PETSC_FALSE;
  PetscInt fillin_levels = 0;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "HYPRE Euclid Options");
  CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_euclid_levels",
                         "Number of levels of fill ILU(k)",
                         "None",
                         fillin_levels,
                         &fillin_levels,
                         &flag);
  CHKERRQ(ierr);
  if (flag)
    {
      if (fillin_levels < 0)
        SETERRQ1(PetscObjectComm((PetscObject)pc),
                 PETSC_ERR_ARG_OUTOFRANGE,
                 "Number of levels %d must be nonegative",
                 levels);
      ierr = PetscSNPrintf(levels, sizeof(levels), "%D", fillin_levels);
      CHKERRQ(ierr);
      args[cnt++] = (char *)"-level";
      args[cnt++] = levels;
    }
  ierr = PetscOptionsBool("-pc_hypre_euclid_bj",
                          "Use block Jacobi ILU(k)",
                          "None",
                          bjilu,
                          &bjilu,
                          NULL);
  CHKERRQ(ierr);
  if (bjilu)
    {
      args[cnt++] = (char *)"-bj";
      args[cnt++] = (char *)"1";
    }

  ierr = PetscOptionsBool("-pc_hypre_euclid_print_statistics",
                          "Print statistics",
                          "None",
                          jac->printstatistics,
                          &jac->printstatistics,
                          NULL);
  CHKERRQ(ierr);
  if (jac->printstatistics)
    {
      args[cnt++] = (char *)"-eu_stats";
      args[cnt++] = (char *)"1";
      args[cnt++] = (char *)"-eu_mem";
      args[cnt++] = (char *)"1";
    }
  ierr = PetscOptionsTail();
  CHKERRQ(ierr);
  if (cnt)
    PetscStackCallStandard(HYPRE_EuclidSetParams, (jac->hsolver, cnt, args));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_HYPRE_Euclid(PC pc, PetscViewer viewer)
{
  (void)pc;
  PetscErrorCode ierr;
  PetscBool iascii;
  PetscInt levels = 0;
  PetscBool bjilu = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii);
  CHKERRQ(ierr);
  if (iascii)
    {
      ierr = PetscViewerASCIIPrintf(viewer, "  HYPRE Euclid preconditioning\n");
      CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(
        viewer, "  HYPRE Euclid: number of levels %d\n", levels);
      CHKERRQ(ierr);
      if (bjilu)
        {
          ierr = PetscViewerASCIIPrintf(
            viewer,
            "  HYPRE Euclid: Using block Jacobi ILU instead of parallel ILU\n");
          CHKERRQ(ierr);
        }
    }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHYPRESetType_Euclid(PC pc)
{
  MY_PC_HYPRE *jac = (MY_PC_HYPRE *)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HYPRE_EuclidCreate(jac->comm_hypre, &jac->hsolver);
  CHKERRQ(ierr);
  pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Euclid;
  pc->ops->view = PCView_HYPRE_Euclid;
  jac->destroy = HYPRE_EuclidDestroy;
  jac->setup = HYPRE_EuclidSetup;
  jac->solve = HYPRE_EuclidSolve;
  PetscFunctionReturn(0);
}

/* ----------------- PreconditionPilut ------------------------ */

PreconditionPilut::AdditionalData::AdditionalData(
  const unsigned int maxiter,
  const unsigned int factorrowsize,
  const double tolerance)
  : maxiter(maxiter), factorrowsize(factorrowsize), tolerance(tolerance)
{
}

PreconditionPilut::PreconditionPilut(const PETScWrappers::MatrixBase &matrix,
                                     const AdditionalData &additional_data)
{
  initialize(matrix, additional_data);
}

void PreconditionPilut::initialize(const PETScWrappers::MatrixBase &matrix_,
                                   const AdditionalData &additional_data_)
{
  clear();

  matrix = static_cast<Mat>(matrix_);
  additional_data = additional_data_;

  MPI_Comm comm = matrix_.get_mpi_communicator();

  PetscErrorCode ierr = PCCreate(comm, &pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetOperators(pc, matrix, matrix);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetType(pc, const_cast<char *>(PCHYPRE));
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCHYPRESetType(pc, "pilut");

  std::stringstream ssStream;

  PETScWrappers::set_option_value(
    "-pc_hypre_pilut_maxiter", Utilities::to_string(additional_data.maxiter));

  ssStream << additional_data.factorrowsize;
  PETScWrappers::set_option_value("-pc_hypre_pilut_factorrowsize",
                                  ssStream.str());

  ssStream.str(""); // empty the stringstream
  ssStream << additional_data.tolerance;
  PETScWrappers::set_option_value("-pc_hypre_pilut_tol", ssStream.str());

  ierr = PCSetFromOptions(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetUp(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));
}

/* ----------------- PreconditionEuclid ------------------------ */

PreconditionEuclid::PreconditionEuclid(const PETScWrappers::MatrixBase &matrix)
{
  initialize(matrix);
}

void PreconditionEuclid::initialize(const PETScWrappers::MatrixBase &matrix_)
{
  clear();

  matrix = static_cast<Mat>(matrix_);

  MPI_Comm comm = matrix_.get_mpi_communicator();

  PetscErrorCode ierr = PCCreate(comm, &pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetOperators(pc, matrix, matrix);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetType(pc, const_cast<char *>(PCHYPRE));
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCHYPRESetType_Euclid(pc);

  ierr = PCSetFromOptions(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));

  ierr = PCSetUp(pc);
  AssertThrow(ierr == 0, ExcPETScError(ierr));
}

/* ----------------- PreconditionEuclid ------------------------ */

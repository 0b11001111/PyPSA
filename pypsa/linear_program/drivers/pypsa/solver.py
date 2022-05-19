import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, FilePath

from pypsa.linear_program import LinearProgram
from pypsa.linear_program.solver import Solver, SolverStatus, TerminationCondition, Status, Solution, Result
from .linear_program import LinearProgramPypsa
from .solver_util import run_and_read

logger = logging.getLogger(__name__)


class SolverPypsa(Solver):
    basis_fn: Optional[FilePath] = Field(
        default=None,
        description="Use this to warmstart the optimization. Pass a path to "
                    "the basis file."
    )
    store_basis: bool = Field(
        default=True,
        description="Whether to store the basis of the optimization results. "
                    "Note that a basis can only be stored  if simplex, "
                    "dual-simplex, or barrier *with* crossover is  used for "
                    "solving."
    )
    solver_dir: Optional[Path] = Field(
        default=None,
        description="Path to directory where necessary files are written, "
                    "None leads to the default temporary directory used by "
                    "`tempfile.mkstemp()`."
    )

    def new_linear_program(self) -> LinearProgram:
        return LinearProgramPypsa(self.solver_dir, self.keep_files)

    def solve(self, lp: LinearProgramPypsa) -> Result:
        assert isinstance(lp, LinearProgramPypsa)

        with lp.temp_file(prefix="pypsa-problem-", suffix=".lp") as problem_f, \
                lp.temp_file(prefix="pypsa-solution-", suffix=".sol") as solution_f:
            lp.render(problem_f)
            problem_f.flush()

            # Solve problem
            logger.info(f"Solve linear problem using {self.solver_backend} (warmstart={self.basis_fn is not None}=")
            status_str, termination_condition_str, variables_sol, constraints_dual, objective = run_and_read(
                solver=self.solver_backend,
                basis_fn=self.basis_fn,
                problem_fn=problem_f.name,
                solution_fn=solution_f.name,
                solver_logfile=str(self.solver_logfile),
                solver_options=self.solver_options or {},
                store_basis=self.store_basis,
            )

        # Build solution
        status = Status.from_pypsa(status_str, termination_condition_str)
        if variables_sol is not None and constraints_dual is not None and objective is not None:
            solution = Solution(
                variables_sol=variables_sol,
                constraints_dual=constraints_dual,
                objective=objective
            )
        else:
            solution = None

        if status.status == SolverStatus.ok and status.termination_condition == TerminationCondition.optimal:
            logger.info(f"Optimization successful. Objective value: {objective:.2e}")
        elif status == SolverStatus.ok and status.termination_condition != TerminationCondition.optimal:
            logger.warning(
                "Optimization solution is sub-optimal. "
                f"Objective value: {objective:.2e}"
            )
        else:
            logger.warning(
                f"Optimization failed with status {status.status} and "
                f"termination condition {status.termination_condition}"
            )

        return Result(status=status, solution=solution)

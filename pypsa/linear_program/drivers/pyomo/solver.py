import logging
from typing import Optional, Set

from pydantic import Field, validator

from pypsa.linear_program import LinearProgram
from pypsa.linear_program.solver import Solver
from .linear_program import LinearProgramPyomo

logger = logging.getLogger(__name__)


class SolverPyomo(Solver):
    free_memory: Set[str] = Field(
        default_factory=lambda: {"pyomo"},
        description="Any subset of {'pypsa', 'pyomo'}. Allows to stash `pypsa` "
                    "time-series data away while the solver runs (as a pickle "
                    "to disk) and/or free `pyomo` data after the solution has "
                    "been extracted."
    )
    solver_io: Optional[str] = Field(
        default=None,
        description="Solver Input-Output option, e.g. 'python' to use "
                    "'gurobipy' for solver_backend='gurobi'"
    )

    @validator("free_memory", pre=True)
    def _validate_free_memory(cls, value):
        allowed_superset = {"pypsa", "pyomo"}
        if not isinstance(value, set):
            value = set(value)
        if not set.issubset(value, allowed_superset):
            raise ValueError(f"{value} is no subset of {allowed_superset}")
        return value

    def new_linear_program(self) -> LinearProgram:
        return LinearProgramPyomo()

    def solve(self, lp: LinearProgram):
        raise NotImplementedError

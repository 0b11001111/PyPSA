from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from pyomo.opt import TerminationCondition as PyomoCondition, SolverStatus as PyomoStatus

from pypsa.descriptors import Dict
from .base import LinearProgram


class SolverBackend(Enum):
    highs = "highs"
    cbc = "cbc"
    gurobi = "gurobi"
    glpk = "glpk"
    cplex = "cplex"
    xpress = "xpress"


LegacyStatus = Union[Tuple[str, str], Tuple[PyomoStatus, PyomoCondition]]


class SolverStatus(Enum):
    """Compatible with `pyomo.opt.SolverStatus`"""
    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"


class TerminationCondition(Enum):
    """Compatible with `pyomo.opt.TerminationCondition`"""
    # UNKNOWN
    unknown = "unknown"

    # OK
    max_time_limit = "maxTimeLimit"
    max_iterations = "maxIterations"
    min_function_value = "minFunctionValue"
    min_step_length = "minStepLength"
    globally_optimal = "globallyOptimal"
    locally_optimal = "locallyOptimal"
    feasible = "feasible"
    optimal = "optimal"
    max_evaluations = "maxEvaluations"
    other = "other"

    # WARNING
    unbounded = "unbounded"
    infeasible = "infeasible"
    infeasible_or_unbounded = "infeasibleOrUnbounded"
    invalid_problem = "invalidProblem"
    intermediate_non_integer = "intermediateNonInteger"
    no_solution = "noSolution"

    # ERROR
    solver_failure = "solverFailure"
    internal_solver_error = "internalSolverError"
    error = "error"

    # ABORTED
    user_interrupt = "userInterrupt"
    resource_interrupt = "resourceInterrupt"
    licensing_problems = "licensingProblems"

    def to_solver_status(self) -> SolverStatus:
        tc = PyomoCondition(self.value)
        status = PyomoCondition.to_solver_status(tc)
        return SolverStatus(status.value)

    @classmethod
    def from_pypsa(cls, termination_condition: str) -> "TerminationCondition":
        return cls(termination_condition)

    @classmethod
    def from_pyomo(cls, termination_condition: PyomoCondition) -> "TerminationCondition":
        return cls(termination_condition.value)


class Status(BaseModel):
    legacy_status: LegacyStatus
    status: SolverStatus
    termination_condition: TerminationCondition

    @classmethod
    def from_pypsa(cls, status: str, termination_condition: str) -> "Status":
        tc = TerminationCondition.from_pypsa(termination_condition)
        return cls(
            legacy_status=(status, termination_condition),
            status=tc.to_solver_status(),
            termination_condition=tc
        )

    @classmethod
    def from_pyomo(cls, status: PyomoStatus, termination_condition: PyomoCondition) -> "Status":
        tc = TerminationCondition(termination_condition.value)
        return cls(
            legacy_status=(status, termination_condition),
            status=tc.to_solver_status(),
            termination_condition=tc
        )


class Solution(BaseModel):
    variables_sol: pd.Series
    constraints_dual: pd.Series
    objective: float

    class Config:
        arbitrary_types_allowed = True


class Result(BaseModel):
    status: Status
    solution: Optional[Solution] = None


class Solver(BaseModel, metaclass=ABCMeta):
    solver_backend: SolverBackend = Field(
        default=SolverBackend.glpk,
        description="The solver backend to use",
        alias="solver_name"
    )
    solver_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional options that get passed to the solver. "
                    "(e.g. {'threads':2} tells gurobi to use only 2 cpus)"
    )
    solver_logfile: Optional[Path] = Field(
        default=None,
        description="If not None, sets the logfile option of the solver"
    )
    keep_files: bool = Field(
        default=False,
        description="Keep the files that pyomo constructs from OPF problem "
                    "construction, e.g. .lp file - useful for debugging"
    )

    @abstractmethod
    def new_linear_program(self) -> LinearProgram:
        raise NotImplementedError

    @abstractmethod
    def solve(self, lp: LinearProgram):
        raise NotImplementedError

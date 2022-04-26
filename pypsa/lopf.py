from abc import ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator
from pyomo.opt import SolverStatus as PyomoStatus, TerminationCondition as PyomoCondition

from pypsa.components import Network
from pypsa.linopf import network_lopf as network_lopf_native
from pypsa.opf import network_lopf as network_lopf_pyomo


class SolverBackend(Enum):
    highs = "highs"
    cbc = "cbc"
    gurobi = "gurobi"
    glpk = "glpk"
    cplex = "cplex"
    xpress = "xpress"


class Formulation(Enum):
    angles = "angles"
    cycles = "cycles"
    kirchoff = "kirchhoff"
    ptdf = "ptdf"


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
    def from_native(cls, termination_condition: str) -> "TerminationCondition":
        return cls(termination_condition)

    @classmethod
    def from_pyomo(cls, termination_condition: PyomoCondition) -> "TerminationCondition":
        return cls(termination_condition.value)


LegacyResult = Union[Tuple[str, str], Tuple[PyomoStatus, PyomoCondition]]


class Result(BaseModel):
    legacy_result: LegacyResult
    status: SolverStatus
    termination_condition: TerminationCondition

    @classmethod
    def from_native(cls, status: str, termination_condition: str) -> "Result":
        tc = TerminationCondition.from_native(termination_condition)
        return cls(
            legacy_result=(status, termination_condition),
            status=tc.to_solver_status(),
            termination_condition=tc
        )

    @classmethod
    def from_pyomo(cls, status: PyomoStatus, termination_condition: PyomoCondition) -> "Result":
        tc = TerminationCondition(termination_condition.value)
        return cls(
            legacy_result=(status, termination_condition),
            status=tc.to_solver_status(),
            termination_condition=tc
        )


Snapshots = Union[pd.Index, pd.MultiIndex, pd.DatetimeIndex, pd.RangeIndex]
Duals = pd.Series
ExtraFunctionality = Callable[[Network, Snapshots], None]
PostProcessor = Callable[[Network, Snapshots, Duals], None]


class _LinearOptimalPowerFlow(BaseModel, metaclass=ABCMeta):
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
    formulation: Formulation = Field(
        default=Formulation.kirchoff,
        description="Formulation of the linear power flow equations to use"
    )
    extra_functionality: Optional[ExtraFunctionality] = Field(
        default=None,
        description="This function must take two arguments "
                    "`extra_functionality(network, snapshots)` and is called "
                    "after the model building is complete, but before it is "
                    "sent to the solver. It allows the user to add/change "
                    "constraints and add/change the objective function."
    )
    multi_investment_periods: bool = Field(
        default=False,
        description="Whether to optimise as a single investment period or to "
                    "optimise in multiple investment periods. Then, snapshots "
                    "should be a ``pd.MultiIndex``."
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    def legacy_args(self, **kwargs):
        args = self.dict()
        args["solver_name"] = args.pop("solver_backend").value
        args["formulation"] = args["formulation"].value
        args.update(kwargs)
        return args

    @staticmethod
    @abstractmethod
    def legacy_solver() -> Callable:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def legacy_result_parser() -> Callable[..., Result]:
        raise NotImplementedError

    def solve(self, network: Network, snapshots: Optional[Snapshots] = None) -> Result:
        if snapshots is None:
            snapshots = network.snapshots
        session = _LinearOptimalPowerFlowSession(network=network, snapshots=snapshots, config=self)
        return session.solve()


class _LinearOptimalPowerFlowSession(BaseModel):
    network: Network
    snapshots: Snapshots
    config: _LinearOptimalPowerFlow

    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def _validate_snapshots(cls, values):
        network: Network = values["network"]
        snapshots: Snapshots = values["snapshots"]

        if not snapshots.isin(network.snapshots).all():
            raise ValueError(f"{snapshots} is no subset of {network.snapshots}")

        if values.get("multi_investment_periods", False) and not isinstance(snapshots, pd.MultiIndex):
            raise ValueError(f"if `multi_investment_periods=True` `snapshots` must be of type `pandas.MultiIndex`")

        return values

    def solve(self) -> Result:
        solver = self.config.legacy_solver()
        result_parser = self.config.legacy_result_parser()
        args = self.config.legacy_args(snapshots=self.snapshots)
        status, termination_condition = solver(self.network, **args)
        return result_parser(status, termination_condition)


class LinearOptimalPowerFlowNative(_LinearOptimalPowerFlow):
    skip_objective: bool = Field(
        default=False,
        description="Skip writing the default objective function. If False, a "
                    "custom objective has to be defined via extra_functionality."
    )
    warmstart: Optional[Path] = Field(
        default=None,
        description="Use this to warmstart the optimization. Pass a string "
                    "which gives the path to the basis file. If set to True, a "
                    "path to a basis file must be given in network.basis_fn."
    )
    store_basis: bool = Field(
        default=True,
        description="Whether to store the basis of the optimization results. "
                    "If True, the path to the basis file is saved in "
                    "`network.basis_fn`. Note that a basis can only be stored "
                    "if simplex, dual-simplex, or barrier *with* crossover is "
                    "used for solving."
    )
    keep_references: bool = Field(
        default=False,
        description="Keep the references of variable and constraint names "
                    "withing the network. These can be looked up in `n.vars` "
                    "and `n.cons` after solving."
    )
    keep_shadowprices: Union[bool, List[str]] = Field(
        default_factory=lambda: ['Bus', 'Line', 'GlobalConstraint'],
        description="Keep the references of variable and constraint names "
                    "withing the network. These can be looked up in `n.vars` "
                    "and `n.cons` after solving."
    )
    solver_dir: Optional[Path] = Field(
        default=None,
        description="Path to directory where necessary files are written, "
                    "default None leads to the default temporary directory "
                    "used by `tempfile.mkstemp()`."
    )

    @staticmethod
    def legacy_solver() -> Callable:
        return network_lopf_native

    @staticmethod
    def legacy_result_parser() -> Callable:
        return Result.from_native


class LinearOptimalPowerFlowPyomo(_LinearOptimalPowerFlow):
    ptdf_tolerance: float = Field(
        default=0.0,
        description="Value below which PTDF entries are ignored"
    )
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
    extra_postprocessing: Optional[PostProcessor] = Field(
        default=None,
        description="This function must take three arguments "
                    "`extra_postprocessing(network,snapshots,duals)` and is "
                    "called after the model has solved and the results are "
                    "extracted. It allows the user to extract further "
                    "information about the solution, such as additional "
                    "shadow prices."
    )

    @validator("free_memory", pre=True)
    def _validate_free_memory(cls, value):
        allowed_superset = {"pypsa", "pyomo"}
        if not isinstance(value, set):
            value = set(value)
        if not set.issubset(value, allowed_superset):
            raise ValueError(f"{value} is no subset of {allowed_superset}")
        return value

    @staticmethod
    def legacy_solver() -> Callable:
        return network_lopf_pyomo

    @staticmethod
    def legacy_result_parser() -> Callable:
        return Result.from_pyomo


LinearOptimalPowerFlow = LinearOptimalPowerFlowPyomo

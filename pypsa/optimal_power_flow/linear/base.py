import logging
from abc import ABCMeta
from enum import Enum
from typing import List, Union

import pandas as pd
from pydantic import BaseModel, Field, root_validator

from pypsa.components import Network
from pypsa.linear_program import LinearProgram
from pypsa.linear_program.solver import Result
from pypsa.pf import _as_snapshots
from .type_definitions import Snapshots
from .util import assign_solution, build_lopf_problem

logger = logging.getLogger(__name__)


class Formulation(Enum):
    angles = "angles"
    cycles = "cycles"
    kirchoff = "kirchhoff"
    ptdf = "ptdf"


class LinearOptimalPowerFlow(BaseModel, metaclass=ABCMeta):
    network: Network
    snapshots: Snapshots
    formulation: Formulation = Field(
        default=Formulation.kirchoff,
        description="Formulation of the linear power flow equations to use"
    )
    multi_investment_periods: bool = Field(
        default=False,
        description="Whether to optimise as a single investment period or to "
                    "optimise in multiple investment periods. Then, snapshots "
                    "should be a ``pd.MultiIndex``."
    )
    skip_pre: bool = Field(
        default=False,
        description="Skip the preliminary steps of computing topology, "
                    "calculating dependent values and finding bus controls."
    )
    skip_objective: bool = Field(
        default=False,
        description="Skip writing the default objective function. If False, a "
                    "custom objective has to be defined via extra_functionality."
    )
    keep_references: bool = Field(
        default=False,
        description="Keep the references of variable and constraint names "
                    "withing the network. These can be looked up in `n.vars` "
                    "and `n.cons` after solving."
    )
    keep_shadowprices: Union[bool, List[str]] = Field(
        default_factory=lambda: ["Bus", "Line", "Transformer", "Link", "GlobalConstraint"],
        description="Keep the references of variable and constraint names "
                    "withing the network. These can be looked up in `n.vars` "
                    "and `n.cons` after solving."
    )
    ptdf_tolerance: float = Field(
        default=0.0,
        description="Value below which PTDF entries are ignored"
    )

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def _validate_snapshots(cls, values):
        network: Network = values["network"]
        snapshots: Snapshots = _as_snapshots(network, values["snapshots"])

        if not snapshots.isin(network.snapshots).all():
            raise ValueError(f"{snapshots} is no subset of {network.snapshots}")

        if values.get("multi_investment_periods", False) and not isinstance(snapshots, pd.MultiIndex):
            raise ValueError(f"if `multi_investment_periods=True` `snapshots` must be of type `pandas.MultiIndex`")

        return values

    def build_problem(self, lp: LinearProgram):
        if not self.network.shunt_impedances.empty:
            logger.warning(
                "You have defined one or more shunt impedances. "
                "Shunt impedances are ignored by the linear optimal "
                "power flow (LOPF)."
            )

        if self.network.generators.committable.any():
            logger.warning(
                "Unit commitment is not yet completely implemented for "
                "optimising without pyomo. Thus minimum up time, minimum down time, "
                "start up costs, shut down costs will be ignored."
            )

        # TODO remove
        if self.skip_objective:
            logger.warning(
                "Skipping construction of objective without providing an alternative"
                "via `extra_functionality` will lead to an invalid problem formulation!"
            )

        if self.multi_investment_periods:
            logger.info("Perform multi-investment optimization.")
            assert not self.network.investment_periods.empty, "No investment periods defined."
            assert (
                self.snapshots.levels[0].difference(self.network.investment_periods).empty
            ), "Not all first-level snapshots values in investment periods."

        if self.formulation != Formulation.kirchoff:
            raise ValueError("As of now, the PyPsa driver supports Kirchoff formulation only!")

        if not self.skip_pre:
            self.network.calculate_dependent_values()
            self.network.determine_network_topology()

        build_lopf_problem(
            lp=lp,
            network=self.network,
            snapshots=self.snapshots,
            skip_objective=self.skip_objective,
            multi_invest=self.multi_investment_periods,
        )

    def assign_solution(self, lp: LinearProgram, result: Result):
        if not result.solution:
            raise ValueError("Result does not contain a solution")

        assign_solution(
            lp=lp,
            network=self.network,
            snapshots=self.snapshots,
            solution=result.solution,
            keep_references=self.keep_references,
            keep_shadowprices=self.keep_shadowprices,
            multi_invest=self.multi_investment_periods
        )

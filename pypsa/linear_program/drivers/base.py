from enum import Enum

from pypsa.linear_program.solver import Solver
from .pyomo import SolverPyomo
from .pypsa import SolverPypsa


class Driver(Enum):
    pyomo = SolverPyomo
    pypsa = SolverPypsa

    def new_solver(self, **kwargs) -> Solver:
        return self.value(**kwargs)

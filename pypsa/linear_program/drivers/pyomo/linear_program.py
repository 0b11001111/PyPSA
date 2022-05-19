import logging
from typing import Optional

from pypsa.linear_program import LinearProgram
from pypsa.linear_program.util import Reference, Bound, Axes, Mask, Terms, Sense

logger = logging.getLogger(__name__)


class LinearProgramPyomo(LinearProgram):
    def _write_bound(
            self,
            lower: Bound,
            upper: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Mask] = None
    ) -> Reference:
        raise NotImplementedError

    def _write_constraint(
            self,
            lhs: Bound,
            sense: Sense,
            rhs: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Axes] = None
    ) -> Reference:
        raise NotImplementedError

    def _write_binary(self, axes: Axes, mask: Optional[Mask] = None) -> Reference:
        raise NotImplementedError

    def _write_objective(self, terms: Terms):
        raise NotImplementedError

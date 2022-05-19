import logging
import sys
from contextlib import ExitStack
from io import FileIO, StringIO
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Generator, Optional

import numpy as np
import pandas as pd

from pypsa.linear_program import LinearProgram
from pypsa.linear_program.util import Reference, Bound, Axes, Mask, Terms, Sense, str_array, \
    to_pandas, join_expressions, _broadcasted_axes

logger = logging.getLogger(__name__)


def get_handlers(axes, *maybearrays):
    axes = [axes] if isinstance(axes, pd.Index) else axes
    if axes is None:
        axes, shape = _broadcasted_axes(*maybearrays)
    else:
        shape = tuple(map(len, axes))
    size = np.prod(shape)
    return axes, shape, size


class LinearProgramPypsa(LinearProgram):
    def __init__(
            self,
            working_dir: Optional[Path] = None,
            keep_files: bool = False
    ):
        super().__init__()

        if working_dir and not working_dir.is_dir():
            raise ValueError(f"{working_dir} is no directory")

        self._temp_file_defaults = dict(mode="wt+", suffix=".tmp", dir=working_dir, delete=not keep_files)
        self._exit_stack = ExitStack().__enter__()

        self._c_counter: int = 1
        self._x_counter: int = 1

        # Auxiliary files
        self._objective_f = self._exit_stack.enter_context(self.temp_file(prefix="pypsa-objective-"))
        self._constraints_f = self._exit_stack.enter_context(self.temp_file(prefix="pypsa-constraints-"))
        self._bounds_f = self._exit_stack.enter_context(self.temp_file(prefix="pypsa-bounds-"))
        self._binaries_f = self._exit_stack.enter_context(self.temp_file(prefix="pypsa-binaries-"))

    def _on_close(self):
        self._exit_stack.close()

    def temp_file(self, **kwargs):
        defaults = self._temp_file_defaults.copy()
        defaults.update(kwargs)
        return NamedTemporaryFile(**defaults)

    def _concat_aux_files(self) -> Generator[StringIO, None, None]:
        yield StringIO("\\* Linear Optimal Power Flow *\\\n\nmin\nobj:\n")
        yield self._objective_f
        yield StringIO("\n\ns.t.\n\n")
        yield self._constraints_f
        yield StringIO("\nbounds\n")
        yield self._bounds_f
        yield StringIO("\nbinary\n")
        yield self._binaries_f
        yield StringIO("end\n")

    def render(self, target: FileIO = sys.stdout):
        """Render .lp format to given file buffer, or, by default, stdout"""
        for file in self._concat_aux_files():
            pos = file.tell()
            file.seek(0)
            copyfileobj(file, target)
            file.seek(pos)

    def _write_bound(
            self,
            lower: Bound,
            upper: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Mask] = None
    ) -> Reference:
        axes, shape, size = get_handlers(axes, lower, upper)
        if not size:
            return pd.Series(dtype=float)
        self._x_counter += size
        variables = np.arange(self._x_counter - size, self._x_counter).reshape(shape)
        lower, upper = str_array(lower), str_array(upper)
        exprs = lower + " <= x" + str_array(variables, True) + " <= " + upper + "\n"
        if mask is not None:
            exprs = np.where(mask, exprs, "")
            variables = np.where(mask, variables, -1)
        self._bounds_f.write(join_expressions(exprs))
        return to_pandas(variables, *axes)

    def _write_constraint(
            self,
            lhs: Bound,
            sense: Sense,
            rhs: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Axes] = None
    ) -> Reference:
        axes, shape, size = get_handlers(axes, lhs, sense, rhs)
        if not size:
            return pd.Series()
        self._c_counter += size
        cons = np.arange(self._c_counter - size, self._c_counter).reshape(shape)
        if isinstance(sense, str):
            sense = "=" if sense == "==" else sense
        lhs, sense, rhs = str_array(lhs), str_array(sense), str_array(rhs)
        exprs = "c" + str_array(cons, True) + ":\n" + lhs + sense + " " + rhs + "\n\n"
        if mask is not None:
            exprs = np.where(mask, exprs, "")
            cons = np.where(mask, cons, -1)
        self._constraints_f.write(join_expressions(exprs))
        return to_pandas(cons, *axes)

    def _write_binary(self, axes: Axes, mask: Optional[Mask] = None) -> Reference:
        axes, shape, size = get_handlers(axes)
        self._x_counter += size
        variables = np.arange(self._x_counter - size, self._x_counter).reshape(shape)
        expressions = "x" + str_array(variables, True) + "\n"
        if mask is not None:
            expressions = np.where(mask, expressions, "")
            variables = np.where(mask, variables, -1)
        self._binaries_f.write(join_expressions(expressions))
        return to_pandas(variables, *axes)

    def _write_objective(self, terms: Terms):
        self._objective_f.write(join_expressions(terms))

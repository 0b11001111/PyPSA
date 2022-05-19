from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from pypsa.descriptors import Dict as NamespaceDict
from .util import Reference, Bound, Axes, Mask, Terms, Sense


def _add_reference(ref_dict: NamespaceDict, df: Reference, attr: str, pnl: bool = True):
    if pnl:
        if attr in ref_dict.pnl:
            ref_dict.pnl[attr].loc[df.index, df.columns] = df
        else:
            ref_dict.pnl[attr] = df
    else:
        if attr in ref_dict.df:
            ref_dict.df = pd.concat([ref_dict.df, df.to_frame(attr)])
        else:
            ref_dict.df[attr] = df


class LinearProgram(ABC):
    def __init__(self):
        self._is_open = True

        # Model indexes
        cols = ["component", "name", "pnl", "specification"]
        self.variables = pd.DataFrame(columns=cols).set_index(cols[:2])
        self.constraints = pd.DataFrame(columns=cols).set_index(cols[:2])
        self.vars = NamespaceDict()
        self.cons = NamespaceDict()

        # Solution variables
        self.solutions = pd.DataFrame()
        self.sols = dict()
        self.dual_values = pd.DataFrame()
        self.duals = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def _on_close(self):
        pass

    def close(self):
        if self.is_open:
            self._on_close()
            self._is_open = False

    # ==========================================================================
    # writing functions
    # ==========================================================================

    @abstractmethod
    def _write_bound(
            self,
            lower: Bound,
            upper: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Mask] = None
    ) -> Reference:
        """
        Writer function for writing out multiple variables at a time.

        If lower and upper are floats it demands to give pass axes, a tuple
        of (index, columns) or (index), for creating the variable of same
        upper and lower bounds. Return a series or frame with variable
        references.
        """
        raise NotImplementedError

    @abstractmethod
    def _write_constraint(
            self,
            lhs: Bound,
            sense: Sense,
            rhs: Bound,
            axes: Optional[Axes] = None,
            mask: Optional[Axes] = None
    ) -> Reference:
        """
        Writer function for writing out multiple constraints.

        If lower and upper are numpy.ndarrays its axes must not be None but a
        tuple of (index, columns) or (index). Return a series or frame with
        constraint references.
        """
        raise NotImplementedError

    @abstractmethod
    def _write_binary(self, axes: Axes, mask: Optional[Mask] = None) -> Reference:
        """
        Writer function for writing out multiple binary-variables at a time.

        According to the axes it writes out binaries for each entry the
        pd.Series or pd.DataFrame spanned by axes. Returns a series or frame
        with variable references.
        """
        raise NotImplementedError

    @abstractmethod
    def _write_objective(self, terms: Terms):
        """
        Writer function for writing out one or multiple objective terms.
        """
        raise NotImplementedError

    # ==========================================================================
    #  references to variables and constraints
    # ==========================================================================

    def set_variable_references(self, variables: Reference, name: str, attr: str, spec: str = ""):
        """
        Sets variable references. One-dimensional variable references will be
        collected at `self.vars[name].df`, two-dimensional variables in
        `self.vars[name].pnl`.
        """
        if not variables.empty:
            pnl = variables.ndim == 2
            if name not in self.variables.index:
                self.vars[name] = NamespaceDict(df=pd.DataFrame(), pnl=NamespaceDict())
            if ((name, attr) in self.variables.index) and (spec != ""):
                self.variables.at[pd.IndexSlice[name, attr], "specification"] += ", " + spec
            else:
                self.variables.loc[pd.IndexSlice[name, attr], :] = [pnl, spec]
            _add_reference(self.vars[name], variables, attr, pnl=pnl)

    def set_constraint_references(self, constraints: Reference, name: str, attr: str, spec: str = ""):
        """
        Sets constraint references. One-dimensional constraint references will
        be collected at `self.cons[c].df`, two-dimensional in `self.cons[c].pnl`.
        """
        if not constraints.empty:
            pnl = constraints.ndim == 2
            if name not in self.constraints.index:
                self.cons[name] = NamespaceDict(df=pd.DataFrame(), pnl=NamespaceDict())
            if ((name, attr) in self.constraints.index) and (spec != ""):
                self.constraints.at[pd.IndexSlice[name, attr], "specification"] += ", " + spec
            else:
                self.constraints.loc[pd.IndexSlice[name, attr], :] = [pnl, spec]
            _add_reference(self.cons[name], constraints, attr, pnl=pnl)

    def get_variable_references(self, name: str, attr: str, pop: bool = False) -> Reference:
        """
        Retrieves variable references for a given static or time-depending
        attribute of a given component. The function looks into `self.variables`
        to detect whether the variable is a time-dependent or static.
        """
        variables = self.vars[name].pnl if self.variables.pnl[name, attr] else self.vars[name].df
        return variables.pop(attr) if pop else variables[attr]

    def get_constraint_references(self, name: str, attr: str, pop: bool = False) -> Reference:
        """
        Retrieves constraint references.
        """
        constraints = self.cons[name].pnl if self.constraints.pnl[name, attr] else self.cons[name].df
        return constraints.pop(attr) if pop else constraints[attr]

    def get_solution(self, name: str, attr: str = "") -> Reference:
        """
        Retrieves solution for a given variable.
        Note that a lookup of all stored solutions is given in `self.solutions`.
        """
        pnl = self.solutions.at[(name, attr), "pnl"]
        return self.sols[name].pnl[attr] if pnl else self.sols[name].df[attr]

    def get_dual(self, name, attr="") -> Reference:
        """
        Retrieves shadow price for a given constraint. Note that for retrieving
        shadow prices of a custom constraint, its name has to be passed to
        `keep_references` in the lp, or `keep_references` has to be set to True.
        Note that a lookup of all stored shadow prices is given in
        `self.dualvalues`.
        """
        pnl = self.dual_values.at[(name, attr), "pnl"]
        return self.duals[name].pnl[attr] if pnl else self.duals[name].df[attr]

    # ==========================================================================
    # model building
    # ==========================================================================

    def define_variables(
            self,
            lower: Bound,
            upper: Bound,
            name: Optional[str] = None,
            attr: Optional[str] = None,
            spec: Optional[str] = None,
            axes: Optional[Axes] = None,
            mask: Optional[Mask] = None,
    ) -> Reference:
        """
        Defines variable(s) with given lower and upper bound(s). The variables
        are stored under `self.vars` with key of the variable name. If multiple
        variables are defined at ones, at least one of lower and upper has to be
        an array like of `shape > (1,)` or axes have to define the dimensions of
        the variables.
        """
        variables = self._write_bound(lower, upper, axes, mask)
        if name is not None:
            self.set_variable_references(variables, name, attr or "", spec or "")
        return variables

    def define_binaries(
            self,
            axes: Axes,
            name: Optional[str] = None,
            attr: Optional[str] = None,
            spec: Optional[str] = None,
            mask: Optional[Mask] = None
    ) -> Reference:
        """
        Defines binary-variable(s). The variables are stored under `self.vars`
        with key of the variable name. For each entry for the `pd.Series` of
        `pd.DataFrame` spanned by the `axes` argument the function defines a
        binary.
        """
        binaries = self._write_binary(axes, mask)
        if name is not None:
            self.set_variable_references(binaries, name, attr or "", spec or "")
        return binaries

    def define_constraints(
            self,
            lhs: Bound,
            sense: Sense,
            rhs: Bound,
            name: Optional[str] = None,
            attr: Optional[str] = None,
            spec: Optional[str] = None,
            axes: Optional[Axes] = None,
            mask: Optional[Mask] = None
    ) -> Reference:
        """
        Defines constraint(s) with given left-hand side (lhs), sense and
        right-hand side (rhs). The constraints are stored in the network object
        under `self.cons` with key of the constraint name. If multiple
        constraints are defined at ones, only using `np.arrays`, then the axes
        argument can be used for defining the axes for the constraints (this is
        especially recommended for time-dependent constraints). If one of lhs,
        sense and rhs is a pd.Series/pd.DataFrame the axes argument is not
        necessary.
        """
        constraints = self._write_constraint(lhs, sense, rhs, axes, mask)
        if name is not None:
            self.set_constraint_references(constraints, name, attr or "", spec or "")
        return constraints

    def define_objective(self, terms: Terms):
        """
        Define one or multiple objective terms.
        """
        self._write_objective(terms)

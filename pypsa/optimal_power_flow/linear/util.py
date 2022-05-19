# -*- coding: utf-8 -*-

## Copyright 2015-2021 PyPSA Developers

## You can find the list of PyPSA Developers at
## https://pypsa.readthedocs.io/en/latest/developers.html

## PyPSA is released under the open source MIT License, see
## https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt

"""
Build optimisation problems from PyPSA networks without Pyomo.

Originally retrieved from nomopyomo ( -> 'no more Pyomo').
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2021 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
import os
import re
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from numpy import inf

import pypsa
from pypsa.components import Network
from pypsa.descriptors import Dict, additional_linkports, expand_series, get_active_assets, get_activity_mask, \
    get_bounds_pu, get_extendable_i, get_non_extendable_i, nominal_attrs
from pypsa.linear_program import LinearProgram
from pypsa.linear_program.solver import Solution
from pypsa.linear_program.util import linear_expression, join_expressions
from pypsa.optimal_power_flow.linear.type_definitions import Snapshots
from pypsa.pf import get_switchable_as_dense as get_as_dense

pd_version = LooseVersion(pd.__version__)
agg_group_kwargs = dict(numeric_only=False) if pd_version >= "1.3" else {}

logger = logging.getLogger(__name__)

lookup = pd.read_csv(
    os.path.join(os.path.dirname(pypsa.__file__), "variables.csv"),
    index_col=["component", "variable"],
)


def align_with_static_component(
        lp: LinearProgram,
        network: Network,
        component: str,
        attr: str
):
    """
    Alignment of time-dependent variables with static components.

    If c is a pypsa.component name, it will sort the columns of the
    variable according to the static component.
    """
    if component in network.all_components and (component, attr) in lp.variables.index:
        if not lp.variables.pnl[component, attr]:
            return
        if len(lp.vars[component].pnl[attr].columns) != len(network.df(component).index):
            return
        lp.vars[component].pnl[attr] = lp.vars[component].pnl[attr].reindex(
            columns=network.df(component).index)


def define_nominal_for_extendable_variables(
        lp: LinearProgram,
        network: Network,
        component: str,
        attr: str
):
    """
    Initializes variables for nominal capacities for a given component and a
    given attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : str
        network component of which the nominal capacity should be defined
    attr : str
        name of the variable, e.g. 'p_nom'
    """
    ext_i = get_extendable_i(network, component)
    if ext_i.empty:
        return
    lower = network.df(component)[attr + "_min"][ext_i]
    upper = network.df(component)[attr + "_max"][ext_i]
    lp.define_variables(lower, upper, component, attr)


def define_dispatch_for_extendable_and_committable_variables(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        attr: str,
        multi_invest: bool,
):
    """
    Initializes variables for power dispatch for a given component and a given
    attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    ext_i = get_extendable_i(network, component)
    if component == "Generator":
        ext_i = ext_i.union(network.generators.query("committable").index)
    if ext_i.empty:
        return
    active = get_activity_mask(network, component, snapshots, multi_invest)[ext_i] if multi_invest else None
    lp.define_variables(-inf, inf, component, attr, spec="ext", axes=[snapshots, ext_i], mask=active)


def define_dispatch_for_non_extendable_variables(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        attr: str,
        multi_invest: bool,
):
    """
    Initializes variables for power dispatch for a given component and a given
    attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    fix_i = get_non_extendable_i(network, component)
    if component == "Generator":
        fix_i = fix_i.difference(network.generators.query("committable").index)
    if fix_i.empty:
        return
    nominal_fix = network.df(component)[nominal_attrs[component]][fix_i]
    min_pu, max_pu = get_bounds_pu(network, component, snapshots, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)
    axes = [snapshots, fix_i]

    active = get_activity_mask(network, component, snapshots, multi_invest)[fix_i] if multi_invest else None
    kwargs = dict(spec="non_ext", mask=active)

    dispatch = lp.define_variables(-inf, inf, component, attr, axes=axes, **kwargs)
    dispatch = linear_expression((1, dispatch))
    lp.define_constraints(dispatch, ">=", lower, component, "mu_lower", **kwargs)
    lp.define_constraints(dispatch, "<=", upper, component, "mu_upper", **kwargs)


def define_dispatch_for_extendable_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        attr: str,
        multi_invest: bool,
):
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    network : pypsa.Network
    component : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    ext_i = get_extendable_i(network, component)
    if ext_i.empty:
        return
    min_pu, max_pu = get_bounds_pu(network, component, snapshots, ext_i, attr)
    operational_ext_v = lp.get_variable_references(component, attr)[ext_i]
    nominal_v = lp.get_variable_references(component, nominal_attrs[component])[ext_i]
    rhs = 0

    active = get_activity_mask(network, component, snapshots, multi_invest)[ext_i] if multi_invest else None
    kwargs = dict(spec=attr, mask=active)

    lhs, *axes = linear_expression((max_pu, nominal_v), (-1, operational_ext_v), return_axes=True)
    lp.define_constraints(lhs, ">=", rhs, component, "mu_upper", axes=axes, **kwargs)

    lhs, *axes = linear_expression((min_pu, nominal_v), (-1, operational_ext_v), return_axes=True)
    lp.define_constraints(lhs, "<=", rhs, component, "mu_lower", axes=axes, **kwargs)


def define_fixed_variable_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        attr: str,
        pnl: bool,
        multi_invest: bool,
):
    """
    Sets constraints for fixing variables of a given component and attribute to
    the corresponding values in n.df(c)[attr + '_set'] if pnl is True, or
    n.pnl(c)[attr + '_set']

    Parameters
    ----------
    network : pypsa.Network
    component : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    pnl : bool, default True
        Whether variable which should be fixed is time-dependent
    """

    if pnl:
        if attr + "_set" not in network.pnl(component):
            return

        fix = network.pnl(component)[attr + "_set"].loc[snapshots]
        if fix.empty:
            return

        if multi_invest:
            active = get_activity_mask(network, component, snapshots, multi_invest)
            fix = fix.where(active)

        fix = fix.stack()
        lhs = linear_expression((1, lp.get_variable_references(component, attr).stack()[fix.index]), as_pandas=False)
        constraints = lp.define_constraints(lhs, "=", fix).unstack().T
    else:
        if attr + "_set" not in network.df(component):
            return
        fix = network.df(component)[attr + "_set"].dropna()
        if fix.empty:
            return
        lhs = linear_expression((1, lp.get_variable_references(component, attr)[fix.index]), as_pandas=False)
        constraints = lp.define_constraints(lhs, "=", fix)
    lp.set_constraint_references(constraints, component, f"mu_{attr}_set")


def define_generator_status_variables(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool,
):
    component = "Generator"
    com_i = network.generators.query("committable").index
    ext_i = get_extendable_i(network, component)
    if not (ext_i.intersection(com_i)).empty:
        logger.warning(
            "The following generators have both investment optimisation"
            f" and unit commitment:\n\n\t{', '.join((ext_i.intersection(com_i)))}\n\nCurrently PyPSA cannot "
            "do both these functions, so PyPSA is choosing investment optimisation "
            "for these generators."
        )
        com_i = com_i.difference(ext_i)
    if com_i.empty:
        return
    active = get_activity_mask(network, component, snapshots, multi_invest)[com_i] if multi_invest else None
    lp.define_binaries((snapshots, com_i), "Generator", "status", mask=active)


def define_committable_generator_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool,
):
    component, attr = "Generator", "status"
    com_i = network.df(component).query("committable and not p_nom_extendable").index
    if com_i.empty:
        return
    nominal = network.df(component)[nominal_attrs[component]][com_i]
    min_pu, max_pu = get_bounds_pu(network, component, snapshots, com_i, "p")
    lower = min_pu.mul(nominal)
    upper = max_pu.mul(nominal)

    status = lp.get_variable_references(component, attr)
    p = lp.get_variable_references(component, "p")[com_i]

    lhs = linear_expression((lower, status), (-1, p))
    active = get_activity_mask(network, component, snapshots, multi_invest)[com_i] if multi_invest else None
    lp.define_constraints(lhs, "<=", 0, "Generators", "committable_lb", mask=active)

    lhs = linear_expression((upper, status), (-1, p))
    lp.define_constraints(lhs, ">=", 0, "Generators", "committable_ub", mask=active)


def define_ramp_limit_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        multi_invest: bool
):
    """
    Defines ramp limits for a given component with valid ramplimit.
    """
    rup_i = network.df(component).query("ramp_limit_up == ramp_limit_up").index
    rdown_i = network.df(component).query("ramp_limit_down == ramp_limit_down").index
    if rup_i.empty & rdown_i.empty:
        return
    fix_i = get_non_extendable_i(network, component)
    ext_i = get_extendable_i(network, component)
    if "committable" in network.df(component):
        com_i = network.df(component).query("committable").index.difference(ext_i)
    else:
        com_i = []

    # Check if ramping is not at start of n.snapshots
    start_i = network.snapshots.get_loc(snapshots[0]) - 1
    pnl = network.pnl(component)
    # get dispatch for either one or two ports
    attr = ({"p", "p0"} & set(pnl)).pop()
    p_prev_fix = pnl[attr].iloc[start_i]
    is_rolling_horizon = (snapshots[0] != network.snapshots[0]) and not p_prev_fix.empty

    if is_rolling_horizon:
        active = get_activity_mask(network, component, snapshots, multi_invest)
        p = lp.get_variable_references(component, "p")
        p_prev = lp.get_variable_references(component, "p").shift(1, fill_value=-1)
        rhs_prev = pd.DataFrame(0, *p.axes)
        rhs_prev.loc[snapshots[0]] = p_prev_fix
    else:
        active = get_activity_mask(network, component, snapshots[1:], multi_invest)
        p = lp.get_variable_references(component, "p").loc[snapshots[1:]]
        p_prev = lp.get_variable_references(component, "p").shift(1, fill_value=-1).loc[snapshots[1:]]
        rhs_prev = pd.DataFrame(0, *p.axes)

    # fix up
    gens_i = rup_i.intersection(fix_i)
    if not gens_i.empty:
        lhs = linear_expression((1, p[gens_i]), (-1, p_prev[gens_i]))
        rhs = rhs_prev[gens_i] + network.df(component).loc[gens_i].eval("ramp_limit_up * p_nom")
        kwargs = dict(spec="nonext.", mask=active[gens_i])
        lp.define_constraints(lhs, "<=", rhs, component, "mu_ramp_limit_up", **kwargs)

    # ext up
    gens_i = rup_i.intersection(ext_i)
    if not gens_i.empty:
        limit_pu = network.df(component)["ramp_limit_up"][gens_i]
        p_nom = lp.get_variable_references(component, "p_nom")[gens_i]
        lhs = linear_expression((1, p[gens_i]), (-1, p_prev[gens_i]), (-limit_pu, p_nom))
        rhs = rhs_prev[gens_i]
        kwargs = dict(spec="ext.", mask=active[gens_i])
        lp.define_constraints(lhs, "<=", rhs, component, "mu_ramp_limit_up", **kwargs)

    # com up
    gens_i = rup_i.intersection(com_i)
    if not gens_i.empty:
        limit_start = network.df(component).loc[gens_i].eval("ramp_limit_start_up * p_nom")
        limit_up = network.df(component).loc[gens_i].eval("ramp_limit_up * p_nom")
        status = lp.get_variable_references(component, "status").loc[p.index, gens_i]
        status_prev = (
            lp.get_variable_references(component, "status").shift(1, fill_value=-1).loc[p.index, gens_i]
        )
        lhs = linear_expression(
            (1, p[gens_i]),
            (-1, p_prev[gens_i]),
            (limit_start - limit_up, status_prev),
            (-limit_start, status),
        )
        rhs = rhs_prev[gens_i]
        if is_rolling_horizon:
            status_prev_fix = network.pnl(component)["status"][com_i].iloc[start_i]
            rhs.loc[snapshots[0]] += (limit_up - limit_start) * status_prev_fix
        kwargs = dict(spec="com.", mask=active[gens_i])
        lp.define_constraints(lhs, "<=", rhs, component, "mu_ramp_limit_up", **kwargs)

    # fix down
    gens_i = rdown_i.intersection(fix_i)
    if not gens_i.empty:
        lhs = linear_expression((1, p[gens_i]), (-1, p_prev[gens_i]))
        rhs = rhs_prev[gens_i] + network.df(component).loc[gens_i].eval(
            "-1 * ramp_limit_down * p_nom"
        )
        kwargs = dict(spec="nonext.", mask=active[gens_i])
        lp.define_constraints(lhs, ">=", rhs, component, "mu_ramp_limit_down", **kwargs)

    # ext down
    gens_i = rdown_i.intersection(ext_i)
    if not gens_i.empty:
        limit_pu = network.df(component)["ramp_limit_down"][gens_i]
        p_nom = lp.get_variable_references(component, "p_nom")[gens_i]
        lhs = linear_expression((1, p[gens_i]), (-1, p_prev[gens_i]), (limit_pu, p_nom))
        rhs = rhs_prev[gens_i]
        kwargs = dict(spec="ext.", mask=active[gens_i])
        lp.define_constraints(lhs, ">=", rhs, component, "mu_ramp_limit_down", **kwargs)

    # com down
    gens_i = rdown_i.intersection(com_i)
    if not gens_i.empty:
        limit_shut = network.df(component).loc[gens_i].eval("ramp_limit_shut_down * p_nom")
        limit_down = network.df(component).loc[gens_i].eval("ramp_limit_down * p_nom")
        status = lp.get_variable_references(component, "status").loc[p.index, gens_i]
        status_prev = (
            lp.get_variable_references(component, "status").shift(1, fill_value=-1).loc[p.index, gens_i]
        )
        lhs = linear_expression(
            (1, p[gens_i]),
            (-1, p_prev[gens_i]),
            (limit_down - limit_shut, status),
            (limit_shut, status_prev),
        )
        rhs = rhs_prev[gens_i]
        if is_rolling_horizon:
            status_prev_fix = network.pnl(component)["status"][com_i].iloc[start_i]
            rhs.loc[snapshots[0]] += -limit_shut * status_prev_fix
        kwargs = dict(spec="com.", mask=active[gens_i])
        lp.define_constraints(lhs, ">=", rhs, component, "mu_ramp_limit_down", **kwargs)


def define_nominal_constraints_per_bus_carrier(
        lp: LinearProgram,
        network: Network,
):
    for carrier in network.carriers.index:
        for bound, sense in [("max", "<="), ("min", ">=")]:

            col = f"nom_{bound}_{carrier}"
            if col not in network.buses.columns:
                continue
            rhs = network.buses[col].dropna()
            lhs = pd.Series("", rhs.index)

            for c, attr in nominal_attrs.items():
                if c not in network.one_port_components:
                    continue
                attr = nominal_attrs[c]
                # TODO change access!
                if (c, attr) not in lp.variables.index:
                    continue
                nominals = lp.get_variable_references(c, attr)[network.df(c).carrier == carrier]
                if nominals.empty:
                    continue
                per_bus = (
                    linear_expression((1, nominals)).groupby(network.df(c).bus).sum(**agg_group_kwargs)
                )
                lhs += per_bus.reindex(lhs.index, fill_value="")

            if bound == "max":
                lhs = lhs[lhs != ""]
                rhs = rhs.reindex(lhs.index)
            else:
                assert (lhs != "").all(), (
                    f"No extendable components of carrier {carrier} on bus "
                    f'{list(lhs[lhs == ""].index)}'
                )
            lp.define_constraints(lhs, sense, rhs, "Bus", "mu_" + col)


def define_nodal_balance_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots
):
    """
    Defines nodal balance constraint.
    """

    def bus_injection(c, attr, groupcol="bus", sign=1):
        # additional sign only necessary for branches in reverse direction
        if "sign" in network.df(c):
            sign = sign * network.df(c).sign
        expr = linear_expression((sign, lp.get_variable_references(c, attr))).rename(columns=network.df(c)[groupcol])
        # drop empty bus2, bus3 if multiline link
        if c == "Link":
            expr.drop(columns="", errors="ignore", inplace=True)
        return expr

    # one might reduce this a bit by using n.branches and lookup
    args = [
        ["Generator", "p"],
        ["Store", "p"],
        ["StorageUnit", "p_dispatch"],
        ["StorageUnit", "p_store", "bus", -1],
        ["Line", "s", "bus0", -1],
        ["Line", "s", "bus1", 1],
        ["Transformer", "s", "bus0", -1],
        ["Transformer", "s", "bus1", 1],
        ["Link", "p", "bus0", -1],
        ["Link", "p", "bus1", get_as_dense(network, "Link", "efficiency", snapshots)],
    ]
    args = [arg for arg in args if not network.df(arg[0]).empty]

    if not network.links.empty:
        for i in additional_linkports(network):
            eff = get_as_dense(network, "Link", f"efficiency{i}", snapshots)
            args.append(["Link", "p", f"bus{i}", eff])

    lhs = (
        pd.concat([bus_injection(*arg) for arg in args], axis=1)
            .groupby(axis=1, level=0)
            .sum(**agg_group_kwargs)
            .reindex(columns=network.buses.index, fill_value="")
    )

    if (lhs == "").any().any():
        raise ValueError("Empty LHS in nodal balance constraint.")

    sense = "="
    rhs = (
        (-get_as_dense(network, "Load", "p_set", snapshots) * network.loads.sign)
            .groupby(network.loads.bus, axis=1)
            .sum()
            .reindex(columns=network.buses.index, fill_value=0)
    )
    lp.define_constraints(lhs, sense, rhs, "Bus", "marginal_price")


def define_kirchhoff_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool,
):
    """
    Defines Kirchhoff voltage constraints.
    """
    comps = network.passive_branch_components & set(lp.variables.index.levels[0])
    if len(comps) == 0:
        return
    branch_vars = pd.concat({c: lp.get_variable_references(c, "s") for c in comps}, axis=1)

    def cycle_flow(ds, sns):
        if sns is None:
            sns = slice(None)
        ds = ds[lambda ds: ds != 0.0].dropna()
        vals = linear_expression((ds, branch_vars.loc[sns, ds.index]), as_pandas=False)
        return vals.sum(1)

    constraints = []
    periods = snapshots.unique("period") if multi_invest else [None]
    for period in periods:
        network.determine_network_topology(investment_period=period)
        subconstraints = []
        for sub in network.sub_networks.obj:
            branches = sub.branches()
            C = pd.DataFrame(sub.C.todense(), index=branches.index)
            if C.empty:
                continue
            carrier = network.sub_networks.carrier[sub.name]
            weightings = branches.x_pu_eff if carrier == "AC" else branches.r_pu_eff
            C_weighted = 1e5 * C.mul(weightings, axis=0)
            cycle_sum = C_weighted.apply(cycle_flow, sns=period)
            _snapshots = snapshots if period == None else snapshots[snapshots.get_loc(period)]
            cycle_sum.set_index(_snapshots, inplace=True)

            con = lp.define_constraints(cycle_sum, "=", 0)
            subconstraints.append(con)
        if len(subconstraints) == 0:
            continue
        constraints.append(pd.concat(subconstraints, axis=1, ignore_index=True))
    if constraints:
        constraints = pd.concat(constraints).rename_axis(columns="cycle")
        lp.set_constraint_references(constraints, "SubNetwork", "mu_kirchhoff_voltage_law")


def define_storage_unit_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool
):
    """
    Defines state of charge (soc) constraints for storage units. In principal
    the constraints states:

    previous_soc + p_store - p_dispatch + inflow - spill == soc
    """
    sus_i = network.storage_units.index
    if sus_i.empty:
        return
    c = "StorageUnit"
    # spillage
    has_periods = isinstance(snapshots, pd.MultiIndex)
    active = get_activity_mask(network, c, snapshots, multi_invest)

    upper = get_as_dense(network, c, "inflow", snapshots).loc[:, lambda df: df.max() > 0]
    spill = lp.define_variables(0, upper, "StorageUnit", "spill", mask=active[upper.columns])

    # elapsed hours
    eh = expand_series(network.snapshot_weightings.stores[snapshots], sus_i)
    # efficiencies
    eff_stand = expand_series(1 - network.df(c).standing_loss, snapshots).T.pow(eh)
    eff_dispatch = expand_series(network.df(c).efficiency_dispatch, snapshots).T
    eff_store = expand_series(network.df(c).efficiency_store, snapshots).T

    soc = lp.get_variable_references(c, "state_of_charge")

    if has_periods:
        cyclic_i = (
            network.df(c)
                .query("cyclic_state_of_charge & " "~cyclic_state_of_charge_per_period")
                .index
        )
        cyclic_pp_i = (
            network.df(c)
                .query("cyclic_state_of_charge & " "cyclic_state_of_charge_per_period")
                .index
        )
        noncyclic_i = (
            network.df(c)
                .query("~cyclic_state_of_charge & " "~state_of_charge_initial_per_period")
                .index
        )
        noncyclic_pp_i = (
            network.df(c)
                .query("~cyclic_state_of_charge & " "state_of_charge_initial_per_period")
                .index
        )
    else:
        cyclic_i = network.df(c).query("cyclic_state_of_charge").index
        cyclic_pp_i = None
        noncyclic_i = network.df(c).query("~cyclic_state_of_charge ").index
        noncyclic_pp_i = None

    # cyclic constraint for whole optimization horizon
    previous_soc_cyclic = (
        soc.where(active).ffill().apply(lambda ds: np.roll(ds, 1)).ffill()
    )

    # non cyclic constraint: determine the first active snapshot
    first_active_snapshot = active.cumsum()[noncyclic_i] == 1

    coeff_var = [
        (-1, soc),
        (-1 / eff_dispatch * eh, lp.get_variable_references(c, "p_dispatch")),
        (eff_store * eh, lp.get_variable_references(c, "p_store")),
    ]

    lhs, *axes = linear_expression(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return (
            linear_expression((coeff[cols], var[cols]))
                .reindex(index=axes[0], columns=axes[1], fill_value="")
                .values
        )

    if (c, "spill") in lp.variables.index:
        lhs += masked_term(-eh, lp.get_variable_references(c, "spill"), spill.columns)

    lhs += masked_term(eff_stand, previous_soc_cyclic, cyclic_i)
    lhs += masked_term(
        eff_stand[~first_active_snapshot],
        soc.shift()[~first_active_snapshot],
        noncyclic_i,
    )

    # rhs set e at beginning of optimization horizon for noncyclic
    rhs = -get_as_dense(network, c, "inflow", snapshots).mul(eh).astype(float)

    rhs[noncyclic_i] = rhs[noncyclic_i].where(
        ~first_active_snapshot, rhs - network.df(c).state_of_charge_initial, axis=1
    )

    if has_periods:
        # cyclic constraint for soc per period - cyclic soc within each period
        previous_soc_cyclic_pp = soc.groupby(level=0).transform(
            lambda ds: np.roll(ds, 1)
        )
        lhs += masked_term(eff_stand, previous_soc_cyclic_pp, cyclic_pp_i)

        # set the initial enery at the beginning of each period
        first_active_snapshot_pp = active[noncyclic_pp_i].groupby(level=0).cumsum() == 1

        lhs += masked_term(
            eff_stand[~first_active_snapshot_pp],
            soc.shift()[~first_active_snapshot_pp],
            noncyclic_pp_i,
        )

        rhs[noncyclic_pp_i] = rhs[noncyclic_pp_i].where(
            ~first_active_snapshot_pp, rhs - network.df(c).state_of_charge_initial, axis=1
        )

    lp.define_constraints(lhs, "==", rhs, c, "mu_state_of_charge", mask=active)


def define_store_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool
):
    """
    Defines energy balance constraints for stores. In principal this states:

    previous_e - p == e
    """
    stores_i = network.stores.index
    if stores_i.empty:
        return
    c = "Store"

    has_periods = isinstance(snapshots, pd.MultiIndex)
    active = get_activity_mask(network, c, snapshots, multi_invest)

    lp.define_variables(-inf, inf, name=c, attr="p", axes=[snapshots, stores_i], mask=active)

    # elapsed hours
    eh = expand_series(network.snapshot_weightings.stores[snapshots], stores_i)  # elapsed hours
    eff_stand = expand_series(1 - network.df(c).standing_loss, snapshots).T.pow(eh)

    e = lp.get_variable_references(c, "e")

    if has_periods:
        cyclic_i = network.df(c).query("e_cyclic & ~e_cyclic_per_period").index
        cyclic_pp_i = network.df(c).query("e_cyclic & e_cyclic_per_period").index
        noncyclic_i = network.df(c).query("~e_cyclic & ~e_initial_per_period").index
        noncyclic_pp_i = network.df(c).query("~e_cyclic & e_initial_per_period").index
    else:
        cyclic_i = network.df(c).query("e_cyclic").index
        cyclic_pp_i = None
        noncyclic_i = network.df(c).query("~e_cyclic").index
        noncyclic_pp_i = None

    # cyclic constraint for whole optimization horizon
    previous_e_cyclic = e.where(active).ffill().apply(lambda ds: np.roll(ds, 1)).ffill()

    # non cyclic constraint: determine the first active snapshot
    first_active_snapshot = active.cumsum()[noncyclic_i] == 1

    coeff_var = [(-eh, lp.get_variable_references(c, "p")), (-1, e)]

    lhs, *axes = linear_expression(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return (
            linear_expression((coeff[cols], var[cols]))
                .reindex(index=snapshots, columns=stores_i, fill_value="")
                .values
        )

    lhs += masked_term(eff_stand, previous_e_cyclic, cyclic_i)
    lhs += masked_term(
        eff_stand[~first_active_snapshot],
        e.shift()[~first_active_snapshot],
        noncyclic_i,
    )

    # rhs set e at beginning of optimization horizon for noncyclic
    rhs = pd.DataFrame(0.0, snapshots, stores_i)

    rhs[noncyclic_i] = rhs[noncyclic_i].where(
        ~first_active_snapshot, -network.df(c).e_initial, axis=1
    )

    if has_periods:
        # cyclic constraint for soc per period - cyclic soc within each period
        previous_e_cyclic_pp = e.groupby(level=0).transform(lambda ds: np.roll(ds, 1))
        lhs += masked_term(eff_stand, previous_e_cyclic_pp, cyclic_pp_i)

        # set the initial enery at the beginning of each period
        first_active_snapshot_pp = active[noncyclic_pp_i].groupby(level=0).cumsum() == 1

        lhs += masked_term(
            eff_stand[~first_active_snapshot_pp],
            e.shift()[~first_active_snapshot_pp],
            noncyclic_pp_i,
        )

        rhs[noncyclic_pp_i] = rhs[noncyclic_pp_i].where(
            ~first_active_snapshot_pp, -network.df(c).e_initial, axis=1
        )

    lp.define_constraints(lhs, "==", rhs, c, "mu_state_of_charge", mask=active)


def define_growth_limit(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        component: str,
        attr: str,
        multi_invest: bool,
):
    """
    Constraint new installed capacity per investment period.

    Parameters
    ----------
    network    : pypsa.Network
    component    : str
           network component of which the nominal capacity should be defined
    attr : str
           name of the variable, e.g. 'p_nom'
    """
    if not multi_invest:
        return

    ext_i = get_extendable_i(network, component)
    if "carrier" not in network.df(component) or network.df(component).empty:
        return
    with_limit = network.carriers.query("max_growth != inf").index
    limit_i = network.df(component).query("carrier in @with_limit").index.intersection(ext_i)
    if limit_i.empty:
        return

    periods = snapshots.unique("period")

    v = lp.get_variable_references(component, attr)
    carriers = network.df(component).loc[limit_i, "carrier"]
    caps = pd.concat(
        {
            period: linear_expression((1, v)).where(network.get_active_assets(component, period), "")
            for period in periods
        },
        axis=1,
    ).T[limit_i]
    lhs = caps.groupby(carriers, axis=1).sum(**agg_group_kwargs)
    rhs = network.carriers.max_growth[with_limit]

    lp.define_constraints(lhs, "<=", rhs, "Carrier", "growth_limit_{}".format(component))


def define_global_constraints(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool,
):
    """
    Defines global constraints for the optimization. Possible types are.

    1. primary_energy
        Use this to constraint the byproducts of primary energy sources as
        CO2
    2. transmission_volume_expansion_limit
        Use this to set a limit for line volume expansion. Possible carriers
        are 'AC' and 'DC'
    3. transmission_expansion_cost_limit
        Use this to set a limit for line expansion costs. Possible carriers
        are 'AC' and 'DC'
    4. tech_capacity_expansion_limit
        Use this to se a limit for the summed capacitiy of a carrier (e.g.
        'onwind') for each investment period at choosen nodes. This limit
        could e.g. represent land resource/ building restrictions for a
        technology in a certain region. Currently, only the
        capacities of extendable generators have to be below the set limit.
    """

    if multi_invest:
        period_weighting = network.investment_period_weightings["years"]
        weightings = network.snapshot_weightings.mul(period_weighting, level=0, axis=0).loc[
            snapshots
        ]
    else:
        weightings = network.snapshot_weightings.loc[snapshots]

    def get_period(glc, sns):
        period = slice(None)
        if multi_invest and not np.isnan(glc["investment_period"]):
            period = int(glc["investment_period"])
            if period not in sns.unique("period"):
                logger.warning(
                    "Optimized snapshots do not contain the investment "
                    f"period required for global constraint `{glc.name}`."
                )
        return period

    # (1) primary_energy
    glcs = network.global_constraints.query('type == "primary_energy"')
    for name, glc in glcs.iterrows():
        rhs = glc.constant
        lhs = ""
        carattr = glc.carrier_attribute
        emissions = network.carriers.query(f"{carattr} != 0")[carattr]
        period = get_period(glc, snapshots)

        if emissions.empty:
            continue

        # generators
        gens = network.generators.query("carrier in @emissions.index")
        if not gens.empty:
            em_pu = gens.carrier.map(emissions) / gens.efficiency
            em_pu = (
                    weightings["generators"].to_frame("weightings")
                    @ em_pu.to_frame("weightings").T
            ).loc[period]
            p = lp.get_variable_references("Generator", "p").loc[snapshots, gens.index].loc[period]

            vals = linear_expression((em_pu, p), as_pandas=False)
            lhs += join_expressions(vals)

        # storage units
        sus = network.storage_units.query(
            "carrier in @emissions.index and " "not cyclic_state_of_charge"
        )
        sus_i = sus.index
        if not sus.empty:
            em_pu = sus.carrier.map(emissions)
            soc = (
                lp.get_variable_references("StorageUnit", "state_of_charge").loc[snapshots, sus_i].loc[period]
            )
            soc = soc.where(soc != -1).ffill().iloc[-1]
            vals = linear_expression((-em_pu, soc), as_pandas=False)
            lhs = lhs + "\n" + join_expressions(vals)
            rhs -= em_pu @ sus.state_of_charge_initial

        # stores
        network.stores["carrier"] = network.stores.bus.map(network.buses.carrier)
        stores = network.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            em_pu = stores.carrier.map(emissions)
            e = lp.get_variable_references("Store", "e").loc[snapshots, stores.index].loc[period]
            e = e.where(e != -1).ffill().iloc[-1]
            vals = linear_expression((-em_pu, e), as_pandas=False)
            lhs = lhs + "\n" + join_expressions(vals)
            rhs -= stores.carrier.map(emissions) @ stores.e_initial

        lp.define_constraints(lhs, glc.sense, rhs, "GlobalConstraint", "mu", spec=name, axes=pd.Index([name]))

    # (2) transmission_volume_expansion_limit
    glcs = network.global_constraints.query(
        "type == " '"transmission_volume_expansion_limit"'
    )
    substr = lambda s: re.sub(r"[\[\]()]", "", s)
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        lhs = ""
        period = get_period(glc, snapshots)
        for c, attr in (("Line", "s_nom"), ("Link", "p_nom")):
            if network.df(c).empty:
                continue
            ext_i = network.df(c).query(f"carrier in @car and {attr}_extendable").index
            ext_i = ext_i[get_activity_mask(network, c, snapshots, multi_invest)[ext_i].loc[period].any()]

            if ext_i.empty:
                continue
            v = linear_expression(
                (network.df(c).length[ext_i], lp.get_variable_references(c, attr)[ext_i]), as_pandas=False
            )
            lhs += "\n" + join_expressions(v)
        if lhs == "":
            continue
        sense = glc.sense
        rhs = glc.constant
        lp.define_constraints(lhs, sense, rhs, "GlobalConstraint", "mu", spec=name, axes=pd.Index([name]))

    # (3) transmission_expansion_cost_limit
    glcs = network.global_constraints.query("type == " '"transmission_expansion_cost_limit"')
    for name, glc in glcs.iterrows():
        car = [substr(c.strip()) for c in glc.carrier_attribute.split(",")]
        lhs = ""
        period = get_period(glc, snapshots)
        for c, attr in (("Line", "s_nom"), ("Link", "p_nom")):
            ext_i = network.df(c).query(f"carrier in @car and {attr}_extendable").index
            ext_i = ext_i[get_activity_mask(network, c, snapshots, multi_invest)[ext_i].loc[period].any()]

            if ext_i.empty:
                continue

            v = linear_expression(
                (network.df(c).capital_cost[ext_i], lp.get_variable_references(c, attr)[ext_i]),
                as_pandas=False,
            )
            lhs += "\n" + join_expressions(v)
        if lhs == "":
            continue
        sense = glc.sense
        rhs = glc.constant
        lp.define_constraints(lhs, sense, rhs, "GlobalConstraint", "mu", spec=name, axes=pd.Index([name]))

    # (4) tech_capacity_expansion_limit
    # TODO: Generalize to carrier capacity expansion limit (i.e. also for stores etc.)
    glcs = network.global_constraints.query("type == " '"tech_capacity_expansion_limit"')
    c, attr = "Generator", "p_nom"

    for name, glc in glcs.iterrows():
        period = get_period(glc, snapshots)
        car = glc["carrier_attribute"]
        bus = str(glc.get("bus", ""))  # in pypsa buses are always strings
        ext_i = network.df(c).query("carrier == @car and p_nom_extendable").index
        if bus:
            ext_i = network.df(c).loc[ext_i].query("bus == @bus").index
        ext_i = ext_i[get_activity_mask(network, c, snapshots, multi_invest)[ext_i].loc[period].any()]

        if ext_i.empty:
            continue

        cap_vars = lp.get_variable_references(c, attr)[ext_i]

        lhs = join_expressions(linear_expression((1, cap_vars)))
        rhs = glc.constant
        sense = glc.sense

        lp.define_constraints(lhs, sense, rhs, "GlobalConstraint", "mu", spec=name, axes=pd.Index([name]))


def define_objective(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        multi_invest: bool,
):
    """
    Defines and writes out the objective function.
    """
    if multi_invest:
        period_weighting = network.investment_period_weightings.objective[snapshots.unique("period")]
    else:
        period_weighting = None

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(network, c)
        cost = network.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(network, c, period)[ext_i]
                    for period in snapshots.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += cost @ network.df(c)[attr][ext_i]

    # TODO use generic methods
    object_const = lp.define_variables(constant, constant)
    lp.define_objective(linear_expression((-1, object_const), as_pandas=False)[0])
    network.objective_constant = constant

    # marginal cost
    if multi_invest:
        weighting = network.snapshot_weightings.objective.mul(period_weighting, level=0).loc[
            snapshots
        ]
    else:
        weighting = network.snapshot_weightings.objective.loc[snapshots]

    for c, attr in lookup.query("marginal_cost").index:
        cost = (
            get_as_dense(network, c, "marginal_cost", snapshots)
                .loc[:, lambda ds: (ds != 0).all()]
                .mul(weighting, axis=0)
        )
        if cost.empty:
            continue
        terms = linear_expression((cost, lp.get_variable_references(c, attr).loc[snapshots, cost.columns]))
        lp.define_objective(terms)

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = get_extendable_i(network, c)
        cost = network.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if multi_invest:
            active = pd.concat(
                {
                    period: get_active_assets(network, c, period)[ext_i]
                    for period in snapshots.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = lp.get_variable_references(c, attr).loc[ext_i]
        terms = linear_expression((cost, caps))
        lp.define_objective(terms)


def build_lopf_problem(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        skip_objective: bool,
        multi_invest: bool,
):
    """
    Sets up the linear problem and writes it out to a lp file.

    Returns
    -------
    Tuple (fdp, problem_fn) indicating the file descriptor and the file name of
    the lp file
    """
    for c, attr in lookup.query("nominal and not handle_separately").index:
        define_nominal_for_extendable_variables(lp, network, c, attr)
        # define constraint for newly installed capacity per investment period
        define_growth_limit(lp, network, snapshots, c, attr, multi_invest)
        # define_fixed_variable_constraints(lp, network, snapshots, attr, pnl=False, multi_invest)
    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_dispatch_for_non_extendable_variables(lp, network, snapshots, c, attr, multi_invest)
        define_dispatch_for_extendable_and_committable_variables(lp, network, snapshots, c, attr, multi_invest)
        align_with_static_component(lp, network, c, attr)
        define_dispatch_for_extendable_constraints(lp, network, snapshots, c, attr, multi_invest)
        # define_fixed_variable_constraints(lp, network, snapshots, c, attr, multi_invest)
    define_generator_status_variables(lp, network, snapshots, multi_invest)
    define_nominal_constraints_per_bus_carrier(lp, network)

    # consider only state_of_charge_set for the moment
    define_fixed_variable_constraints(lp, network, snapshots, "StorageUnit", "state_of_charge", True, multi_invest)
    define_fixed_variable_constraints(lp, network, snapshots, "Store", "e", True, multi_invest)

    define_committable_generator_constraints(lp, network, snapshots, multi_invest)
    define_ramp_limit_constraints(lp, network, snapshots, component="Generator", multi_invest=multi_invest)
    define_ramp_limit_constraints(lp, network, snapshots, component="Link", multi_invest=multi_invest)
    define_storage_unit_constraints(lp, network, snapshots, multi_invest)
    define_store_constraints(lp, network, snapshots, multi_invest)
    define_kirchhoff_constraints(lp, network, snapshots, multi_invest)
    define_nodal_balance_constraints(lp, network, snapshots)
    define_global_constraints(lp, network, snapshots, multi_invest)
    if skip_objective:
        logger.info(
            "The argument `skip_objective` is set to True. Expecting a "
            "custom objective to be build via `extra_functionality`."
        )
    else:
        define_objective(lp, network, snapshots, multi_invest)


def assign_solution(
        lp: LinearProgram,
        network: Network,
        snapshots: Snapshots,
        solution: Solution,
        keep_references: bool,
        keep_shadowprices: bool,
        multi_invest: bool
):
    """
    Map solution to network components.

    Helper function. Assigns the solution of a succesful optimization to
    the network.
    """

    def set_from_frame(pnl, attr, df):
        if attr not in pnl:  # use this for subnetworks_t
            pnl[attr] = df.reindex(network.snapshots, fill_value=0)
        elif pnl[attr].empty:
            pnl[attr] = df.reindex(network.snapshots, fill_value=0)
        else:
            pnl[attr].loc[snapshots, df.columns] = df

    pop = not keep_references

    lp_sols = Dict()
    lp_solutions = pd.DataFrame(index=lp.variables.index, columns=["in_comp", "pnl"])

    def map_solution(c, attr):
        variables = lp.get_variable_references(c, attr, pop=pop)
        predefined = True
        if (c, attr) not in lookup.index:
            predefined = False
            lp_sols[c] = lp_sols[c] if c in lp_sols else Dict(df=pd.DataFrame(), pnl={})
        lp_solutions.at[(c, attr), "in_comp"] = predefined
        if isinstance(variables, pd.DataFrame):
            # case that variables are timedependent
            lp_solutions.at[(c, attr), "pnl"] = True
            pnl = network.pnl(c) if predefined else lp_sols[c].pnl
            solution.variables_sol.loc[-1] = 0
            values = variables.applymap(lambda x: solution.variables_sol.loc[x])
            if c in network.passive_branch_components and attr == "s":
                set_from_frame(pnl, "p0", values)
                set_from_frame(pnl, "p1", -values)
            elif c == "Link" and attr == "p":
                set_from_frame(pnl, "p0", values)
                for i in ["1"] + additional_linkports(network):
                    i_eff = "" if i == "1" else i
                    eff = get_as_dense(network, "Link", f"efficiency{i_eff}", snapshots)
                    set_from_frame(pnl, f"p{i}", -values * eff)
                    pnl[f"p{i}"].loc[
                        snapshots, network.links.index[network.links[f"bus{i}"] == ""]
                    ] = network.component_attrs["Link"].loc[f"p{i}", "default"]
            else:
                set_from_frame(pnl, attr, values)
        else:
            # case that variables are static
            lp_solutions.at[(c, attr), "pnl"] = False
            sol = variables.map(solution.variables_sol)
            if predefined:
                non_ext = network.df(c)[attr]
                network.df(c)[attr + "_opt"] = sol.reindex(non_ext.index).fillna(non_ext)
            else:
                lp_sols[c].df[attr] = sol

    for c, attr in lp.variables.index:
        map_solution(c, attr)

    # if nominal capacity was no variable set optimal value to nominal
    for c, attr in lookup.query("nominal").index.difference(lp.variables.index):
        network.df(c)[attr + "_opt"] = network.df(c)[attr]

    # recalculate storageunit net dispatch
    if not network.df("StorageUnit").empty:
        c = "StorageUnit"
        network.pnl(c)["p"] = network.pnl(c)["p_dispatch"] - network.pnl(c)["p_store"]

    # duals
    if keep_shadowprices == False:
        keep_shadowprices = []

    sp = lp.constraints.index
    if isinstance(keep_shadowprices, list):
        sp = sp[sp.isin(keep_shadowprices, level=0)]

    lp_duals = Dict()
    lp_dualvalues = pd.DataFrame(index=sp, columns=["in_comp", "pnl"])

    def map_dual(c, attr):
        # If c is a pypsa component name the dual is stored at n.pnl(c)
        # or n.df(c). For the second case the index of the constraints have to
        # be a subset of n.df(c).index otherwise the dual is stored at
        # n.duals[c].df
        constraints = lp.get_constraint_references(c, attr, pop=pop)
        is_pnl = isinstance(constraints, pd.DataFrame)
        # TODO: setting the sign is not very clear
        sign = 1 if "upper" in attr or attr == "marginal_price" else -1
        lp_dualvalues.at[(c, attr), "pnl"] = is_pnl
        to_component = c in network.all_components
        if is_pnl:
            lp_dualvalues.at[(c, attr), "in_comp"] = to_component
            duals = constraints.applymap(
                lambda x: sign * solution.constraints_dual.loc[x]
                if x in solution.constraints_dual.index
                else np.nan
            )
            if c not in lp_duals and not to_component:
                lp_duals[c] = Dict(df=pd.DataFrame(), pnl={})
            pnl = network.pnl(c) if to_component else lp_duals[c].pnl
            set_from_frame(pnl, attr, duals)
        else:
            # here to_component can change
            duals = constraints.map(sign * solution.constraints_dual)
            if to_component:
                to_component = duals.index.isin(network.df(c).index).all()
            lp_dualvalues.at[(c, attr), "in_comp"] = to_component
            if c not in lp_duals and not to_component:
                lp_duals[c] = Dict(df=pd.DataFrame(), pnl={})
            df = network.df(c) if to_component else lp_duals[c].df
            df[attr] = duals

    # extract shadow prices attached to components
    for c, attr in sp:
        map_dual(c, attr)

    # correct prices with objective weightings
    if multi_invest:
        period_weighting = network.investment_period_weightings.objective
        weightings = network.snapshot_weightings.objective.mul(
            period_weighting, level=0, axis=0
        ).loc[snapshots]
    else:
        weightings = network.snapshot_weightings.objective.loc[snapshots]

    network.buses_t.marginal_price.loc[snapshots] = network.buses_t.marginal_price.loc[snapshots].divide(
        weightings, axis=0
    )

    # discard remaining if wanted
    if not keep_references:
        for c, attr in lp.constraints.index.difference(sp):
            lp.get_constraint_references(c, attr, pop)  # TODO wtf?

    # load
    if len(network.loads):
        set_from_frame(network.pnl("Load"), "p", get_as_dense(network, "Load", "p_set", snapshots))

    # clean up vars and cons
    for c in list(lp.vars):
        if lp.vars[c].df.empty and lp.vars[c].pnl == {}:
            lp.vars.pop(c)
    for c in list(lp.cons):
        if lp.cons[c].df.empty and lp.cons[c].pnl == {}:
            lp.cons.pop(c)

    # recalculate injection
    ca = [
        ("Generator", "p", "bus"),
        ("Store", "p", "bus"),
        ("Load", "p", "bus"),
        ("StorageUnit", "p", "bus"),
        ("Link", "p0", "bus0"),
        ("Link", "p1", "bus1"),
    ]
    for i in additional_linkports(network):
        ca.append(("Link", f"p{i}", f"bus{i}"))

    def sign(c: str) -> int:
        # sign for 'Link'
        return network.df(c).sign if "sign" in network.df(c) else -1

    network.buses_t.p = (
        pd.concat(
            [
                network.pnl(c)[attr].mul(sign(c)).rename(columns=network.df(c)[group])
                for c, attr, group in ca
            ],
            axis=1,
        )
            .groupby(level=0, axis=1)
            .sum()
            .reindex(columns=network.buses.index, fill_value=0)
    )

    def v_ang_for_(sub):
        buses_i = sub.buses_o
        if len(buses_i) == 1:
            return pd.DataFrame(0, index=snapshots, columns=buses_i)
        sub.calculate_B_H(skip_pre=True)
        Z = pd.DataFrame(np.linalg.pinv((sub.B).todense()), buses_i, buses_i)
        Z -= Z[sub.slack_bus]
        return network.buses_t.p.reindex(columns=buses_i) @ Z

    network.buses_t.v_ang = pd.concat(
        [v_ang_for_(sub) for sub in network.sub_networks.obj], axis=1
    ).reindex(columns=network.buses.index, fill_value=0)

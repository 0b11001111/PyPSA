import io
import logging
import os
import re
import subprocess
from distutils.version import LooseVersion
from importlib.util import find_spec
from typing import Any, Dict, Optional

import pandas as pd

from pypsa.linear_program.solver import SolverBackend

logger = logging.getLogger(__name__)


def set_int_index(ser):
    ser.index = ser.index.str[1:].astype(int)
    return ser


def run_and_read_highs(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Highs solver function. Reads a linear problem file and passes it to the highs
    solver. If the solution is feasible the function returns the objective,
    solution and dual constraint variables. Highs must be installed for usage.
    Documentation: https://www.maths.ed.ac.uk/hall/HiGHS/

    Notes
    -----

    The script might only work for version HiGHS 1.1.1. Installation steps::
        sudo apt-get install cmake  # if not installed
        git clone git@github.com:ERGO-Code/HiGHS.git
        cd HiGHS
        git checkout 95342daa73543cc21e5b27db3e0fbf7330007541 # moves to HiGHS 1.1.1
        mkdir build
        cd build
        cmake ..
        make
        ctest

    Then in .bashrc add paths of executables and library ::
        export PATH="${PATH}:/foo/HiGHS/build/bin"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/foo/HiGHS/build/lib"
        source .bashrc

    Now when typing ``highs`` in the terminal you should see something like ::
        Running HiGHS 1.1.1 [date: 2021-11-14, git hash: 95342daa]

    The function reads and execute (i.e. subprocess.Popen,...) terminal
    commands of the solver. Meaning the command can be also executed at your
    command window/terminal if HiGHs is installed. Executing the commands on
    your local terminal helps to identify the raw outputs that are useful for
    developing the interface further.

    All functions below the "process = ..." do only read and save the outputs
    generated from the HiGHS solver. These parts are solver specific and
    depends on the solver output.

    Solver options are read by the 1) command window and the 2) option_file.txt

    1) An example list of solver options executable by the command window is given here:
    Examples:
    --model_file arg 	File of model to solve.
    --presolve arg 	    Presolve: "choose" by default - "on"/"off" are alternatives.
    --solver arg 	    Solver: "choose" by default - "simplex"/"ipm" are alternatives.
    --parallel arg 	    Parallel solve: "choose" by default - "on"/"off" are alternatives.
    --time_limit arg 	Run time limit (double).
    --options_file arg 	File containing HiGHS options.
    -h, --help 	        Print help.

    2) The options_file.txt gives some more options, see a full list here:
    https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set
    By default, we insert a couple of options for the ipm solver. The dictionary
    can be overwritten by simply giving the new values. For instance, you could
    write a dictionary replacing some of the default values or adding new options:

    ```
    solver_options = {
        name: highs,
        method: ipm,
        parallel: "on",
        <option_name>: <value>,
    }
    ```

    Note, the <option_name> and <value> must be equivalent to the name convention
    of HiGHS. Some function exist that are not documented, check their GitHub file:
    https://github.com/ERGO-Code/HiGHS/blob/master/src/lp_data/HighsOptions.h

    Returns
    -------

    status : string,
        "ok" or "warning"
    termination_condition : string,
        Contains "optimal", "infeasible",
    variables_sol : series
    constraints_dual : series
    objective : float

    """
    logger.warning(
        "The HiGHS solver can potentially solve towards variables that slightly deviate from Gurobi,cbc,glpk"
    )
    options_fn = "highs_options.txt"
    default_dict = {
        "method": "ipm",
        "primal_feasibility_tolerance": 1e-04,
        "dual_feasibility_tolerance": 1e-05,
        "ipm_optimality_tolerance": 1e-6,
        "presolve": "on",
        "run_crossover": True,
        "parallel": "off",
        "threads": 4,
        "solution_file": solution_fn,
        "write_solution_to_file": True,
        "write_solution_style": 1,
        "log_to_console": True,
    }
    # update default_dict through solver_options and write to file
    default_dict.update(solver_options)
    method = default_dict.pop("method", "ipm")
    logger.info(
        f'Options: "{default_dict}". List of options: https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set'
    )
    f1 = open(options_fn, "w")
    f1.write("\n".join([f"{k} = {v}" for k, v in default_dict.items()]))
    f1.close()

    # write (terminal) commands
    command = f"highs --model_file {problem_fn} "
    if basis_fn:
        logger.warning("basis_fn, not available in HiGHS. Will be ignored.")
    command += f"--solver {method} --options_file {options_fn}"
    logger.info(f'Solver command: "{command}"')
    # execute command and store command window output
    process = subprocess.Popen(
        command.split(" "), stdout=subprocess.PIPE, universal_newlines=True
    )

    def read_until_break():
        # Function that reads line by line the command window
        while True:
            out = process.stdout.readline(1)
            if out == "" and process.poll() != None:
                break
            if out != "":
                yield out

    # converts stdout (standard terminal output) to pandas dataframe
    log = io.StringIO("".join(read_until_break())[:])
    log = pd.read_csv(log, sep=":", index_col=0, header=None)[1].squeeze()
    if solver_logfile is not None:
        log.to_csv(solver_logfile, sep="\t")
    log.index = log.index.str.strip()
    os.remove(options_fn)

    # read out termination_condition from `info`
    model_status = log["Model   status"].strip().lower()
    if "optimal" in model_status:
        status = "ok"
        termination_condition = model_status
    elif "infeasible" in model_status:
        status = "warning"
        termination_condition = model_status
    else:
        status = "warning"
        termination_condition = model_status
    objective = float(log["Objective value"])

    # read out solution file (.sol)
    f = open(solution_fn, "rb")
    trimed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())
    f.close()

    sol = pd.read_csv(io.BytesIO(trimed_sol_fn), header=[1], sep=r"\s+")
    row_no = sol[sol["Index"] == "Rows"].index[0]
    sol = sol.drop(row_no + 1)  # Removes header line after "Rows"
    sol_rows = sol[(sol.index > row_no)]
    sol_cols = sol[(sol.index < row_no)].set_index("Name").pipe(set_int_index)
    variables_sol = pd.to_numeric(sol_cols["Primal"], errors="raise")
    constraints_dual = pd.to_numeric(sol_rows["Dual"], errors="raise").reset_index(
        drop=True
    )
    constraints_dual.index += 1

    return (status, termination_condition, variables_sol, constraints_dual, objective)


def run_and_read_cbc(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Solving function. Reads the linear problem file and passes it to the cbc
    solver. If the solution is successful it returns variable solutions and
    constraint dual values.

    For more information on the solver options, run 'cbc' in your shell
    """
    with open(problem_fn, "rb") as f:
        for str in f.readlines():
            assert ("> " in str.decode("utf-8")) == False, ">, must be" "changed to >="
            assert ("< " in str.decode("utf-8")) == False, "<, must be" "changed to <="

    # printingOptions is about what goes in solution file
    command = f"cbc -printingOptions all -import {problem_fn} "
    if basis_fn:
        command += f"-basisI {basis_fn} "
    if (solver_options is not None) and (solver_options != {}):
        command += solver_options
    command += f"-solve -solu {solution_fn} "
    if store_basis:
        basis_fn = solution_fn.replace(".sol", ".bas")
        command += f"-basisO {basis_fn} "

    if not os.path.exists(solution_fn):
        os.mknod(solution_fn)

    log = open(solver_logfile, "w") if solver_logfile is not None else subprocess.PIPE
    result = subprocess.Popen(command.split(" "), stdout=log)
    result.wait()

    with open(solution_fn, "r") as f:
        data = f.readline()

    if data.startswith("Optimal - objective value"):
        status = "ok"
        termination_condition = "optimal"
        objective = float(data[len("Optimal - objective value "):])
    elif "Infeasible" in data:
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = "warning"
        termination_condition = "other"

    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    f = open(solution_fn, "rb")
    trimed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())
    f.close()

    sol = pd.read_csv(
        io.BytesIO(trimed_sol_fn),
        header=None,
        skiprows=[0],
        sep=r"\s+",
        usecols=[1, 2, 3],
        index_col=0,
    )
    variables_b = sol.index.str[0] == "x"
    variables_sol = sol[variables_b][2].pipe(set_int_index)
    constraints_dual = sol[~variables_b][3].pipe(set_int_index)
    return (status, termination_condition, variables_sol, constraints_dual, objective)


def run_and_read_glpk(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Solving function. Reads the linear problem file and passes it to the glpk
    solver. If the solution is successful it returns variable solutions and
    constraint dual values.

    For more information on the glpk solver options:
    https://kam.mff.cuni.cz/~elias/glpk.pdf
    """
    # TODO use --nopresol argument for non-optimal solution output
    command = f"glpsol --lp {problem_fn} --output {solution_fn}"
    if solver_logfile is not None:
        command += f" --log {solver_logfile}"
    if basis_fn:
        command += f" --ini {basis_fn}"
    if store_basis:
        basis_fn = solution_fn.replace(".sol", ".bas")
        command += f" -w {basis_fn}"
    if (solver_options is not None) and (solver_options != {}):
        command += solver_options

    result = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE)
    result.wait()

    f = open(solution_fn)

    def read_until_break(f):
        linebreak = False
        while not linebreak:
            line = f.readline()
            linebreak = line == "\n"
            yield line

    info = io.StringIO("".join(read_until_break(f))[:-2])
    info = pd.read_csv(info, sep=":", index_col=0, header=None)[1]
    termination_condition = info.Status.lower().strip()
    objective = float(re.sub(r"[^0-9\.\+\-e]+", "", info.Objective))

    if termination_condition in ["optimal", "integer optimal"]:
        status = "ok"
        termination_condition = "optimal"
    elif termination_condition == "undefined":
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = "warning"

    if termination_condition != "optimal":
        return status, termination_condition, None, None, None

    duals = io.StringIO("".join(read_until_break(f))[:-2])
    duals = pd.read_fwf(duals)[1:].set_index("Row name")
    if "Marginal" in duals:
        constraints_dual = (
            pd.to_numeric(duals["Marginal"], "coerce").fillna(0).pipe(set_int_index)
        )
    else:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=duals.index, dtype=float)

    sol = io.StringIO("".join(read_until_break(f))[:-2])
    variables_sol = (
        pd.read_fwf(sol)[1:]
            .set_index("Column name")["Activity"]
            .astype(float)
            .pipe(set_int_index)
    )
    f.close()

    return (status, termination_condition, variables_sol, constraints_dual, objective)


def run_and_read_cplex(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Solving function.

    Reads the linear problem file and passes it to the cplex solver. If
    the solution is successful it returns variable solutions and
    constraint dual values. Cplex must be installed for using this
    function
    """
    if find_spec("cplex") is None:
        raise ModuleNotFoundError(
            "Optional dependency 'cplex' not found."
            "Install via 'conda install -c ibmdecisionoptimization cplex' "
            "or 'pip install cplex'"
        )
    import cplex

    _version = LooseVersion(cplex.__version__)
    m = cplex.Cplex()
    if solver_logfile is not None:
        if _version >= "12.10":
            log_file_or_path = open(solver_logfile, "w")
        else:
            log_file_or_path = solver_logfile
        m.set_log_stream(log_file_or_path)
    if solver_options is not None:
        for key, value in solver_options.items():
            param = m.parameters
            for key_layer in key.split("."):
                param = getattr(param, key_layer)
            param.set(value)
    m.read(problem_fn)
    if basis_fn:
        m.start.read_basis(basis_fn)
    m.solve()
    is_lp = m.problem_type[m.get_problem_type()] == "LP"
    if solver_logfile is not None:
        if isinstance(log_file_or_path, io.IOBase):
            log_file_or_path.close()

    termination_condition = m.solution.get_status_string()
    if "optimal" in termination_condition:
        status = "ok"
        termination_condition = "optimal"
    else:
        status = "warning"

    if (status == "ok") and store_basis and is_lp:
        basis_fn = solution_fn.replace(".sol", ".bas")
        try:
            m.solution.basis.write(basis_fn)
        except cplex.exceptions.errors.CplexSolverError:
            logger.info("No model basis stored")
            del basis_fn

    objective = m.solution.get_objective_value()
    variables_sol = pd.Series(m.solution.get_values(), m.variables.get_names()).pipe(
        set_int_index
    )
    if is_lp:
        constraints_dual = pd.Series(
            m.solution.get_dual_values(), m.linear_constraints.get_names()
        ).pipe(set_int_index)
    else:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=m.linear_constraints.get_names()).pipe(
            set_int_index
        )
    del m
    return (status, termination_condition, variables_sol, constraints_dual, objective)


def run_and_read_gurobi(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Solving function. Reads the linear problem file and passes it to the gurobi
    solver. If the solution is successful it returns variable solutions and
    constraint dual values. Gurobipy must be installed for using this function.

    For more information on solver options:
    https://www.gurobi.com/documentation/{gurobi_verion}/refman/parameter_descriptions.html
    """
    if find_spec("gurobipy") is None:
        raise ModuleNotFoundError(
            "Optional dependency 'gurobipy' not found. "
            "Install via 'conda install -c gurobi gurobi'  or follow the "
            "instructions on the documentation page "
            "https://www.gurobi.com/documentation/"
        )
    import gurobipy

    # disable logging for this part, as gurobi output is doubled otherwise
    logging.disable(50)

    m = gurobipy.read(problem_fn)
    if solver_options is not None:
        for key, value in solver_options.items():
            m.setParam(key, value)
    if solver_logfile is not None:
        m.setParam("logfile", solver_logfile)

    if basis_fn:
        m.read(basis_fn)
    m.optimize()
    logging.disable(1)

    if store_basis:
        basis_fn = solution_fn.replace(".sol", ".bas")
        try:
            m.write(basis_fn)
        except gurobipy.GurobiError:
            logger.info("No model basis stored")
            del basis_fn

    Status = gurobipy.GRB.Status
    statusmap = {
        getattr(Status, s): s.lower() for s in Status.__dir__() if not s.startswith("_")
    }
    termination_condition = statusmap[m.status]

    if termination_condition == "optimal":
        status = "ok"
    elif termination_condition == "suboptimal":
        status = "warning"
    elif termination_condition == "infeasible":
        status = "warning"
    elif termination_condition == "inf_or_unbd":
        status = "warning"
        termination_condition = "infeasibleOrUnbounded"
    else:
        status = "warning"

    if termination_condition not in ["optimal", "suboptimal"]:
        return status, termination_condition, None, None, None

    variables_sol = pd.Series({v.VarName: v.x for v in m.getVars()}).pipe(set_int_index)
    try:
        constraints_dual = pd.Series({c.ConstrName: c.Pi for c in m.getConstrs()}).pipe(
            set_int_index
        )
    except AttributeError:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=[c.ConstrName for c in m.getConstrs()])
    objective = m.ObjVal
    del m
    return (status, termination_condition, variables_sol, constraints_dual, objective)


def run_and_read_xpress(
        basis_fn,
        problem_fn,
        solution_fn,
        solver_logfile,
        solver_options,
        store_basis,
):
    """
    Solving function. Reads the linear problem file and passes it to the Xpress
    solver. If the solution is successful it returns variable solutions and
    constraint dual values. The xpress module must be installed for using this
    function.

    For more information on solver options:
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html
    """

    import xpress

    m = xpress.problem()

    m.read(problem_fn)
    m.setControl(solver_options)

    if solver_logfile is not None:
        m.setlogfile(solver_logfile)

    if basis_fn:
        m.readbasis(basis_fn)

    m.solve()

    if store_basis:
        basis_fn = solution_fn.replace(".sol", ".bas")
        try:
            m.writebasis(basis_fn)
        except:
            logger.info("No model basis stored")
            del basis_fn

    termination_condition = m.getProbStatusString()

    if termination_condition == "mip_optimal" or termination_condition == "lp_optimal":
        status = "ok"
        termination_condition = "optimal"
    elif (
            termination_condition == "mip_unbounded"
            or termination_condition == "mip_infeasible"
            or termination_condition == "lp_unbounded"
            or termination_condition == "lp_infeasible"
    ):
        status = "infeasible or unbounded"
    else:
        status = "warning"

    if termination_condition not in ["optimal"]:
        return status, termination_condition, None, None, None

    var = [str(v) for v in m.getVariable()]
    variables_sol = pd.Series(m.getSolution(var), index=var).pipe(set_int_index)

    try:
        dual = [str(d) for d in m.getConstraint()]
        constraints_dual = pd.Series(m.getDual(dual), index=dual).pipe(set_int_index)
    except xpress.SolverError:
        logger.warning("Shadow prices of MILP couldn't be parsed")
        constraints_dual = pd.Series(index=dual).pipe(set_int_index)

    objective = m.getObjVal()

    del m

    return (status, termination_condition, variables_sol, constraints_dual, objective)


SOLVERS = {
    SolverBackend.cbc: run_and_read_cbc,
    SolverBackend.cplex: run_and_read_cplex,
    SolverBackend.glpk: run_and_read_glpk,
    SolverBackend.gurobi: run_and_read_gurobi,
    SolverBackend.highs: run_and_read_highs,
    SolverBackend.xpress: run_and_read_xpress,
}


def run_and_read(
        solver: SolverBackend,
        basis_fn: Optional[str],
        problem_fn: str,
        solution_fn: str,
        solver_logfile: str,
        solver_options: Dict[str, Any],
        store_basis: bool,
) -> (str, str, Optional[pd.Series], Optional[pd.Series], Optional[float]):
    solve = SOLVERS[solver]
    return solve(basis_fn, problem_fn, solution_fn, solver_logfile, solver_options, store_basis)

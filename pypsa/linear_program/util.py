from typing import Union, Tuple

import numpy as np
import pandas as pd

Reference = Union[pd.Series, pd.DataFrame]


def _to_float_str(value: float) -> str:
    return f"{value:+f}"


_v_to_float_str = np.vectorize(_to_float_str, otypes=[object])


def _to_int_str(value) -> str:
    return f"{int(value):d}"


_v_to_int_str = np.vectorize(_to_int_str, otypes=[object])


def _broadcasted_axes(*dfs):
    """
    Helper function which, from a collection of arrays, series, frames or other
    values, retrieves the axes of series and frames which result from
    broadcasting operations.

    It checks whether index and columns of given series and frames,
    respectively, are aligned. Using this function allows to subsequently use
    pure numpy operations and keep the axes in the background.
    """
    axes = []
    shape = (1,)

    if set(map(type, dfs)) == {tuple}:
        dfs = sum(dfs, ())

    for df in dfs:
        shape = np.broadcast_shapes(shape, np.asarray(df).shape)
        if isinstance(df, (pd.Series, pd.DataFrame)):
            if len(axes):
                assert (axes[-1] == df.axes[-1]).all(), (
                    "Series or DataFrames "
                    "are not aligned. Please make sure that all indexes and "
                    "columns of Series and DataFrames going into the linear "
                    "expression are equally sorted."
                )
            axes = df.axes if len(df.axes) > len(axes) else axes
    return axes, shape


def str_array(array, integer_string=False):
    if isinstance(array, (float, int)):
        if integer_string:
            return _to_int_str(array)
        return _to_float_str(array)
    array = np.asarray(array)
    if array.dtype.type == np.str_:
        array = np.asarray(array, dtype=object)
    if array.dtype < str and array.size:
        if integer_string:
            array = np.nan_to_num(array, False, -1)
            return _v_to_int_str(array)
        return _v_to_float_str(array)
    else:
        return array


def to_pandas(array, *axes) -> Reference:
    """
    Convert a numpy array to pandas.Series if 1-dimensional or to a
    pandas.DataFrame if 2-dimensional.

    Provide index and columns if needed
    """
    return pd.Series(array, *axes) if array.ndim == 1 else pd.DataFrame(array, *axes)


def linear_expression(*tuples, as_pandas=True, return_axes=False):
    """
    Elementwise concatenation of tuples in the form (coefficient, variables).
    Coefficient and variables can be arrays, series or frames. Per default
    returns a pandas.Series or pandas.DataFrame of strings. If return_axes is
    set to True the return value is split into values and axes, where values
    are the numpy.array and axes a tuple containing index and column if
    present.

    Parameters
    ----------
    tuples: tuple of tuples
        Each tuple must of the form (coeff, var), where

        * coeff is a numerical  value, or a numerical array, series, frame
        * var is a str or a array, series, frame of variable strings
    as_pandas : bool, default True
        Whether to return to resulting array as a series, if 1-dimensional, or
        a frame, if 2-dimensional. Supersedes return_axes argument.
    return_axes: Boolean, default False
        Whether to return index and column (if existent)

    Example
    -------
    Initialize coefficients and variables

    >>> coeff1 = 1
    >>> var1 = pd.Series(['a1', 'a2', 'a3'])
    >>> coeff2 = pd.Series([-0.5, -0.3, -1])
    >>> var2 = pd.Series(['b1', 'b2', 'b3'])

    Create the linear expression strings

    >>> linear_expression((coeff1, var1), (coeff2, var2))
    0    +1.0 a1 -0.5 b1
    1    +1.0 a2 -0.3 b2
    2    +1.0 a3 -1.0 b3
    dtype: object

    For a further step the resulting frame can be used as the lhs of
    :func:`pypsa.linopt.define_constraints`

    For retrieving only the values:

    >>> linear_expression((coeff1, var1), (coeff2, var2), as_pandas=False)
    array(['+1.0 a1 -0.5 b1', '+1.0 a2 -0.3 b2', '+1.0 a3 -1.0 b3'], dtype=object)
    """
    axes, shape = _broadcasted_axes(*tuples)
    expression = np.repeat("", np.prod(shape)).reshape(shape).astype(object)
    if np.prod(shape):
        for coefficient, var in tuples:
            new_expression = str_array(coefficient) + " x" + str_array(var, True) + "\n"
            if isinstance(expression, np.ndarray):
                isna = np.isnan(coefficient) | np.isnan(var) | (var == -1)
                new_expression = np.where(isna, "", new_expression)
            expression = expression + new_expression
    if return_axes:
        return [expression, *axes]
    if as_pandas:
        return to_pandas(expression, *axes)
    return expression


def join_expressions(df) -> str:
    """
    Helper function to join arrays, series or frames of strings together.
    """
    return "".join(np.asarray(df).flatten())


# TODO use generics that enforce uniform input
Bound = Sense = Union[pd.Series, pd.DataFrame, np.array, str, float]
Axes = Union[pd.Index, Tuple[pd.Index, ...]]
Mask = Union[pd.DataFrame, np.array]
Terms = Union[str, np.array, pd.Series, pd.DataFrame]

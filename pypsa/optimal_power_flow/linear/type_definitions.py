from typing import Union

import pandas as pd

Snapshots = Union[pd.Index, pd.MultiIndex, pd.DatetimeIndex, pd.RangeIndex]
Duals = pd.Series

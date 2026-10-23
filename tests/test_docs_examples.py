import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize(
    "result_name, data_name, frame_name",
    [
        ("llwc", "largest_locally_weighted_components", "llwc1_df"),
        ("exp_var_ratio", "explained_variance_ratio", "evr1_df"),
    ],
)
def test_gwpca_example_dataframe_coordinate_columns(result_name, data_name, frame_name):
    result = xr.DataArray(
        [[1, 2], [3, 4], [5, 6]],
        dims=("station", "mode"),
        coords={
            "mode": [1, 2],
            "x": ("station", [10.0, 30.0, 20.0]),
            "y": ("station", [40.0, 20.0, 30.0]),
        },
        name=data_name,
    ).set_index(station=("x", "y"))

    example = (
        Path(__file__).resolve().parents[1]
        / "docs/content/user_guide/examples/1single/plot_gwpca.py"
    )
    # Run the actual conversion without the example's R downloads or plotting.
    conversion = next(
        statement.value
        for statement in ast.parse(example.read_text()).body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == frame_name
    )
    frame = eval(
        compile(ast.Expression(conversion), str(example), "eval"),
        {result_name: result},
    )

    np.testing.assert_array_equal(frame["x"], result.x)
    np.testing.assert_array_equal(frame["y"], result.y)
    np.testing.assert_array_equal(frame[data_name], result.sel(mode=1))

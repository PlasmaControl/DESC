"""Test for neoclassical transport compute functions."""

import numpy as np
import pytest

from desc.compute.bounce_integral import desc_grid_from_field_line_coords
from desc.examples import get


@pytest.mark.unit
def test_effective_ripple():
    """Compare DESC effective ripple against neo stellopt."""
    eq = get("HELIOTRON")
    grid_desc, grid_fl = desc_grid_from_field_line_coords(eq)
    # just want to pass some custom keyword arguments into compute func
    data = eq.compute(["ripple", "effective ripple"], grid=grid_desc)
    assert np.isfinite(data["ripple"]).all()
    assert np.isfinite(data["effective ripple"]).all()

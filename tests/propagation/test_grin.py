"""Unit tests for the GrinPropagation model."""

from __future__ import annotations

import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials.gradient_material import GradientMaterial
from optiland.materials.ideal import IdealMaterial
from optiland.propagation.grin import GRINPropagation
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

from ..utils import assert_allclose


def test_grin_propagation_executes_without_error(set_test_backend):
    """Verify that GRIN propagation reaches the exit plane."""
    grin_material = GradientMaterial(n0=1.5, nr2=-0.001)
    distance = 10.0

    surface_in = Surface(
        previous_surface=None,
        material_post=grin_material,
        geometry=StandardGeometry(CoordinateSystem(), radius=be.inf),
    )
    surface_out = Surface(
        previous_surface=surface_in,
        material_post=IdealMaterial(n=1.0),
        geometry=StandardGeometry(CoordinateSystem(z=distance), radius=be.inf),
    )
    rays_in = RealRays(
        x=[0.0, 1.0],
        y=[0.0, 0.0],
        z=[0.0, 0.0],
        L=[0.0, 0.0],
        M=[0.0, 0.0],
        N=[1.0, 1.0],
        intensity=[1.0, 1.0],
        wavelength=[0.55, 0.55],
    )
    model = GRINPropagation(step_size=0.1, max_steps=1000)

    try:
        rays_out = model.propagate(rays_in, surface_in, surface_out)
    except Exception as exc:
        pytest.fail(f"GRINPropagation.propagate raised an exception: {exc}")

    assert_allclose(rays_out.z, be.full_like(rays_out.z, distance), atol=1e-5)
    assert_allclose(rays_out.x[0], be.array(0.0), atol=1e-9)
    assert_allclose(rays_out.y[0], be.array(0.0), atol=1e-9)

    final_x_off_axis = be.to_numpy(rays_out.x)[1]
    initial_x_off_axis = be.to_numpy(rays_in.x)[1]
    assert final_x_off_axis < initial_x_off_axis

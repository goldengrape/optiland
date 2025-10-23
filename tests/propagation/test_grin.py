# path: tests/propagation/test_grin.py

"""Unit tests for the GrinPropagation model."""
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
    """
    Verify that GRINPropagation completes a propagation between two surfaces
    without raising an error, and the final z-coordinate is correct.
    """
    # 1. SETUP
    # Define a simple GRIN medium (parabolic index profile, focusing)
    grin_material = GradientMaterial(n0=1.5, nr2=-0.001)
    
    # Define entry and exit surfaces
    distance = 10.0
    
    surface_in = Surface(
        geometry=StandardGeometry(CoordinateSystem(), radius=be.inf),
        material_pre=IdealMaterial(n=1.0),
        material_post=grin_material
    )
    
    surface_out = Surface(
        geometry=StandardGeometry(CoordinateSystem(z=distance), radius=be.inf),
        material_pre=grin_material,
        material_post=IdealMaterial(n=1.0)
    )
    
    # Create rays starting at the origin, one on-axis, one off-axis
    rays_in = RealRays(
        x=[0.0, 1.0], y=[0.0, 0.0], z=[0.0, 0.0],
        L=[0.0, 0.0], M=[0.0, 0.0], N=[1.0, 1.0],
        intensity=[1.0, 1.0], wavelength=[0.55, 0.55]
    )

    # Instantiate the propagation model
    model = GRINPropagation(step_size=0.1, max_steps=1000)

    # 2. ACTION
    # Propagate the rays. This should complete without exceptions.
    try:
        rays_out = model.propagate(rays_in, surface_in, surface_out)
    except Exception as e:
        pytest.fail(f"GRINPropagation.propagate raised an unexpected exception: {e}")

    # 3. VERIFICATION
    # The most basic check: did the rays arrive at the exit surface?
    assert_allclose(rays_out.z, be.full_like(rays_out.z, distance), atol=1e-5)
    
    # Sanity check: the on-axis ray should not have moved in x or y.
    assert_allclose(rays_out.x[0], be.array(0.0), atol=1e-9)
    assert_allclose(rays_out.y[0], be.array(0.0), atol=1e-9)
    
    # Sanity check: the off-axis ray should have been bent towards the axis.
    # Its final x position should be less than its initial x position.
    final_x_off_axis = be.to_numpy(rays_out.x)[1]
    initial_x_off_axis = be.to_numpy(rays_in.x)[1]
    assert final_x_off_axis < initial_x_off_axis
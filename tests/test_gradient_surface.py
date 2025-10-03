import pytest
import optiland.backend as be
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.gradient_surface import GradientBoundarySurface
from optiland.geometries.standard import StandardGeometry
from optiland.materials import IdealMaterial

def test_gradient_boundary_surface_inheritance():
    """Tests that GradientBoundarySurface is a subclass of Surface."""
    assert issubclass(GradientBoundarySurface, Surface)

def test_gradient_boundary_surface_default_instantiation():
    """Tests that GradientBoundarySurface can be instantiated with default values."""
    try:
        gbs = GradientBoundarySurface()
    except Exception as e:
        pytest.fail(f"Default instantiation failed: {e}")

    assert isinstance(gbs.geometry, StandardGeometry)
    assert be.isinf(gbs.geometry.radius)
    assert gbs.geometry.k == 0.0
    assert gbs.thickness == 0.0
    assert gbs.aperture is None
    assert isinstance(gbs.material_pre, IdealMaterial)
    assert gbs.material_pre.n(0.55) == 1.0
    assert isinstance(gbs.material_post, IdealMaterial)
    assert gbs.material_post.n(0.55) == 1.5

def test_gradient_boundary_surface_specific_instantiation():
    """Tests that GradientBoundarySurface can be instantiated with specific parameters."""
    radius = 50.0
    thickness = 5.0
    semi_diameter = 10.0
    conic = -0.5
    material_pre = IdealMaterial(n=1.1)
    material_post = IdealMaterial(n=1.6)

    gbs = GradientBoundarySurface(
        radius_of_curvature=radius,
        thickness=thickness,
        semi_diameter=semi_diameter,
        conic=conic,
        material_pre=material_pre,
        material_post=material_post,
    )

    # Check geometry attributes
    assert isinstance(gbs.geometry, StandardGeometry)
    assert gbs.geometry.radius == radius
    assert gbs.geometry.k == conic

    # Check surface attributes
    assert gbs.thickness == thickness
    assert gbs.aperture.semi_diameter == semi_diameter

    # Check materials
    assert gbs.material_pre == material_pre
    assert gbs.material_post == material_post

def test_gradient_boundary_surface_is_a_marker_type():
    """
    Tests that GradientBoundarySurface is a distinct type for identification purposes.
    """
    gbs = GradientBoundarySurface()
    # A default Surface cannot be instantiated without arguments, so we cannot compare directly.
    # Instead, we check the type and inheritance.

    assert type(gbs) is GradientBoundarySurface
    assert isinstance(gbs, Surface)
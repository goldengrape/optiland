# path: tests/propagation/test_homogeneous.py

"""Unit tests for the HomogeneousPropagation model."""

import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface
from ..utils import assert_allclose

# --- Helper function to create test surfaces ---

def create_test_surfaces(distance: float, material_medium: IdealMaterial):
    """Creates a pair of planar surfaces separated by a distance."""
    # surface_in is at z=0, with the propagation medium as its post-material.
    surface_in = Surface(
        geometry=StandardGeometry(coordinate_system=CoordinateSystem(), radius=be.inf),
        material_pre=IdealMaterial(n=1.0), # Air
        material_post=material_medium
    )
    
    # surface_out is at z=distance.
    cs_out = CoordinateSystem(z=distance)
    surface_out = Surface(
        geometry=StandardGeometry(coordinate_system=cs_out, radius=be.inf),
        material_pre=material_medium,
        material_post=IdealMaterial(n=1.0) # Air
    )
    return surface_in, surface_out


def test_homogeneous_propagation_position_update(set_test_backend):
    """Verify that ray coordinates are updated correctly."""
    # setup
    rays_in = RealRays(x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5])
    material = IdealMaterial(n=1.5)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)
    
    # action
    rays_out = model.propagate(rays_in, surface_in, surface_out)
    
    # verification
    assert_allclose(rays_out.x, be.array([0.0]))
    assert_allclose(rays_out.y, be.array([0.0]))
    assert_allclose(rays_out.z, be.array([distance]))


def test_homogeneous_propagation_no_attenuation_with_k0(set_test_backend):
    """Verify ray intensity is unchanged when k=0."""
    # setup
    rays_in = RealRays(x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5])
    material = IdealMaterial(n=1.0, k=0.0)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    initial_intensity = be.copy(rays_in.i)
    
    # action
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    # verification
    assert_allclose(rays_out.i, initial_intensity)


def test_homogeneous_propagation_attenuation_with_k_gt_0(set_test_backend):
    """Verify ray intensity is correctly attenuated when k > 0."""
    # setup
    rays_in = RealRays(x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5])
    k_val = 0.1
    distance = 10.0  # distance in mm
    wavelength = be.to_numpy(rays_in.w)[0]

    material = IdealMaterial(n=1.0, k=k_val)
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    initial_intensity = be.to_numpy(rays_in.i)[0]

    # action
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    # verification
    # Calculate expected intensity based on Beer-Lambert law
    # alpha = 4 * pi * k / lambda
    # I = I_0 * exp(-alpha * z)
    # distance is in mm, wavelength w is in um. Convert distance to um.
    alpha = 4 * be.pi * k_val / wavelength
    expected_intensity = initial_intensity * be.exp(-alpha * distance * 1e3)

    assert_allclose(rays_out.i, be.array([expected_intensity]))


def test_homogeneous_propagation_normalizes_rays(set_test_backend):
    """Verify that unnormalized rays are normalized after propagation."""
    # setup
    rays_in = RealRays(x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5])
    material = IdealMaterial(n=1.0)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    # Manually un-normalize the rays
    rays_in.L = rays_in.L * 2.0
    rays_in.is_normalized = False

    # action
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    # verification
    assert rays_out.is_normalized is True
    magnitude = be.sqrt(rays_out.L**2 + rays_out.M**2 + rays_out.N**2)
    assert_allclose(magnitude, be.array([1.0]))
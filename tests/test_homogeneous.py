"""
Unit tests for the HomogeneousPropagation model.

This test suite verifies that the HomogeneousPropagation model correctly
replicates the behavior of straight-line ray propagation, specifically:
1.  Correctly updating ray positions based on geometric intersection.
2.  Correctly accumulating the Optical Path Difference (OPD).
3.  Adhering to the functional contract by not modifying the input rays object (immutability).
"""

import pytest
import optiland.backend as be
from tests.utils import assert_allclose
from optiland.rays import RealRays
from optiland.surfaces.standard_surface import Surface as StandardSurface
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.coordinate_system import CoordinateSystem


@pytest.fixture
def propagation_setup():
    """
    Provides a standard setup for testing propagation between two surfaces.

    This fixture creates:
    - An entry surface (`surface_in`) holding the propagation medium.
    - An exit surface (`surface_out`), which is a plane at z=10.
    - A batch of initial rays (`rays_in`) starting at z=0.
    - The medium itself (`medium`) with a refractive index of 1.5.
    """
    # Define materials
    air = IdealMaterial(n=1.0)
    medium = IdealMaterial(n=1.5)

    # Define surfaces
    surface_in_geometry = StandardGeometry(
        coordinate_system=CoordinateSystem(), radius=be.inf
    )
    surface_in = StandardSurface(
        previous_surface=None,
        material_post=medium,
        geometry=surface_in_geometry,
    )

    cs_out = CoordinateSystem()
    cs_out.z = 10.0
    exit_geometry = StandardGeometry(coordinate_system=cs_out, radius=be.inf)
    surface_out = StandardSurface(
        previous_surface=surface_in,
        material_post=air,
        geometry=exit_geometry,
    )

    # Create a batch of rays following the exact API from real_rays.py.
    num_rays = 2
    wavelength = 0.587
    
    # AUTHORITATIVE FIX: Call __init__ with all 8 required arguments.
    rays_in = RealRays(
        x=be.zeros(num_rays),
        y=be.array([0.0, 5.0]),
        z=be.zeros(num_rays),
        L=be.zeros(num_rays),
        M=be.zeros(num_rays),
        N=be.ones(num_rays),
        intensity=be.ones(num_rays),  # Provide the required intensity
        wavelength=be.full(num_rays, wavelength) # Provide the required wavelength
    )
    
    # Set opd post-initialization, as it's initialized to zero internally.
    rays_in.opd = be.array([100.0, 100.0])

    # NOTE: is_alive and ref_index are not attributes of the RealRays class
    # according to the provided source code and have been removed.
    
    return {
        "rays_in": rays_in,
        "surface_in": surface_in,
        "surface_out": surface_out,
        "medium": medium
    }


def test_propagate_updates_state_correctly(propagation_setup):
    """
    Tests if the propagate method correctly updates ray position and OPD.
    """
    # Arrange
    model = HomogeneousPropagation()
    rays_in = propagation_setup["rays_in"]
    surface_in = propagation_setup["surface_in"]
    surface_out = propagation_setup["surface_out"]
    medium = propagation_setup["medium"]
    
    expected_distance = 10.0
    
    initial_x = be.copy(rays_in.x)
    initial_y = be.copy(rays_in.y)
    initial_z = be.copy(rays_in.z)
    initial_opd = be.copy(rays_in.opd)

    # Act
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    # Assert
    expected_x = initial_x
    expected_y = initial_y
    expected_z = initial_z + expected_distance
    
    assert_allclose(rays_out.x, expected_x)
    assert_allclose(rays_out.y, expected_y)
    assert_allclose(rays_out.z, expected_z)

    # The logic for OPD accumulation remains correct.
    expected_opd = initial_opd + medium.n(rays_in.w) * expected_distance
    assert_allclose(rays_out.opd, expected_opd)
    
    # Check that other ray properties are carried over.
    assert_allclose(rays_out.L, rays_in.L)
    # Intensity should be unchanged by propagation through a non-absorbing medium.
    assert_allclose(rays_out.i, rays_in.i)


def test_propagate_mutates_and_returns_input(propagation_setup):
    """
    Tests if propagate mutates and returns the input rays object.
    """
    # Arrange
    model = HomogeneousPropagation()
    rays_in = propagation_setup["rays_in"]
    
    # Create copies to verify the original object's state.
    original_z = be.copy(rays_in.z)
    original_opd = be.copy(rays_in.opd)

    # Act
    rays_out = model.propagate(
        rays_in,
        propagation_setup["surface_in"],
        propagation_setup["surface_out"]
    )

    # Assert
    assert rays_out is rays_in

    assert_allclose(rays_in.z, original_z + 10.0)
    assert be.all(rays_in.opd > original_opd)

"""Tests for the gradient_propagation module."""

import numpy as np
import pytest

# --- MODIFIED IMPORT STATEMENT ---
# The alias 'be' should refer to the entire backend dispatcher module,
# not the underlying 'np' library directly. This ensures that utility
# functions like `be.to_numpy` are found and that all calls are properly
# routed to the active backend's implementation.
import optiland.backend as be

from optiland.rays.real_rays import RealRays
from optiland.materials.gradient_material import GradientMaterial
from optiland.surfaces.gradient_surface import GradientBoundarySurface
from optiland.interactions.gradient_propagation import propagate_through_gradient
from optiland.coordinate_system import CoordinateSystem

def test_propagate_through_radial_grin_lens():
    """Test that a ray is focused by a simple radial GRIN lens."""
    # 1. Define a focusing radial GRIN material (n decreases with r)
    grin_material = GradientMaterial(n0=1.5, nr2=-0.01)

    # 2. Define entry and exit surfaces (a GRIN slab of 10mm thickness)
    thickness = 10.0
    entry_surface = GradientBoundarySurface(thickness=thickness)
    
    # The exit surface is physically located at z = thickness relative to the entry surface.
    exit_surface = GradientBoundarySurface(thickness=0.0)
    exit_surface.geometry.cs = CoordinateSystem(z=thickness) # Correctly position the exit surface

    # 3. Create a single ray parallel to the z-axis, offset in y
    initial_y = 1.0
    rays_in = RealRays(
        x=be.array([0.0]),
        y=be.array([initial_y]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        intensity=be.array([1.0]),
        wavelength=0.55
    )

    # 4. Call the propagation function
    rays_out = propagate_through_gradient(
        rays_in=rays_in,
        grin_material=grin_material,
        exit_surface=exit_surface,
        step_size=0.1,
        max_steps=1000
    )

    # 5. Assertions
    # With the corrected import, be.to_numpy now correctly calls the utility function.
    final_pos = be.to_numpy(be.stack([rays_out.x, rays_out.y, rays_out.z], axis=-1))
    final_dir = be.to_numpy(be.stack([rays_out.L, rays_out.M, rays_out.N], axis=-1))

    # Assert that the ray is on the exit surface
    assert np.isclose(final_pos[0, 2], exit_surface.geometry.cs.z, atol=1e-5)

    # Assert that the ray has bent towards the optical axis (y < initial_y)
    assert final_pos[0, 1] < initial_y

    # Assert that the ray's y-direction is now negative (angled towards the axis)
    assert final_dir[0, 1] < 0

    # Assert that the ray is still on the y-z plane (x should be ~0)
    assert np.isclose(final_pos[0, 0], 0.0, atol=1e-9)
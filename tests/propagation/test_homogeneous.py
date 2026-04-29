"""Unit tests for the HomogeneousPropagation model."""

from __future__ import annotations

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials.ideal import IdealMaterial
from optiland.propagation.homogeneous import HomogeneousPropagation
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

from ..utils import assert_allclose


def create_test_surfaces(
    distance: float, material_medium: IdealMaterial
) -> tuple[Surface, Surface]:
    """Create two planar surfaces separated by distance."""
    surface_in = Surface(
        previous_surface=None,
        material_post=material_medium,
        geometry=StandardGeometry(coordinate_system=CoordinateSystem(), radius=be.inf),
    )
    surface_out = Surface(
        previous_surface=surface_in,
        material_post=IdealMaterial(n=1.0),
        geometry=StandardGeometry(
            coordinate_system=CoordinateSystem(z=distance),
            radius=be.inf,
        ),
    )
    return surface_in, surface_out


def test_homogeneous_propagation_position_update(set_test_backend):
    """Verify that ray coordinates are updated correctly."""
    rays_in = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.5)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    rays_out = model.propagate(rays_in, surface_in, surface_out)

    assert_allclose(rays_out.x, be.array([0.0]))
    assert_allclose(rays_out.y, be.array([0.0]))
    assert_allclose(rays_out.z, be.array([distance]))


def test_homogeneous_propagation_no_attenuation_with_k0(set_test_backend):
    """Verify ray intensity is unchanged when k=0."""
    rays_in = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.0, k=0.0)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    initial_intensity = be.copy(rays_in.i)
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    assert_allclose(rays_out.i, initial_intensity)


def test_homogeneous_propagation_attenuation_with_k_gt_0(set_test_backend):
    """Verify ray intensity is correctly attenuated when k > 0."""
    rays_in = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    k_val = 0.1
    distance = 10.0
    wavelength = be.to_numpy(rays_in.w)[0]
    material = IdealMaterial(n=1.0, k=k_val)
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    initial_intensity = be.to_numpy(rays_in.i)[0]
    rays_out = model.propagate(rays_in, surface_in, surface_out)

    alpha = 4 * be.pi * k_val / wavelength
    expected_intensity = initial_intensity * be.exp(-alpha * distance * 1e3)
    assert_allclose(rays_out.i[0], expected_intensity)


def test_homogeneous_propagation_normalizes_rays(set_test_backend):
    """Verify that unnormalized rays are normalized after propagation."""
    rays_in = RealRays(
        x=[0], y=[0], z=[0], L=[0], M=[0], N=[1], intensity=[1.0], wavelength=[0.5]
    )
    material = IdealMaterial(n=1.0)
    distance = 10.0
    surface_in, surface_out = create_test_surfaces(distance, material)
    model = HomogeneousPropagation(material)

    rays_in.L = be.array([0.0])
    rays_in.M = be.array([0.0])
    rays_in.N = be.array([2.0])

    rays_out = model.propagate(rays_in, surface_in, surface_out)

    assert_allclose(rays_out.N, be.array([1.0]))
    norm = be.sqrt(rays_out.L**2 + rays_out.M**2 + rays_out.N**2)
    assert_allclose(norm, be.array([1.0]))

"""Unit tests for the GradientMaterial class."""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError
import icontract

from optiland.materials.gradient_material import GradientMaterial


@pytest.fixture
def grin_material_radial():
    """A fixture for a GRIN material with only radial components."""
    return GradientMaterial(n0=1.5, nr2=0.1, nr4=0.01)


@pytest.fixture
def grin_material_axial():
    """A fixture for a GRIN material with only axial components."""
    return GradientMaterial(n0=1.6, nz1=0.2, nz2=0.05)


def test_gradient_material_initialization():
    """Test the default and custom initialization of GradientMaterial."""
    # Test default initialization
    default_mat = GradientMaterial()
    assert default_mat.n0 == 1.0
    assert default_mat.nr2 == 0.0
    assert default_mat.name == "GRIN Material"

    # Test custom initialization
    custom_mat = GradientMaterial(n0=1.5, nr2=0.1, name="Custom GRIN")
    assert custom_mat.n0 == 1.5
    assert custom_mat.nr2 == 0.1
    assert custom_mat.name == "Custom GRIN"


def test_immutability(grin_material_radial):
    """Test that the GradientMaterial instance is immutable."""
    with pytest.raises(FrozenInstanceError):
        grin_material_radial.n0 = 1.6


def test_get_index_on_axis(grin_material_radial, grin_material_axial):
    """Test get_index method at coordinates on the z-axis (x=0, y=0)."""
    # For radial material, index on-axis should just be n0
    assert grin_material_radial.get_index(0, 0, 0) == pytest.approx(1.5)
    assert grin_material_radial.get_index(0, 0, 5) == pytest.approx(1.5)

    # For axial material, index on-axis depends on z
    assert grin_material_axial.get_index(0, 0, 0) == pytest.approx(1.6)
    # n = 1.6 + 0.2*z + 0.05*z^2 => 1.6 + 0.2*2 + 0.05*4 = 1.6 + 0.4 + 0.2 = 2.2
    assert grin_material_axial.get_index(0, 0, 2) == pytest.approx(2.2)


def test_get_index_off_axis(grin_material_radial, grin_material_axial):
    """Test get_index method at off-axis coordinates."""
    # For radial material, r^2 = 1^2 + 2^2 = 5
    # n = 1.5 + 0.1*r^2 + 0.01*r^4 = 1.5 + 0.1*5 + 0.01*25 = 1.5 + 0.5 + 0.25 = 2.25
    assert grin_material_radial.get_index(1, 2, 5) == pytest.approx(2.25)

    # For axial material, index should not depend on r
    assert grin_material_axial.get_index(1, 2, 0) == pytest.approx(1.6)


def test_get_gradient_on_axis(grin_material_radial, grin_material_axial):
    """Test get_gradient method on the z-axis (x=0, y=0)."""
    # For radial material, gradient on-axis should be zero in x and y
    grad_radial = grin_material_radial.get_gradient(0, 0, 5)
    np.testing.assert_allclose(grad_radial, [0, 0, 0])

    # For axial material, gradient on-axis should be zero in x and y
    # dn/dz = nz1 + 2*nz2*z = 0.2 + 2*0.05*z = 0.2 + 0.1*z
    # At z=3, dn/dz = 0.2 + 0.3 = 0.5
    grad_axial = grin_material_axial.get_gradient(0, 0, 3)
    np.testing.assert_allclose(grad_axial, [0, 0, 0.5])


def test_get_gradient_off_axis(grin_material_radial):
    """Test get_gradient method at off-axis coordinates."""
    x, y, z = 1, 2, 5
    r2 = x**2 + y**2  # 5

    # dn/dr2 = nr2 + 2*nr4*r2 = 0.1 + 2*0.01*5 = 0.1 + 0.1 = 0.2
    # dn/dx = 2*x*dn/dr2 = 2*1*0.2 = 0.4
    # dn/dy = 2*y*dn/dr2 = 2*2*0.2 = 0.8
    # dn/dz = 0
    expected_gradient = [0.4, 0.8, 0.0]
    calculated_gradient = grin_material_radial.get_gradient(x, y, z)
    np.testing.assert_allclose(calculated_gradient, expected_gradient)


def test_get_index_and_gradient(grin_material_radial):
    """Test that get_index_and_gradient returns consistent results."""
    x, y, z = 1, 2, 5
    index, gradient = grin_material_radial.get_index_and_gradient(x, y, z)

    expected_index = grin_material_radial.get_index(x, y, z)
    expected_gradient = grin_material_radial.get_gradient(x, y, z)

    assert index == pytest.approx(expected_index)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_icontract_invariant_violation():
    """Test that creating a material with non-numeric coefficients raises an error."""
    with pytest.raises(icontract.errors.ViolationError):
        GradientMaterial(n0="not a number")


def test_icontract_require_violation(grin_material_radial):
    """Test that calling methods with invalid inputs raises an error."""
    with pytest.raises(icontract.errors.ViolationError):
        grin_material_radial.get_index("a", "b", "c")

    with pytest.raises(icontract.errors.ViolationError):
        grin_material_radial.get_gradient(None, 1, 2)
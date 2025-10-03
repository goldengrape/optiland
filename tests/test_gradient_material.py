"""Unit tests for the GradientMaterial class."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import is_dataclass, FrozenInstanceError

from optiland.materials.gradient_material import GradientMaterial


class TestGradientMaterial:
    """Test suite for the GradientMaterial class."""

    @pytest.fixture
    def sample_material(self):
        """A sample GradientMaterial for testing."""
        return GradientMaterial(
            n0=1.5,
            nr2=0.1,
            nr4=0.01,
            nr6=0.001,
            nz1=0.2,
            nz2=0.02,
            nz3=0.002,
            name="Test GRIN Material",
        )

    def test_initialization_default(self):
        """Test default initialization of GradientMaterial."""
        material = GradientMaterial()
        assert material.n0 == 1.0
        assert material.nr2 == 0.0
        assert material.nr4 == 0.0
        assert material.nr6 == 0.0
        assert material.nz1 == 0.0
        assert material.nz2 == 0.0
        assert material.nz3 == 0.0
        assert material.name == "GRIN Material"

    def test_initialization_custom(self, sample_material):
        """Test custom initialization of GradientMaterial."""
        assert sample_material.n0 == 1.5
        assert sample_material.nr2 == 0.1
        assert sample_material.nr4 == 0.01
        assert sample_material.nr6 == 0.001
        assert sample_material.nz1 == 0.2
        assert sample_material.nz2 == 0.02
        assert sample_material.nz3 == 0.002
        assert sample_material.name == "Test GRIN Material"

    def test_get_index(self, sample_material):
        """Test the get_index method."""
        x, y, z = 1, 2, 3
        r2 = x**2 + y**2
        expected_n = (
            sample_material.n0
            + sample_material.nr2 * r2
            + sample_material.nr4 * r2**2
            + sample_material.nr6 * r2**3
            + sample_material.nz1 * z
            + sample_material.nz2 * z**2
            + sample_material.nz3 * z**3
        )
        n = sample_material.get_index(x, y, z)
        assert_allclose(n, expected_n)

    def test_get_index_at_origin(self, sample_material):
        """Test get_index at the origin (0,0,0)."""
        n = sample_material.get_index(0, 0, 0)
        assert_allclose(n, sample_material.n0)

    def test_get_gradient(self, sample_material):
        """Test the get_gradient method."""
        x, y, z = 1, 2, 3
        r2 = x**2 + y**2

        # Expected gradient components
        dn_dr2 = (
            sample_material.nr2
            + 2 * sample_material.nr4 * r2
            + 3 * sample_material.nr6 * r2**2
        )
        expected_dn_dx = 2 * x * dn_dr2
        expected_dn_dy = 2 * y * dn_dr2
        expected_dn_dz = (
            sample_material.nz1
            + 2 * sample_material.nz2 * z
            + 3 * sample_material.nz3 * z**2
        )

        grad_n = sample_material.get_gradient(x, y, z)
        expected_grad = np.array([expected_dn_dx, expected_dn_dy, expected_dn_dz])

        assert_allclose(grad_n, expected_grad)

    def test_get_gradient_at_origin(self, sample_material):
        """Test get_gradient at the origin (0,0,0)."""
        grad_n = sample_material.get_gradient(0, 0, 0)
        expected_grad = np.array([0, 0, sample_material.nz1])
        assert_allclose(grad_n, expected_grad)

    def test_get_index_and_gradient(self, sample_material):
        """Test the get_index_and_gradient method for consistency."""
        x, y, z = 1, 2, 3

        # Calculate using combined method
        n_combined, grad_combined = sample_material.get_index_and_gradient(x, y, z)

        # Calculate using separate methods
        n_separate = sample_material.get_index(x, y, z)
        grad_separate = sample_material.get_gradient(x, y, z)

        # Check for consistency
        assert_allclose(n_combined, n_separate)
        assert_allclose(grad_combined, grad_separate)

    def test_immutability(self, sample_material):
        """Test that the dataclass is frozen."""
        assert is_dataclass(sample_material) and sample_material.__dataclass_params__.frozen
        with pytest.raises(FrozenInstanceError):
            sample_material.n0 = 2.0

    def test_n_method_without_coords(self, sample_material):
        """Test the inherited n() method without spatial coordinates."""
        # Should return the base refractive index n0
        assert_allclose(sample_material.n(wavelength=0.5), sample_material.n0)

    def test_n_method_with_coords(self, sample_material):
        """Test the inherited n() method with spatial coordinates."""
        x, y, z = 1, 2, 3
        # Should return the same as get_index
        n_from_n_method = sample_material.n(wavelength=0.5, x=x, y=y, z=z)
        n_from_get_index = sample_material.get_index(x, y, z)
        assert_allclose(n_from_n_method, n_from_get_index)

    def test_k_method(self, sample_material):
        """Test the inherited k() method."""
        # Should always return 0.0
        assert_allclose(sample_material.k(wavelength=0.5), 0.0)
        assert_allclose(sample_material.k(wavelength=0.5, x=1, y=2, z=3), 0.0)

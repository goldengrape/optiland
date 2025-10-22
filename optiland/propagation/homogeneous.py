# path: optiland/propagation/homogeneous.py

"""
Implements the standard straight-line propagation model for homogeneous media.

This module provides the default propagation behavior for rays traveling through
a medium with a uniform refractive index, encapsulating the logic previously
hardcoded in the main trace loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel
from optiland.rays import RealRays

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.surfaces.base import BaseSurface


class HomogeneousPropagation(BasePropagationModel):
    """
    Handles ray propagation in a straight line through a homogeneous,
    isotropic medium.
    """

    def __init__(self, material: "BaseMaterial" | None = None) -> None:
        """
        Initializes the HomogeneousPropagation model.
        
        Args:
            material: The parent material instance. This reference is stored
                      to support serialization and introspection patterns.
        """
        # --- FIX: Store the material reference ---
        # This resolves the AttributeError in serialization tests by ensuring
        # the model holds a back-reference to its parent material as expected.
        self.material = material

    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates rays from an entry surface to an exit surface in a straight line.

        This implementation adheres to the functional contract of immutability and
        correctly handles coordinate system transformations for intersection
        calculations.
        """
        # --- Contract Enforcement: Immutability ---
        # Create a deep copy of the input rays to ensure the original is not modified.
        rays_out = rays_in.copy()
        
        # --- Core Logic ---
        # 1. Calculate the geometric distance to the exit surface.
        #    The surface's geometry object handles any necessary coordinate transforms.
        distance = surface_out.geometry.intersect(rays_in)
        
        # 2. Update the final output ray's positions.
        rays_out.x += distance * rays_out.L
        rays_out.y += distance * rays_out.M
        rays_out.z += distance * rays_out.N

        # 3. Update the Optical Path Difference.
        medium = surface_in.material_post
        n = medium.n(rays_out.w)
        rays_out.opd += n * distance

        return rays_out

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "HomogeneousPropagation":
        """
        Creates a HomogeneousPropagation model from a dictionary.
        
        Args:
            d: The dictionary representation of the model.
            material: The parent material instance, required by the base class factory.

        Returns:
            An instance of the HomogeneousPropagation model.
        """
        return cls(material=material)
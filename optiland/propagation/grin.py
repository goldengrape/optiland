# path: optiland/propagation/grin.py

"""
Implements the propagation model for Graded-Index (GRIN) media.

This module provides the GRINPropagation class, which conforms to the
BasePropagationModel interface. It acts as an adapter, delegating the complex
numerical computation to a dedicated solver function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import icontract

from optiland.propagation.base import BasePropagationModel
from optiland.propagation.gradient_propagation import propagate_through_gradient

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.base import BaseSurface


class GRINPropagation(BasePropagationModel):
    """
    Handles ray propagation through a Graded-Index (GRIN) medium.

    This model uses a fourth-order Runge-Kutta (RK4) numerical integration
    scheme to solve the governing ray equation. It serves as an adapter, 
    configuring and calling a specialized GRIN solver.
    """

    @icontract.require(lambda step_size: step_size > 0)
    @icontract.require(lambda max_steps: max_steps > 0)
    def __init__(self, step_size: float = 0.1, max_steps: int = 10000) -> None:
        """
        Initializes the GRINPropagation model.

        Args:
            step_size: The fixed step size for each RK4 integration step (in mm).
            max_steps: The maximum number of integration steps to prevent
                       infinite loops.
        """
        self.step_size = step_size
        self.max_steps = max_steps

    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates rays from an entry surface to an exit surface through
        a GRIN medium.

        This method validates that the propagation medium is a GradientMaterial
        and then delegates the computation to the core solver function.
        """
        medium = surface_in.material_post
        
        # --- Contract Enforcement: Precondition Check ---
        # Ensure this propagation model is only used with a compatible material.
        if not hasattr(medium, 'get_index_and_gradient'):
             raise TypeError(
                "GRINPropagation can only be used with a material that has a "
                "'get_index_and_gradient' method, such as GradientMaterial."
             )

        # Delegate to the core solver.
        return propagate_through_gradient(
            rays_in,
            medium,
            surface_out,
            self.step_size,
            self.max_steps
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the model and its configuration to a dictionary."""
        return {
            "class": self.__class__.__name__,
            "step_size": self.step_size,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, d: dict, material: "BaseMaterial" = None) -> "GRINPropagation":
        """
        Creates a GRINPropagation model from a dictionary.

        This factory method supports deserialization, allowing configurable
        parameters to be specified in the serialized format.
        """
        step_size = d.get('step_size', 0.1)
        max_steps = d.get('max_steps', 10000)
        return cls(step_size=step_size, max_steps=max_steps)
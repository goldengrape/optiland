# path: optiland/propagation/base.py

"""
Defines the abstract base class for all propagation models.

This module establishes the interface contract for how rays are propagated
between two surfaces within a medium. It uses a factory pattern with a registry
to support dynamic instantiation and deserialization, consistent with the
broader Optiland architecture.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

# Use TYPE_CHECKING to avoid circular imports at runtime, a standard practice.
if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays import RealRays
    from optiland.surfaces.base import BaseSurface


class BasePropagationModel(abc.ABC):
    """
    Abstract base class for all propagation models.

    This class defines the essential `propagate` method and includes a registry-based
    factory pattern to enable deserialization from dictionaries.
    """

    _registry = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically register any subclass into the factory registry.
        
        This allows for dynamic instantiation from serialized data (e.g., a string
        of the class name), decoupling the deserialization logic from specific
        model implementations.
        """
        super().__init_subclass__(**kwargs)
        BasePropagationModel._registry[cls.__name__] = cls

    @abc.abstractmethod
    def propagate(
        self,
        rays_in: "RealRays",
        surface_in: "BaseSurface",
        surface_out: "BaseSurface"
    ) -> "RealRays":
        """
        Propagates a batch of rays from an entry surface to an exit surface.

        This method follows a functional contract: it must not modify the input
        `rays_in` object. It should return a new `RealRays` instance representing
        the state of the rays at the moment they reach the exit surface.

        Args:
            rays_in: The state of the rays immediately after interacting with
                     the entry surface (`surface_in`). This is the input state.
            surface_in: The surface from which the rays are propagating. The
                        `material_post` attribute of this surface defines the
                        medium of propagation.
            surface_out: The destination surface that the rays will intersect.

        Returns:
            A new `RealRays` object representing the final state of the rays
            at the exit surface, before any interaction with it.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the propagation model to a dictionary for persistence.
        
        By default, this stores only the class name, as the model is typically
        reconstructed by its parent material during deserialization.
        """
        return {"class": self.__class__.__name__}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: "BaseMaterial"
    ) -> "BasePropagationModel":
        """
        Deserializes a propagation model from a dictionary using the factory pattern.

        Args:
            d: A dictionary containing the serialized data, including a 'class' key.
            material: The parent material instance. This is passed to the subclass
                      constructor for context, maintaining API compatibility.

        Returns:
            An instance of a specific propagation model subclass.
            
        Raises:
            ValueError: If the class name specified in the dictionary is not found
                        in the registry.
        """
        model_class_name = d.get("class")
        if model_class_name not in cls._registry:
            raise ValueError(f"Unknown propagation model class: {model_class_name}")

        # Look up the specific subclass in the registry.
        model_subclass = cls._registry[model_class_name]

        # Delegate the actual object creation to the subclass's `from_dict` method.
        return model_subclass.from_dict(d, material)
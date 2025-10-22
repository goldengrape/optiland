# path: optiland/surfaces/__init__.py
# flake8: noqa

from .factories.surface_factory import SurfaceFactory
from .image_surface import ImageSurface
from .object_surface import ObjectSurface
from .converters import (
    ParaxialToThickLensConverter,
    convert_to_thick_lens,
)
from .standard_surface import Surface
from .surface_group import SurfaceGroup
from .gradient_surface import GradientBoundarySurface  # Add this line

__all__ = [
    "BasePropagationModel",
    "HomogeneousPropagation",
    "GRINPropagation",
]
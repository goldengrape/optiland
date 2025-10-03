"""Defines a surface that marks the boundary of a gradient-index medium."""

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.surfaces.standard_surface import Surface


class GradientBoundarySurface(Surface):
    """
    A surface that marks the entry into a Gradient Refractive Index (GRIN) medium.

    This class acts as a simplified constructor for a standard surface (one with
    a `StandardGeometry`) that is intended to be used as a boundary for a GRIN
    medium.

    Geometrically, this surface is identical to a standard spherical/conic surface.
    Its primary role is to be a distinct type that can trigger a special
    propagation model in the ray tracing engine. It does not contain any

    physical information about the gradient index itself.
    """

    def __init__(
        self,
        radius_of_curvature=be.inf,
        thickness=0.0,
        semi_diameter=None,
        conic=0.0,
        material_pre=None,
        material_post=None,
        **kwargs,
    ):
        """
        Initializes a GradientBoundarySurface.

        Args:
            radius_of_curvature (float, optional): The radius of curvature.
                Defaults to infinity (a plane).
            thickness (float, optional): The thickness of the material following
                the surface. Defaults to 0.0.
            semi_diameter (float, optional): The semi-diameter of the surface,
                used for aperture clipping. Defaults to None.
            conic (float, optional): The conic constant. Defaults to 0.0.
            material_pre (BaseMaterial, optional): Material before the surface.
                Defaults to ideal air (n=1.0).
            material_post (BaseMaterial, optional): Material after the surface.
                Defaults to a default glass (n=1.5). This will typically be
                replaced by a GradientMaterial by the tracing engine.
            **kwargs: Additional keyword arguments passed to the parent
                `Surface` constructor.
        """
        cs = CoordinateSystem()  # Assumes a simple, non-decentered system
        geometry = StandardGeometry(cs, radius=radius_of_curvature, conic=conic)

        if material_pre is None:
            material_pre = IdealMaterial(n=1.0)
        if material_post is None:
            material_post = IdealMaterial(n=1.5)

        super().__init__(
            geometry=geometry,
            material_pre=material_pre,
            material_post=material_post,
            aperture=semi_diameter * 2 if semi_diameter is not None else None,
            **kwargs,
        )
        self.thickness = thickness
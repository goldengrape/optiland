.. _grin_design_and_implementation:

###################################################
GRIN Lens Support: Design and Implementation Plan
###################################################

:Author: Gemini
:Date: 2025-10-02
:Version: 1.0

This document outlines the user requirements, architectural design, and implementation plan for adding Gradient Refractive Index (GRIN) lens support to Optiland.

.. contents:: Table of Contents
   :local:

********************************
1. User Requirements (URD) - Purpose
********************************

1.1. Problem Statement
======================
Currently, Optiland lacks support for surfaces with a gradient refractive index, such as the ``gradient3`` surface type available in Zemax. This limits the software's capability to design and simulate advanced optical systems that utilize GRIN lenses for superior aberration correction and system simplification (e.g., reducing the number of elements).

1.2. Key Use Cases
==================
The primary motivation for this feature is to enable the accurate modeling of biological optical systems, particularly the human eye. Many established, anatomically correct eye models, such as the **Atchison** and **Liou-Brennan** models, rely on a gradient refractive index to represent the crystalline lens. The absence of this feature is a significant limitation for researchers and engineers in ophthalmology and vision science.

1.3. Desired Solution
=====================
The goal is to implement a new surface or medium type that supports a polynomial-defined gradient refractive index. This includes implementing a robust ray tracing algorithm capable of handling propagation through a medium with a continuously varying refractive index, based on the Runge-Kutta (RK4) method outlined in the original feature request.

**************************************************
2. Architectural Design: An Axiomatic Approach
**************************************************

To ensure a robust, maintainable, and extensible design, we apply the principles of Axiomatic Design.

2.1. Core Principles
====================
*   **Independence Axiom:** Maintain the independence of Functional Requirements (FRs).
*   **Information Axiom:** Minimize complexity by decoupling unrelated concepts.

2.2. Decomposing the Problem
============================
The ``gradient3`` concept combines three independent functional requirements:

*   **FR1:** Define the **geometry** of an optical boundary.
*   **FR2:** Define the **physical properties** of a medium.
*   **FR3:** Define the **behavior** of a ray propagating through that medium.

To satisfy the Independence Axiom, we must create separate Design Parameters (DPs) for each FR.

*   **DP1: A `Surface` Class:** Manages geometry only.
*   **DP2: A `Material` Class:** Manages physical properties (the GRIN coefficients and index calculation).
*   **DP3: A `Propagation` Model:** Manages the ray's behavioral simulation (the RK4 algorithm).

This leads to an ideal, uncoupled design matrix:

.. code-block:: text

      | DP1 (Geometry) | DP2 (Material) | DP3 (Behavior)
---------------------------------------------------------
FR1   |       X        |        0       |        0
FR2   |       0        |        X       |        0
FR3   |       0        |        0       |        X

This decoupled design is simple, easy to test, and highly extensible.

*******************************
3. Module Implementation Plan
*******************************

Based on the axiomatic design, the implementation is broken down into three distinct, independent components.

3.1. DP1: `GradientBoundarySurface` (Geometric Domain)
=======================================================
*   **Responsibility :**
    *   To define the 2D geometric boundary where a GRIN medium begins.
    *   It acts as a "marker" interface for the ray tracing engine. Its geometry is identical to a `StandardSurface`.
*   **Location :**
    *   ``optiland/surfaces/gradient_surface.py``
*   **Proposed Class Definition:**
    .. code-block:: python

        from .standard_surface import StandardSurface

        class GradientBoundarySurface(StandardSurface):
            """
            A surface that marks the entry into a gradient-index medium.
            Its geometric properties are inherited from StandardSurface.
            """
            pass

3.2. DP2: `GradientMaterial` (Physical Property Domain)
========================================================
*   **Responsibility :**
    *   To store the polynomial coefficients (n0, nr2, nr4, nz1, etc.) for the refractive index.
    *   To provide pure functions for calculating the refractive index `n` and its gradient `∇n` at any point `(x, y, z)`. This encapsulates the physics.
*   **Location:**
    *   ``optiland/materials/gradient_material.py``
*   **Proposed Class Definition:**
    .. code-block:: python

        from .material import BaseMaterial

        class GradientMaterial(BaseMaterial):
            """
            A material with a refractive index defined by a polynomial.
            """
            def __init__(self, n0=0.0, nr2=0.0, nr4=0.0, nr6=0.0, nz1=0.0, nz2=0.0, nz3=0.0, **kwargs):
                super().__init__(**kwargs)
                self.n0 = n0
                self.nr2 = nr2
                # ... store all other coefficients

            def get_index_and_gradient(self, x, y, z):
                # ... implementation of n and ∇n calculation
                pass

3.3. DP3: `GradientPropagation` (Behavioral Domain)
====================================================
*   **Responsibility :**
    *   To implement the core ray tracing algorithm within the GRIN medium.
    *   It takes an initial ray, the `GradientMaterial`, and the exit boundary surface as input.
    *   It uses a numerical method (e.g., RK4) to compute the ray's path and returns the final ray state at the exit boundary.
*   **Location:**
    *   ``optiland/interactions/gradient_propagation.py``
*   **Proposed Interface:**
    .. code-block:: python

        def propagate_through_gradient(ray_in, grin_material, exit_surface):
            """
            Traces a ray through a GRIN medium until it hits the exit_surface.

            Implements the RK4 numerical integration algorithm.
            """
            # ... RK4 main loop from the pseudocode
            # ... Intersection checks with exit_surface
            return ray_out

# 角色
你是一位资深的软件架构师，专门为AI编程助手准备精准的开发上下文。

# 任务背景
我有一个大型的代码仓库，现在需要基于一份【新功能/修订设计文档】对其进行修改。由于仓库代码量巨大，直接将整个仓库作为上下文是不现实的。你的任务是 **精准地** 从代码仓库中提取与 **本次特定任务** 相关的最小且必要的上下文，为AI编程助手后续的编码工作铺平道路。

# 核心目标
创建一份“手术刀式”的精准上下文文档。这份文档应只包含与【新功能/修订设计文档】中描述的任务 **直接相关** 的代码和逻辑。AI编程助手将仅依赖此文档来完成指定的开发任务。

# 工作流程
1.  **理解任务 (Analyze the 'What')**: 首先，深入分析【新功能/修订设计文档】，彻底理解要实现的功能或修复的 Bug 的具体要求、范围和目标。
2.  **精准提取上下文 (Extract the 'Where' and 'How')**: 其次，带着对任务的理解去扫描【原repo URL】。识别出为了完成这个任务，需要修改、调用或理解的**所有相关代码部分**（模块、类、函数等）。忽略所有与本次任务无关的代码。

# 关键约束
- **相关性是第一原则**: 你的输出**必须**只包含和本次任务强相关的代码信息。如果一个模块与设计文档中的目标无关，**绝对不要**包含它。
- **深度而非广度**: 不需要对整个项目进行概述。需要的是对相关代码部分的深入、详细的提取。

# 输出文档结构（请严格按照此结构生成）

## 1. 任务目标概述 (Task Objective)
- (用1-2句话总结【新功能/修订设计文档】的核心目标。例如：“为用户登录模块增加SAML 2.0认证方式。”)

## 2. 受影响的核心模块与文件 (Impact Analysis)
- (列出为完成此任务需要关注的文件清单，并简要说明每个文件与此任务的关系)
  - `src/module_a/file_x.py`: 包含需要修改的核心业务逻辑。
  - `src/utils/parser.py`: 提供了需要调用的数据解析函数。

## 3. 精准上下文详情 (Detailed Context)
(对上面列出的每个文件/模块，提供以下详细信息)

### 模块/文件 A: `src/module_a/file_x.py`
- **与任务的关联**: (详细说明为什么这个文件与当前任务相关。例如：“该文件中的 `UserAuth` 类负责用户认证，本次任务需要为其增加一种新的认证方式。”)
- **相关代码片段 (Existing Code)**: (仅粘贴出需要被修改或理解的现有类定义、函数签名或代码块)
  ```python
  class UserAuth:
      def login(self, user, password):
          # ... existing logic ...
````

  - **交互与依赖**: (说明这段代码如何与**其他相关模块**进行交互)

### 模块/文件 B: `src/utils/parser.py`

  - **与任务的关联**: (例如：“新功能需要解析一种新的数据格式，该文件中的 `BaseParser` 类可以被继承和扩展。”)
  - **相关代码片段 (Existing Code)**:
    ```python
    class BaseParser:
        def parse(self, data):
            raise NotImplementedError
    ```

## 4\. 实现建议 (Implementation Guidance)

  - (基于你的分析，为AI程序员提供一个高层次的实现步骤建议。例如：“1. 在 `parser.py` 中创建一个继承自 `BaseParser` 的新类。2. 在 `file_x.py` 的 `UserAuth` 类中添加一个新的方法来调用新的解析器。”)

# 输入信息

【原repo URL】
https://github.com/HarrisonKramer/optiland

【新功能/修订设计文档】
"""
.. _grin_design_and_implementation:

#########################################
Optiland GRIN 功能综合审核与实现指导报告
#########################################

:Author: optiland fork
:Date: 2025-10-02
:Version: 3.0

.. contents:: Table of Contents
   :local:

*************************
1. 概述 (Executive Summary)
*************************

此报告旨在对 Optiland 项目引入梯度折射率 (GRIN) 透镜支持的设计与实现方案进行最终审核。我们综合评估了原始设计文档（版本 1.0）和一份深度评估报告，确认了此功能对于拓展 Optiland 在生物光学（特别是人眼建模）等前沿领域的应用具有重大的战略意义。

原始设计方案的核心优势在于其严格遵循了**公理化设计 (Axiomatic Design)** 原则，将一个复杂问题分解为三个相互独立的模块：几何 (``Surface``)、物理 (``Material``) 和行为 (``Propagation``)。这种解耦的设计思想是构建可维护、可扩展系统的典范，我们对此表示完全认同，并将其作为后续所有技术讨论的基石。

本报告将在该优秀构架的基础上，融合深度评估报告中提出的关键技术挑战与考量，并结合**契约式设计 (Design by Contract)**、**函数式编程**与**数据导向编程**的最佳实践，提供一份更完备、更精确的最终实现蓝图。

*************************
2. 最终架构与模块定义
*************************

我们采纳并优化了原始设计中的三大核心模块。以下是结合了契约式设计和数据导向原则的最终模块定义，旨在确保代码的健壮性、可预测性和优雅性。

====================================================
2.1. DP1: ``GradientBoundarySurface`` (几何域)
====================================================

* **职责**: 此类作为一个标准表面（带有 `StandardGeometry`）的简化构造函数，旨在用作 GRIN 介质的边界。它作为光线追踪引擎的“标记”，用于识别 GRIN 介质的入口。

* **位置**: ``optiland/surfaces/gradient_surface.py``

* **最终代码定义**:

  .. code-block:: python

    """定义标记梯度折射率介质边界的表面。"""

    import optiland.backend as be
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries.standard import StandardGeometry
    from optiland.materials import IdealMaterial
    from optiland.surfaces.standard_surface import Surface


    class GradientBoundarySurface(Surface):
        """
        一个标记梯度折射率 (GRIN) 介质入口的表面。

        此类作为一个标准表面（带有 `StandardGeometry`）的简化构造函数，
        旨在用作 GRIN 介质的边界。

        在几何上，该表面与一个标准的球面/圆锥面相同。
        它的主要作用是作为一个独特的类型，可以在光线追踪引擎中触发特殊的传播模型。
        它本身不包含任何关于梯度折射率的物理信息。
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
            初始化一个 GradientBoundarySurface。

            参数:
                radius_of_curvature (float, optional): 曲率半径。
                    默认为无穷大（平面）。
                thickness (float, optional): 表面后材料的厚度。
                    默认为 0.0。
                semi_diameter (float, optional): 表面的半直径，
                    用于光圈裁剪。默认为 None。
                conic (float, optional): 圆锥常数。默认为 0.0。
                material_pre (BaseMaterial, optional): 表面前的材料。
                    默认为理想空气 (n=1.0)。
                material_post (BaseMaterial, optional): 表面后的材料。
                    默认为默认玻璃 (n=1.5)。这通常会被追踪引擎
                    替换为 GradientMaterial。
                **kwargs: 传递给父类 `Surface` 构造函数的额外关键字参数。
            """
            cs = CoordinateSystem()  # 假设一个简单的、非偏心系统
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

====================================================
2.2. DP2: ``GradientMaterial`` (物理属性域)
====================================================

* **职责**: 封装 GRIN 介质的物理模型，提供折射率及其梯度的计算方法。

* **位置**: ``optiland/materials/gradient_material.py``

* **最终代码定义**:

  .. code-block:: python

    """定义梯度折射率材料及其物理属性的计算。"""

    from dataclasses import dataclass, field
    import icontract
    import numpy as np
    from typing import Tuple

    from optiland.materials.base import BaseMaterial

    @icontract.invariant(
        lambda self: all(isinstance(getattr(self, c), (int, float)) for c in self.__annotations__ if c != 'name'),
        "所有折射率系数必须是数值类型"
    )
    @dataclass(frozen=True)
    class GradientMaterial(BaseMaterial):
        """
        一种由多项式定义的梯度折射率材料。

        折射率 n 的计算公式为：
        n(r, z) = n0 + nr2*r^2 + nr4*r^4 + nr6*r^6 + nz1*z + nz2*z^2 + nz3*z^3
        其中 r^2 = x^2 + y^2。

        所有系数均被视为不可变，以鼓励函数式编程风格。
        """
        n0: float = 1.0
        nr2: float = 0.0
        nr4: float = 0.0
        nr6: float = 0.0
        nz1: float = 0.0
        nz2: float = 0.0
        nz3: float = 0.0
        name: str = "GRIN Material"

        @icontract.require(lambda x, y, z: all(isinstance(v, (int, float, np.ndarray)) for v in [x, y, z]))
        def get_index(self, x: float, y: float, z: float) -> float:
            """
            在给定坐标 (x, y, z) 处计算折射率 n。这是一个纯函数。
            """
            r2 = x**2 + y**2
            n = (self.n0 +
                 self.nr2 * r2 +
                 self.nr4 * r2**2 +
                 self.nr6 * r2**3 +
                 self.nz1 * z +
                 self.nz2 * z**2 +
                 self.nz3 * z**3)
            return float(n)

        @icontract.require(lambda x, y, z: all(isinstance(v, (int, float, np.ndarray)) for v in [x, y, z]))
        @icontract.ensure(lambda result: result.shape == (3,))
        def get_gradient(self, x: float, y: float, z: float) -> np.ndarray:
            """
            在给定坐标 (x, y, z) 处计算折射率的梯度 ∇n = [∂n/∂x, ∂n/∂y, ∂n/∂z]。
            这是一个纯函数。
            """
            r2 = x**2 + y**2
            dn_dr2 = self.nr2 + 2 * self.nr4 * r2 + 3 * self.nr6 * r2**2
            dn_dx = 2 * x * dn_dr2
            dn_dy = 2 * y * dn_dr2
            dn_dz = self.nz1 + 2 * self.nz2 * z + 3 * self.nz3 * z**2
            return np.array([dn_dx, dn_dy, dn_dz], dtype=float)

        def get_index_and_gradient(self, x: float, y: float, z: float) -> Tuple[float, np.ndarray]:
            """
            在一次调用中同时计算折射率 n 和其梯度 ∇n，以优化性能。
            """
            r2 = x**2 + y**2
            n = (self.n0 +
                 self.nr2 * r2 +
                 self.nr4 * r2**2 +
                 self.nr6 * r2**3 +
                 self.nz1 * z +
                 self.nz2 * z**2 +
                 self.nz3 * z**3)

            dn_dr2 = self.nr2 + 2 * self.nr4 * r2 + 3 * self.nr6 * r2**2
            dn_dx = 2 * x * dn_dr2
            dn_dy = 2 * y * dn_dr2
            dn_dz = self.nz1 + 2 * self.nz2 * z + 3 * self.nz3 * z**2

            return float(n), np.array([dn_dx, dn_dy, dn_dz], dtype=float)

====================================================
2.3. DP3: ``GradientPropagation`` (行为域)
====================================================

* **职责**: 实现光线在 GRIN 介质中的传播算法，核心是求解光线轨迹的微分方程。

* **位置**: ``optiland/interactions/gradient_propagation.py``

* **最终代码定义**:

  .. code-block:: python

    """
    实现光线在梯度折射率 (GRIN) 介质中的传播算法。
    采用 RK4 数值积分方法求解光线方程： d/ds(n * dr/ds) = ∇n
    """
    import icontract
    import numpy as np
    from typing import Callable, Tuple

    # 假设 Ray, BaseSurface, GradientMaterial 已在别处定义
    from optiland.rays import Ray
    from optiland.surfaces import BaseSurface
    from optiland.materials.gradient_material import GradientMaterial

    @icontract.require(lambda ray_in: ray_in.position.shape == (3,) and ray_in.direction.shape == (3,))
    @icontract.require(lambda step_size: step_size > 0)
    @icontract.require(lambda max_steps: max_steps > 0)
    @icontract.ensure(lambda result, exit_surface: exit_surface.contains(result.position, tol=1e-6), "光线终点必须在出射面上")
    def propagate_through_gradient(
        ray_in: Ray,
        grin_material: "GradientMaterial",
        exit_surface: "BaseSurface",
        step_size: float = 0.1,
        max_steps: int = 10000
    ) -> Ray:
        """
        通过 GRIN 介质追踪光线，直到与出射面相交。

        Args:
            ray_in: 初始光线状态（位置和方向）。
            grin_material: GRIN 介质的物理模型。
            exit_surface: 标记 GRIN 介质结束的几何表面。
            step_size: RK4 积分的步长 (mm)。
            max_steps: 防止无限循环的最大步数。

        Returns:
            在出射面上的最终光线状态。
        """
        r = ray_in.position.copy()
        n_start, _ = grin_material.get_index_and_gradient(r[0], r[1], r[2])
        k = n_start * ray_in.direction
        opd = 0.0

        def derivatives(current_r: np.ndarray, current_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            n, grad_n = grin_material.get_index_and_gradient(current_r[0], current_r[1], current_r[2])
            dr_ds = current_k / n if n != 0 else np.zeros(3)
            dk_ds = grad_n
            return dr_ds, dk_ds

        for i in range(max_steps):
            n_current = grin_material.get_index(r[0], r[1], r[2])
            
            # RK4 积分步骤
            r1, k1 = derivatives(r, k)
            r2, k2 = derivatives(r + 0.5 * step_size * r1, k + 0.5 * step_size * k1)
            r3, k3 = derivatives(r + 0.5 * step_size * r2, k + 0.5 * step_size * k2)
            r4, k4 = derivatives(r + step_size * r3, k + step_size * k3)

            r_next = r + (step_size / 6.0) * (r1 + 2*r2 + 2*r3 + r4)
            k_next = k + (step_size / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # 累积光程 (OPD)，使用梯形法则估算
            n_next = grin_material.get_index(r_next[0], r_next[1], r_next[2])
            opd += 0.5 * (n_current + n_next) * step_size
            
            # 检查与出射面的交点
            segment_vec = r_next - r
            segment_len = np.linalg.norm(segment_vec)
            if segment_len > 1e-9:
                segment_ray = Ray(position=r, direction=segment_vec / segment_len)
                distance_to_intersect = exit_surface.intersect(segment_ray)

                if 0 < distance_to_intersect <= segment_len:
                    intersection_point = r + distance_to_intersect * segment_ray.direction
                    n_final = grin_material.get_index(intersection_point[0], intersection_point[1], intersection_point[2])
                    final_direction = k_next / n_final
                    
                    # 最终光线
                    ray_out = Ray(position=intersection_point, direction=final_direction / np.linalg.norm(final_direction))
                    ray_out.opd = ray_in.opd + opd # 假设 Ray 对象有 opd 属性
                    return ray_out

            r, k = r_next, k_next

        raise ValueError("光线在达到最大步数后仍未与出射面相交。")

***********************************
3. 关键技术考量与待决议项
***********************************

评估报告精准地指出了从架构设计到工程实现所需关注的核心挑战。这些问题必须在开发过程中得到明确解答，以确保 GRIN 功能的正确性与高效性。

1.  **集成机制**:

      * **问题**: Optiland 的核心光线追迹引擎 (``Optic.trace``) 如何识别并调用 ``propagate_through_gradient``？
      * **建议**: 在光线追迹循环中，应检查当前 surface 是否为 ``GradientBoundarySurface`` 的实例。若是，则其 ``material_post`` 属性应被断言为一个 ``GradientMaterial`` 实例。此时，追迹流程需确定“出射面”（``exit_surface``），然后将控制权转交给 ``propagate_through_gradient``。

2.  **GRIN 区域定义**:

      * **问题**: 如何界定 GRIN 介质的范围？即 ``exit_surface`` 如何确定？
      * **方案 A (推荐)**: 采用成对标记。一个 GRIN 区域由一个 ``GradientBoundarySurface`` (入口) 和序列中的下一个 ``GradientBoundarySurface`` (出口) 界定。这种方式清晰、无歧义。
      * **方案 B**: 从一个 ``GradientBoundarySurface`` 开始，直到下一个表面的 ``material_post`` 不再是 ``GradientMaterial`` 为止。此方案较为灵活，但对系统序列的依赖性更强。
      * **决议**: 建议初期采用方案 A。这可能需要对 ``Optic`` 或 ``SurfaceGroup`` 类进行扩展，以识别并管理这种“表面对”。

3.  **边界折射与衔接**:

      * **问题**: 光线进入 GRIN 介质瞬间的行为如何处理？
      * **建议**: ``GradientBoundarySurface`` 的 ``trace`` 方法应被重写。当光线到达该表面时，应执行一次标准的斯涅尔定律折射，计算光线进入介质后的初始位置与方向。该计算所用的折射率分别是 ``material_pre`` 的折射率和 ``GradientMaterial`` 在交点处的折射率 (即 ``n0``)。之后，将这个新的光线状态作为 ``ray_in`` 传递给 ``propagate_through_gradient`` 函数。这确保了职责的清晰分离。

4.  **算法实现细节**:

      * **步长控制**: RK4 算法的步长选择至关重要。固定步长易于实现，但效率与精度难以兼顾。
          * **短期方案**: 使用一个足够小的固定步长 (``step_size``)，并将其作为用户可配置参数。
          * **长期目标**: 实现自适应步长控制算法（如 Runge-Kutta-Fehlberg, RKF45），根据局部误差动态调整步长，以在保证精度的前提下提升计算效率。
      * **光程累积 (OPD)**: 光程是波前分析的基础。如 ``propagate_through_gradient`` 代码所示，应在 RK4 的每一步迭代中同步累积 ``∫n ds``。

5.  **性能与后端集成**:

      * **挑战**: GRIN 追迹的计算量远大于标准追迹。
      * **建议**:
          * **向量化**: ``GradientMaterial`` 中的 ``get_index_and_gradient`` 方法必须从设计之初就支持 NumPy 向量化操作，以便能同时处理多条光线。
          * **GPU 加速**: 考虑到 Optiland 对 PyTorch 的支持，应将 ``propagate_through_gradient`` 的核心循环（尤其是 RK4 迭代和导数计算）用 PyTorch 张量操作实现。这不仅能利用 GPU 加速，也为未来的自动微分优化铺平了道路。
          * **JIT 编译**: 对于 CPU 性能的极致追求，可考虑使用 Numba 对计算密集型函数进行即时编译。

6.  **扩展性考量**:

      * **色散**: 当前 ``GradientMaterial`` 的系数是常数。为支持色散，应将这些系数设计为可接受波长 ``wavelength`` 参数的函数或对象，与 Optiland 现有的材料模型保持一致。``get_index_and_gradient`` 方法也需增加 ``wavelength`` 参数。
      * **多项式形式**: 当前硬编码了一个多项式形式。未来可将其抽象为一个可配置的策略，允许用户定义不同的梯度折射率模型。

******************
4. 结论与展望
******************

此 GRIN 功能的设计方案在架构层面是卓越的，充分体现了软件工程的解耦原则。我们在此基础上提出的带有契约式设计和明确技术考量的实现方案，构成了一份可以直接指导开发工作的行动蓝图。

成功实现此功能，将使 Optiland 具备模拟复杂生物光学系统（如人眼）和设计先进光学元件的能力，极大拓展其应用范围和学术价值。后续开发工作的重点应放在解决上述“关键技术考量”中的具体问题，尤其是在**核心追迹逻辑的集成**、**RK4 算法的性能优化（向量化与 GPU 加速）以及色散支持**等方面。

我们坚信，通过严谨地执行这一经过充分审核的设计方案，Optiland 将朝着成为一个功能更强大、更专业的顶级开源光学仿真工具迈出坚实的一步。


"""

-----

请开始执行。
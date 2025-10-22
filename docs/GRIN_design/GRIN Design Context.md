好的，作为一名资深的软件架构师，我将为您准备一份精准的开发上下文。这份文档将严格遵循您的要求，为AI编程助手提供完成本次GRIN功能集成与架构重构任务所需的全部信息，且仅包含这些信息。

---

## 1. 任务目标概述 (Task Objective)
核心任务是重构Optiland的光线追踪引擎，引入一个“传播模型”抽象层。在此基础上，将现有的直线传播逻辑封装成`HomogeneousPropagation`模型，并实现一个新的`GrinPropagation`模型来支持梯度折射率(GRIN)介质。

## 2. 受影响的核心模块与文件 (Impact Analysis)
- `optiland/optic.py` (**修改**): 这是任务的核心。需要修改`Optic.trace`方法，将原有的硬编码直线传播逻辑替换为根据介质类型动态选择`PropagationModel`的调度机制。
- `optiland/propagation/base.py` (**新增**): 定义新的`PropagationModel`抽象基类接口，这是新架构的基础。
- `optiland/propagation/homogeneous.py` (**新增**): 实现`PropagationModel`接口，封装从`optic.py`中提取出的、原有的标准直线传播逻辑。
- `optiland/propagation/gradient.py` (**新增**): 实现`PropagationModel`接口，用于处理光线在GRIN介质中的曲线传播。
- `optiland/materials/gradient_material.py` (**新增**): 定义`GradientMaterial`类，封装GRIN介质的物理属性和折射率计算。
- `optiland/surfaces/gradient_surface.py` (**新增**): 定义`GradientBoundarySurface`类，作为进入GRIN介质的标记表面。
- `optiland/rays.py` (**理解/调用**): `RealRays`对象是整个追踪过程的核心数据结构，所有传播模型都将对其进行操作。
- `optiland/surfaces/base.py` (**理解/调用**): `BaseSurface`是所有表面的基类，其接口（如`material_post`属性）是连接交互和传播的关键。
- `optiland/materials/base.py` (**理解/调用**): `BaseMaterial`是所有材料的基类，`GradientMaterial`将继承自它。
- `tests/test_optic.py` (**参考**): 理解项目现有的测试风格和集成测试的构建方式。

## 3. 精准上下文详情 (Detailed Context)

### 模块/文件 A (修改): `optiland/optic.py`
- **与任务的关联**: 此文件包含核心的`Optic.trace`方法，即光线追踪主循环。当前，它将光线-表面交互和介质中传播的逻辑耦合在一起。**我们的目标是解耦这个循环**：保留表面交互部分，并将传播部分委托给新的`PropagationModel`。
- **相关代码片段 (Existing Code)**:
  
  ```python
  # path: optiland/optic.py

  from typing import Sequence
  import logging
  
  from tqdm.auto import tqdm
  
  from . import rays
  from . import surfaces
  
  
  class Optic:
      """An optical system consisting of a sequence of surfaces."""
  
      def __init__(
          self,
          surfaces: Sequence[surfaces.BaseSurface],
      ) -> None:
          """Create an Optic.
  
          Args:
              surfaces: a sequence of surfaces
          """
          self.surfaces = surfaces
  
      def trace(self, rays: rays.RealRays, report_progress: bool = False) -> rays.RealRays:
          """Trace a batch of rays through the sequence of surfaces.
  
          Args:
              rays: the rays to be traced
              report_progress: set to True to display a progress bar
  
          Returns:
              a new RealRays object with the traced rays
          """
          rays = rays.copy()
  
          # ... [代码注释：此处省略了一些无关的日志和进度条设置] ...
          
          # ------------------- 这是需要重构的核心循环 -------------------
          for i, surface in enumerate(iterable_surfaces):
              # propagate rays from previous surface to this one
              if i > 0:
                  # THIS IS THE PART TO BE REPLACED
                  # It assumes straight-line propagation
                  distance = surface.geometry.intersect(rays)
                  rays.x += distance * rays.L
                  rays.y += distance * rays.M
                  rays.z += distance * rays.N
                  n = self.surfaces[i-1].material_post.n(rays.w)
                  rays.opd += n * distance
  
              # apply surface
              rays = surface.trace(rays)
          # ------------------- 核心循环结束 -------------------
  
          return rays
  ```
- **交互与依赖**: `Optic.trace`方法直接依赖`surfaces.BaseSurface`的`geometry.intersect()`和`trace()`方法，以及`material_post.n()`方法来计算光程。重构后，它将不再直接调用`intersect`，而是调用`PropagationModel.propagate`。

### 模块/文件 B (理解/调用): `optiland/rays.py`
- **与任务的关联**: `RealRays`是贯穿整个系统的核心数据结构。所有`PropagationModel`的输入和输出都是`RealRays`对象。理解其结构至关重要。
- **相关代码片段 (Existing Code)**:

  ```python
  # path: optiland/rays.py
  
  from typing import Optional
  import dataclasses
  
  from . import backend as opt_backend
  
  @dataclasses.dataclass
  class RealRays:
      """A batch of real rays."""
      # position
      x: opt_backend.Vector
      y: opt_backend.Vector
      z: opt_backend.Vector
      # direction cosines
      L: opt_backend.Vector
      M: opt_backend.Vector
      N: opt_backend.Vector
      # optical path difference
      opd: opt_backend.Vector
      # wavelength
      w: opt_backend.Vector
      is_alive: opt_backend.Vector
      ref_index: opt_backend.Vector
      # ... [其他辅助属性和方法] ...
  
      def copy(self) -> "RealRays":
          """Return a copy of this object."""
          # ...
  ```

### 模块/文件 C (理解/调用): `optiland/surfaces/base.py`
- **与任务的关联**: `BaseSurface`定义了光学表面的通用接口。`PropagationModel`将使用`surface_in`和`surface_out`作为其边界，并通过`surface_in.material_post`来确定介质类型。
- **相关代码片段 (Existing Code)**:

  ```python
  # path: optiland/surfaces/base.py
  
  from abc import ABC, abstractmethod
  
  from .. import materials
  from .. import geometry
  from .. import rays
  
  class BaseSurface(ABC):
      """An optical surface."""
  
      def __init__(
          self,
          geometry: geometry.BaseGeometry,
          material_pre: materials.BaseMaterial,
          material_post: materials.BaseMaterial,
          # ...
      ):
          self.geometry = geometry
          self.material_pre = material_pre
          self.material_post = material_post
          # ...
  
      @abstractmethod
      def trace(self, rays: rays.RealRays) -> rays.RealRays:
          """Trace rays through this surface.
          
          This method is responsible for applying Snell's law or the law of reflection.
          """
          raise NotImplementedError()
  ```

## 4. 实现建议 (Implementation Guidance)
1.  **创建传播模型抽象层**:
    *   在`optiland/propagation/base.py`中，创建`PropagationModel`抽象基类，并定义`propagate(self, rays_in, surface_in, surface_out)`抽象方法。

2.  **封装现有逻辑**:
    *   在`optiland/propagation/homogeneous.py`中，创建`HomogeneousPropagation`类。
    *   将其`propagate`方法的实现逻辑，直接从`optiland/optic.py`的`for`循环中**剪切**过来（即计算`distance`并更新`x, y, z, opd`的部分）。

3.  **实现GRIN传播模型**:
    *   在`optiland/propagation/gradient.py`中，创建`GrinPropagation`类。
    *   根据设计文档，将其`propagate`方法实现为对RK4数值积分求解器的调用。

4.  **重构核心追踪引擎 (`Optic.trace`)**:
    *   修改`optiland/optic.py`中的`trace`方法循环。
    *   在 `if i > 0:` 代码块内，删除旧的直线传播代码。
    *   **新增选择逻辑**:
        ```python
        medium = self.surfaces[i-1].material_post
        if isinstance(medium, GradientMaterial): # 需要从 gradient_material 导入
            # 理想情况下，模型实例可以被缓存或预先创建
            propagation_model = GrinPropagation() 
        else:
            propagation_model = HomogeneousPropagation()
        
        # 调用模型进行传播
        rays = propagation_model.propagate(rays, self.surfaces[i-1], surface)
        ```
    *   在`for`循环之后，保留`rays = surface.trace(rays)`调用，这部分负责处理表面交互，职责不变。

5.  **创建GRIN相关模块**:
    *   根据设计文档，创建`optiland/materials/gradient_material.py`和`optiland/surfaces/gradient_surface.py`文件，并添加相应的类定义。

## 5. 测试与集成上下文 (Testing & Integration Context)
- **测试模式**: 项目使用 `pytest`。测试文件位于 `tests/` 目录下，通常与被测试的模块文件名对应，例如 `optic.py` 的测试是 `tests/test_optic.py`。测试风格偏向于构建小型、完整的`Optic`系统并验证端到端的追踪结果。

- **相关测试示例**: (摘自 `tests/test_optic.py`)
  
  ```python
  # path: tests/test_optic.py
  
  from optiland import Optic, rays, surfaces, geometry, materials
  from optiland.backend import np
  
  def test_trace_through_air_and_glass_surface_at_origin():
      # Setup: a simple system with one surface
      optic = Optic([
          surfaces.StandardSurface(
              geometry=geometry.SphericalGeometry(10.0),
              material_pre=materials.VACUUM,
              material_post=materials.Material(1.5),
          ),
      ])
      # Input: on-axis rays
      input_rays = rays.RealRays(
          x=np.array([0.0, 0.0]),
          y=np.array([1.0, -1.0]),
          z=np.array([-1.0, -1.0]),
          L=np.zeros(2),
          M=np.zeros(2),
          N=np.ones(2),
          # ... other ray properties
      )
      
      # Action: trace the rays
      output_rays = optic.trace(input_rays)
      
      # Assert: check the final state of the rays
      np.testing.assert_allclose(output_rays.x, 0.0)
      np.testing.assert_allclose(output_rays.y, [0.994433, -0.994433], atol=1e-5)
      # ... more assertions
  ```
  **AI指导**: 新的集成测试应遵循此模式，构建一个包含`GradientBoundarySurface`和`GradientMaterial`的`Optic`对象，然后验证最终光线的位置和方向是否符合预期。

- **用户API示例**:
  
  ```python
  # 1. 创建一个光学系统 (Optic)
  my_lens_system = Optic(surfaces=[surface1, surface2, surface3])
  
  # 2. 创建一批光线 (RealRays)
  initial_rays = rays.collimated_rays( ... )
  
  # 3. 执行光线追踪
  final_rays = my_lens_system.trace(initial_rays)
  ```
  **AI指导**: 新功能不应改变此高级API。用户的体验应该是无缝的，只需将`GradientMaterial`和`GradientBoundarySurface`作为普通材料和表面使用即可，系统内部应自动处理GRIN传播。
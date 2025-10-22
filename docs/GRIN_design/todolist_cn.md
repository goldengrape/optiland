### **GRIN 功能集成与架构重构开发计划**

**目标:** 将 GRIN 功能完全集成到 Optiland 核心追踪引擎中，同时完成向“传播模型”抽象架构的迁移，确保系统的模块化、可扩展性和可维护性。

---

#### **Phase 1: 奠定基础 —— 传播模型抽象层**

此阶段的目标是建立新的传播模型接口，并将现有的直线传播逻辑重构为第一个具体的实现。这为后续集成 GRIN 功能铺平了道路，且对现有系统的影响是可控的。

*   **[x] 1.1: 创建传播模型抽象基类**
    *   **文件:** `optiland/propagation/base.py`
    *   **任务:** 定义 `PropagationModel` 抽象基类 (ABC)，包含一个抽象方法 `propagate(self, rays_in: RealRays, surface_in: BaseSurface, surface_out: BaseSurface) -> RealRays`。

*   **[x] 1.2: 实现均质介质传播模型**
    *   **文件:** `optiland/propagation/homogeneous.py`
    *   **任务:**
        1.  创建 `HomogeneousPropagation(PropagationModel)` 类。
        2.  实现 `propagate` 方法。其逻辑应复现 Optiland 当前的默认行为：计算 `rays_in` 与 `surface_out` 的几何交点，并根据 `surface_in.material_post` 的折射率 `n` 和几何距离更新光程 `opd`。
        3.  这部分逻辑可能需要从现有的 `Optic.trace` 或 `Surface.trace` 方法中提取。

---

#### **Phase 2: 集成 GRIN —— 封装现有算法**

此阶段的核心是将你已经完成并验证过的 GRIN 传播代码，封装到新的传播模型类中。

*   **[ ] 2.1: 迁移 GRIN 模块文件**
    *   **任务:** 确认以下文件已放置在正确的模块路径下，并根据最终的 API 调整 `__init__.py` 文件以暴露接口。
        *   `optiland/surfaces/gradient_surface.py` (包含 `GradientBoundarySurface`)
        *   `optiland/materials/gradient_material.py` (包含 `GradientMaterial`)

*   **[ ] 2.2: 创建 GRIN 传播模型**
    *   **文件:** `optiland/propagation/gradient.py` (或将 `gradient_propagation.py` 重命名并移至此路径)
    *   **任务:**
        1.  创建 `GrinPropagation(PropagationModel)` 类。
        2.  将你已实现的 `propagate_through_gradient` 函数的主体逻辑迁移到 `GrinPropagation` 类的 `propagate` 方法中。
        3.  **适配接口:**
            *   `propagate` 方法的签名为 `(self, rays_in, surface_in, surface_out)`。
            *   在方法内部，通过 `grin_material = surface_in.material_post` 获取 `GradientMaterial` 实例。
            *   `exit_surface` 参数由 `surface_out` 替代。
        4.  此类可以接受 `step_size` 和 `max_steps` 作为 `__init__` 的参数，以方便配置。

---

#### **Phase 3: 核心引擎重构 —— "心脏移植手术"**

这是最关键的一步，我们将修改 Optiland 的核心光线追踪循环，使其能够识别并调度不同的传播模型。

*   **[ ] 3.1: 定位并分析核心追踪循环**
    *   **文件:** `optiland/optic.py` (最有可能) 或相关的 `SurfaceGroup` 类。
    *   **任务:** 找到遍历表面序列并调用 `surface.trace()` 的主循环。

*   **[ ] 3.2: 实现传播模型选择器**
    *   **任务:** 在主循环内部，对于当前表面 `S_i` 和下一个表面 `S_{i+1}`，实现一个选择机制。
    *   **逻辑:**
        1.  获取两个表面之间的介质: `medium = S_i.material_post`。
        2.  根据 `medium` 的类型选择传播模型：
            ```python
            if isinstance(medium, GradientMaterial):
                propagation_model = GrinPropagation() # 或一个预先实例化的对象
            else:
                propagation_model = HomogeneousPropagation()
            ```

*   **[ ] 3.3: 修改追踪流程**
    *   **任务:** 将原有的直线传播逻辑替换为对传播模型的调用。
    *   **新流程:**
        1.  **交互 (Interaction):** 调用 `rays = S_i.trace(rays)`。此步骤处理光线与当前表面的折射/反射。`GradientBoundarySurface` 会继承此方法，并利用 `GradientMaterial.n()` 返回的 `n0` 值正确处理边界折射。
        2.  **传播 (Propagation):**
            *   获取下一个表面 `S_{i+1}`。
            *   选择传播模型 `model = select_model(S_i.material_post)`。
            *   调用 `rays = model.propagate(rays, S_i, S_{i+1})`。
        3.  循环继续，此时 `rays` 已位于 `S_{i+1}` 的表面上，准备进行下一次交互。

---

#### **Phase 4: 验证与测试**

确保重构后的系统不仅功能正确，而且性能可靠。`GRIN_design_reference_context.md` 中提到的测试策略非常全面，可以按此执行。

*   **[ ] 4.1: 单元测试**
    *   **`GradientMaterial`:** 验证 `get_index_and_gradient` 方法对于已知坐标和系数的计算是否精确。
    *   **`HomogeneousPropagation`:** 验证其行为与重构前的直线传播结果完全一致。
    *   **`GrinPropagation`:** 针对已知的、存在解析解的简单 GRIN 模型（如线性梯度），验证 RK4 算法的路径和 OPD 累积的准确性。

*   **[ ] 4.2: 集成测试**
    *   构建一个包含 `GradientBoundarySurface` 和 `GradientMaterial` 的完整 `Optic` 对象（例如，一个 GRIN 棒镜）。
    *   执行端到端的光线追踪，检查最终的光线坐标、方向和 OPD 是否符合预期。
    *   如果条件允许，与商业软件 (Zemax, Code V) 的结果进行交叉验证。

*   **[ ] 4.3: 性能基准测试**
    *   对比 GRIN 追踪与标准追踪的耗时，建立性能基准。
    *   使用 Optiland 的后端切换功能，分别在 NumPy (CPU) 和 PyTorch (GPU) 后端下测试 GRIN 追踪的性能，验证向量化实现的有效性。

---

#### **Phase 5: 文档与示例**

让新功能易于被其他开发者和用户理解与使用。

*   **[ ] 5.1: 更新代码文档 (Docstrings)**
    *   为所有新创建的类和方法 (`PropagationModel`, `GrinPropagation`等) 撰写清晰、完整的文档字符串。

*   **[ ] 5.2: 创建教学示例**
    *   **任务:** 编写一个新的 Jupyter Notebook 教程，类似于 `Tutorial_10a_Custom_Surface_Types.html`。
    *   **内容:** 演示如何：
        1.  实例化一个 `GradientMaterial`。
        2.  使用 `GradientBoundarySurface` 构建一个 GRIN 透镜。
        3.  将此透镜放入一个 `Optic` 系统中。
        4.  执行光线追踪并可视化结果（如光线轨迹图）。

*   **[ ] 5.3: 更新项目文档**
    *   在 Optiland 的官方文档中，增加关于 GRIN 功能和传播模型架构的说明。
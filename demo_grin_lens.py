"""
Optiland GRIN 功能演示程序

本示例程序旨在演示如何在 Optiland 库中构建、追迹并可视化一个
包含梯度折射率（GRIN）介质的光学系统。

核心演示内容：
1.  实例化一个抛物线折射率分布的 GRIN 材料 (GradientMaterial)。
    这种材料也被称为 Wood Lens，具有已知的聚焦特性。
2.  使用一对平面的 GradientBoundarySurface 来定义一个 GRIN 棒的
    物理边界（入口和出口）。
3.  构建一个完整的 Optic 对象，该对象包含 GRIN 棒以及前后的空气间隔。
4.  设置系统参数，如入瞳直径、视场和波长。
5.  调用 draw() 方法来追迹一组均匀分布的准直光线，并使用
    matplotlib 可视化其在 GRIN 介质内部弯曲并最终聚焦的轨迹。

此示例的设计与实现严格遵循了 Optiland 的 GRIN 功能设计文档，
体现了公理设计中几何、物理与行为域的分离原则。
"""
import numpy as np
import matplotlib.pyplot as plt

# 假设 optiland 已安装在当前 Python 环境中
# 导入 Optiland 核心类以及本次新增的 GRIN 相关模块
try:
    from optiland import Optic
    from optiland.surfaces.gradient_surface import GradientBoundarySurface
    from optiland.materials.gradient_material import GradientMaterial
    from optiland.materials import IdealMaterial
except ImportError:
    print("错误：无法导入 Opticland 库。请确保该库已正确安装。")
    exit()


def create_focusing_grin_rod_system():
    """
    构建一个包含聚焦 GRIN 棒的光学系统。

    该系统由四个表面组成：
    1. 物面 (在无穷远)
    2. GRIN 棒入口 (GradientBoundarySurface)
    3. GRIN 棒出口 (GradientBoundarySurface)
    4. 像面

    Returns:
        Optic: 一个配置完成的 Optiland Optic 对象。
    """
    print("开始构建聚焦 GRIN 棒光学系统...")

    # 1. 定义 GRIN 介质的物理属性
    # 采用抛物线分布 n(r) = n0 + nr2 * r^2
    # 为了实现聚焦，nr2 必须为负值。
    n0 = 1.5          # 轴上折射率
    nr2 = -0.005      # 二次梯度系数
    rod_length = 50.0   # GRIN 棒的长度 (mm)

    grin_material = GradientMaterial(n0=n0, nr2=nr2)
    print(f"  - GRIN 材料已定义: n0={n0}, nr2={nr2}")

    # 2. 实例化一个 Optic 对象
    grin_lens_system = Optic()
    print("  - Optic 对象已创建。")

    # 3. 按顺序添加光学表面
    # 表面 0: 物面，位于无穷远
    grin_lens_system.add_surface(thickness=np.inf)

    # 表面 1: GRIN 棒入口。
    # 这是一个平坦的 GradientBoundarySurface，其厚度定义了 GRIN 介质的长度。
    # material_post 属性被设置为我们定义的 GradientMaterial 对象。
    # 仿照 Tutorial_10a_Custom_Surface_Types.ipynb 的做法，
    # 我们预先构建完整的 Surface 对象，以获得最大的灵活性和清晰度。
    entry_surface = GradientBoundarySurface(
        radius_of_curvature=np.inf,
        thickness=rod_length,
        material_post=grin_material,
        is_stop=True,  # 将此表面设为光阑面
        semi_diameter=10.0  # 定义一个 20mm 的物理口径
    )
    grin_lens_system.add_surface(new_surface=entry_surface)
    print(f"  - 表面 1 (GRIN 入口) 已添加，棒长: {rod_length} mm。")

    # 表面 2: GRIN 棒出口。
    # 这同样是一个平坦的 GradientBoundarySurface，用于标记 GRIN 区域的结束。
    # 其厚度定义了从出口到下一个表面的距离。
    exit_surface = GradientBoundarySurface(
        radius_of_curvature=np.inf,
        thickness=100.0,  # 到像面的距离 (mm)
        semi_diameter=10.0
    )
    grin_lens_system.add_surface(new_surface=exit_surface)
    print(f"  - 表面 2 (GRIN 出口) 已添加，追迹距离: 100.0 mm。")
    
    # 表面 3: 像面
    grin_lens_system.add_surface()
    print("  - 表面 3 (像面) 已添加。")

    # 4. 定义系统级别的参数
    # 定义入瞳直径 (Entrance Pupil Diameter)
    grin_lens_system.set_aperture(aperture_type="EPD", value=10)
    print("  - 系统光圈已设置为 10mm EPD。")

    # 定义一个在轴上的视场 (角度为 0)
    grin_lens_system.set_field_type(field_type="angle")
    grin_lens_system.add_field(y=0)
    print("  - 系统视场已设置为轴上视场 (0 度)。")

    # 定义工作波长 (0.55 微米)
    grin_lens_system.add_wavelength(value=0.55, is_primary=True)
    print("  - 系统波长已设置为 0.55 um。")
    
    print("\n系统构建完成。")
    return grin_lens_system


def main():
    """
    主函数：创建系统并进行可视化。
    """
    # 创建 GRIN 透镜系统
    grin_system = create_focusing_grin_rod_system()

    # 追迹光线并绘制 2D 光路图
    # 我们追迹 9 条均匀分布的准直光线
    print("\n开始光线追迹与可视化...")
    fig, ax = grin_system.draw(num_rays=9, distribution="uniform")
    
    # 美化图表
    ax.set_title("聚焦型 GRIN 棒 (Wood Lens) 光线追迹演示", fontsize=16)
    ax.set_xlabel("Z 轴坐标 (mm)", fontsize=12)
    ax.set_ylabel("Y 轴坐标 (mm)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    print("光路图已生成。请查看弹出的 Matplotlib 窗口。")
    plt.show()


if __name__ == "__main__":
    main()
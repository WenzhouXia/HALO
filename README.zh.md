# HALO_public

[English](README.md) | [简体中文](README.zh.md)

`HALO_public` 是 ICLR 2026 论文 **A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems** 的官方实现，
提供了一种 O(n) 内存、GPU 友好的大规模 OT 求解器，尤其适合低维场景，如 2D grid 和 3D 点云。

## 核心思想

求解器通过层次结构中的粗层（coarse level）为原 OT 问题提供良好的 warm-start，然后逐层迭代细化求解，从而高效获得最终 OT 解。

## 模式说明

- `grid` 模式：适合 2D 笛卡尔网格，典型应用是规则 2D 网格上的图像直方图。  
- `tree` 模式：适合低维点云，尤其是 2D 和 3D 点云。提供的 free-support Wasserstein barycenter 示例也使用此模式。  
- 仓库提供的代码适配平方欧氏距离的 OT 问题。

## 示例

运行：

```bash
bash examples/run_examples.sh
```

即可自动运行四个基础示例，包括 2D/3D grid 和 tree 模式的 barycenter 实验。

# HALO_public

[English](README.md) | [简体中文](README.zh.md)

`HALO_public` 是从 `hierarchical_ot` 重建得到的公开 tree/grid 子集。

它提供了层次化最优传输的可安装运行时，以及一组可以直接运行的公开示例。

## 核心求解流程

核心的多层级算法位于 `src/hierarchical_ot/core/multilevel_flow.py`。

从高层看，求解器执行如下流程：

1. 校验问题定义并初始化运行级状态。
2. 构建从粗到细的层次结构 level 列表。
3. 对每个 level 初始化对应 mode 的状态和当前 active support。
4. 在该 level 内反复执行：准备当前子问题、求解一次 LP、完成该次迭代的后处理，并检查停止条件。
5. 记录当前 level 的结果，并把 warm start 信息传给更细一层。
6. 最细层结束后，打包最终的 transport 结果和 profiling 输出。

这也是整个库的核心思想：按从粗到细的一系列 OT 子问题逐层求解，并用粗层结果去 warm-start 细层。

## 模式说明

`grid` mode 对应 2D 笛卡尔网格。
典型场景是规则 2D 网格上的图像类直方图。

`tree` mode 适合低维点云，尤其是 2D 和 3D 点云。
我们提供的 free-support Wasserstein barycenter 示例也使用这一模式。

## 示例

运行：

```bash
bash examples/run_examples.sh
```

这个脚本会直接运行公开仓库中的示例：

- `examples/test_grid_mode.py`：基础 2D grid-mode 示例
- `examples/test_tree_mode.py`：基础 tree-mode 示例
- `examples/show_tree_pairwise_barycenter.py --dimension 2`：2D free-support Wasserstein barycenter
- `examples/show_tree_pairwise_barycenter.py --dimension 3`：3D free-support Wasserstein barycenter

这些 barycenter 脚本可以看作构建在导出 OT 求解器之上的下游应用示例。

## 导出策略

- 保留可安装的 tree/grid 运行时和 API
- 排除 cluster mode、dual-assignment、gromov 和 low-rank 研究路径
- 用公开版本 overlay 覆盖共享入口，确保包可以直接导入
- 导出位于 `examples/` 下的可运行公开示例

导出脚本还会写出：

- `EXPORT_MANIFEST.json`：复制文件、overlay 和排除策略
- `EXPORT_AUDIT.md`：禁用路径 / 禁用文本的审计结果

## 示例附加依赖

- barycenter 可视化示例需要 `matplotlib` 和 `POT`
- 3D barycenter 示例使用了 `examples/assets/` 下预采样的 ModelNet10 chair/toilet 资产

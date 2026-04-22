# hierarchical_ot/hierarchy/tree.py
"""
Tree-based Hierarchy (HALO 迁移)
基于 2^n-Tree 或 KD-Tree 的空间划分层次结构。
支持两种构建模式：
1. recursive: 递归中点划分
2. bucket: 网格量化 + 点重排 (HALO Ultra-Fast V5)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from numba import njit, prange
import time
from scipy.spatial import cKDTree

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from ...types.base import BaseHierarchy, HierarchyLevel


def suggest_num_levels(n_points: int, dim: int, split_mode: str) -> int:
    """
    CN: 根据点数和数据维度，在给定划分模式下（2^n 或 k-d tree），估计一个“足够深”的分层层数 K
    EN: Estimate a sufficiently deep number of hierarchy levels K based on the number of points and data dimensionality under a given split mode (2^n or k-d tree)
    """
    if split_mode == "2^n":
        branch = 2 ** dim
    else:  # "kdtree"
        branch = 2
    # n ≈ branch^K ⇒ K ≈ log(n) / log(branch)
    K = max(1, int(np.ceil(np.log(max(n_points, 1)) / np.log(branch))))
    return K


@dataclass
class _Node:
    """
    CN: 内部树节点（仅用于构建阶段）
    EN: Internal tree node (used only during the construction phase)
    """
    level: int
    bbox: np.ndarray  # (2, D) [min, max]
    parent_node_index: int
    global_node_index: int
    public_idx: int
    center: Optional[np.ndarray] = None
    radius: float = 0.0
    point_indices: Optional[np.ndarray] = None
    mass: float = 0.0
    is_leaf: bool = False
    children: List[Optional['_Node']] = field(default_factory=list)



@njit(cache=True)
def _bucket_split_numba(
    fine_idx,        # shape = (N_all, dim), int32
    point_indices,   # shape = (N,), int32 或 int64
    node_starts,     # shape = (M,), int64
    node_ends,       # shape = (M,), int64
    dim: int,
    bitpos: int
):
    """
    CN: 按网格位拆分点，返回重排后的点索引
    EN: Split points according to grid bits and return the reordered point indices
    """
    M = node_starts.shape[0]
    N = point_indices.shape[0]
    C = 1 << dim  # 每维分裂产生 2^dim 个子节点

    point_indices_next = np.empty(N, dtype=point_indices.dtype)
    child_counts = np.zeros((M, C), dtype=np.int64)
    node_offsets = np.empty(M, dtype=np.int64)

    # Step 0: node_offsets 前缀和 (必须串行)
    offset = 0
    for i in range(M):
        start = node_starts[i]
        end = node_ends[i]
        node_offsets[i] = offset
        offset += (end - start)

    # Pass 1: Count (可并行)
    for i in prange(M):
        start = node_starts[i]
        end = node_ends[i]
        if end <= start:
            continue

        for idx_pos in range(start, end):
            pidx = point_indices[idx_pos]
            c = 0
            for d in range(dim):
                bits_d = (fine_idx[pidx, d] >> bitpos) & 1
                shift = dim - 1 - d
                c |= (bits_d << shift)
            child_counts[i, c] += 1

    # Pass 2: Move (可并行)
    for i in prange(M):
        start = node_starts[i]
        end = node_ends[i]
        if end <= start:
            continue

        node_offset = node_offsets[i]

        # 为该父节点准备局部起始位置
        child_starts_local = np.empty(C, dtype=np.int64)
        s = node_offset
        for c in range(C):
            child_starts_local[c] = s
            s += child_counts[i, c]

        child_pos = child_starts_local.copy()

        for idx_pos in range(start, end):
            pidx = point_indices[idx_pos]
            c = 0
            for d in range(dim):
                bits_d = (fine_idx[pidx, d] >> bitpos) & 1
                shift = dim - 1 - d
                c |= (bits_d << shift)

            dst = child_pos[c]
            point_indices_next[dst] = pidx
            child_pos[c] += 1

    return point_indices_next, child_counts, node_offsets


@njit(cache=True)
def _flatten_specs_kernel(level, M, C, dim, node_offsets, child_counts, child_bboxes_all):
    """
    CN: 高效地将 (M, C) 的拓扑结构展平为下一层的线性节点列表。
    EN: Efficiently flatten the (M, C) topology into a linear node list for the next level.
    """
    # 1. 计算下一层总节点数
    total = 0
    for i in range(M):
        for c in range(C):
            if child_counts[i, c] > 0:
                total += 1

    # 2. 分配结果数组
    # meta format: [level, parent_rel_idx, public_idx, start, end]
    new_meta = np.empty((total, 5), dtype=np.int64)
    new_bbox = np.empty((total, 2, dim), dtype=np.float64)

    idx = 0
    for i in range(M):
        s = node_offsets[i]

        for c in range(C):
            cnt = child_counts[i, c]
            if cnt > 0:
                e = s + cnt

                # 填充元数据
                new_meta[idx, 0] = level - 1
                new_meta[idx, 1] = i
                new_meta[idx, 2] = idx
                new_meta[idx, 3] = s
                new_meta[idx, 4] = e

                # 填充几何信息
                new_bbox[idx] = child_bboxes_all[i, c]

                s = e
                idx += 1

    return new_meta, new_bbox


@njit(cache=True)
def _fill_l1_info_kernel(l0_map, l1_masses, l1_meta, mass_cumsum):
    """
    CN: 填充 L1 信息到 l0_map 和 l1_masses
    EN: Populate L1 information into l0_map and l1_masses
    """
    n = len(l1_meta)
    for i in range(n):
        s = l1_meta[i, 3]
        e = l1_meta[i, 4]
        l0_map[s:e] = i  # i 就是 L1 的 public_idx
        l1_masses[i] = mass_cumsum[e] - mass_cumsum[s]


@njit(cache=True)
def _fill_csr_and_mass_kernel(
    ptr,
    indices,
    mass_dst,
    mass_src,
    node_start_idx,
    child_layer_start_idx,
    num_children,
    running_idx,
):
    """
    CN: 同时填充 CSR 结构，并聚合父层节点的质量信息。
    EN: Fill the CSR structure and aggregate the mass of parent-layer nodes at the same time.
    """
    n_nodes = len(num_children)
    current_child_node = child_layer_start_idx
    ptr[node_start_idx] = running_idx

    for i in range(n_nodes):
        count = num_children[i]
        mass_sum = 0.0

        for k in range(count):
            child_idx = current_child_node
            indices[running_idx + k] = child_idx
            mass_sum += mass_src[child_idx]
            current_child_node += 1

        mass_dst[node_start_idx + i] = mass_sum
        running_idx += count
        ptr[node_start_idx + i + 1] = running_idx

    return running_idx


def _child_bbox_patterns(dim: int):
    """
    CN: 生成所有 2^dim 种子节点边界框的最小值/最大值偏移模式。
    EN: Generate the min/max offset patterns for all 2^dim child bounding boxes.
    """
    pattern_min = np.zeros((1 << dim, dim), dtype=np.float64)
    pattern_max = np.ones((1 << dim, dim), dtype=np.float64)

    for pattern in range(1 << dim):
        for d in range(dim):
            if pattern & (1 << (dim - 1 - d)):
                pattern_min[pattern, d] = 0.5
            else:
                pattern_max[pattern, d] = 0.5

    return pattern_min, pattern_max


@njit(cache=True)
def _compute_child_bboxes_all(
    level_min, level_max, level_center,
    pattern_min, pattern_max,
    child_bboxes_all,
):
    """
    CN: 批量计算所有可能的子节点包围盒。
    EN: Batch compute all possible child bounding boxes.
    """
    M = level_min.shape[0]
    C = pattern_min.shape[0]
    dim = level_min.shape[1]

    for i in range(M):
        for c in range(C):
            # child_min = level_min + (level_max - level_min) * pattern_min
            # child_max = level_min + (level_max - level_min) * pattern_max
            span = level_max[i] - level_min[i]
            child_bboxes_all[i, c, 0] = level_min[i] + span * pattern_min[c]
            child_bboxes_all[i, c, 1] = level_min[i] + span * pattern_max[c]


def _bucket_split_gpu(
    fine_idx,
    point_indices,
    node_starts,
    node_ends,
    dim: int,
    bitpos: int,
):
    """
    CN: GPU 版本 bucket split。
    EN: GPU version of bucket split.
    使用 CuPy 计算每个点的 (node_id, child_id)，按 key 排序得到重排后的索引。
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available for GPU bucket split")

    # 准备输入到 GPU
    if isinstance(fine_idx, np.ndarray):
        fine_idx_gpu = cp.asarray(fine_idx)
    else:
        fine_idx_gpu = fine_idx

    point_indices_cpu = point_indices.astype(np.int32)
    point_indices_gpu = cp.asarray(point_indices_cpu)

    node_starts_cpu = node_starts.astype(np.int64)
    node_ends_cpu = node_ends.astype(np.int64)

    M = int(node_starts_cpu.shape[0])
    N = int(point_indices_cpu.shape[0])
    C = 1 << dim

    if M == 0 or N == 0:
        point_indices_next = point_indices_cpu.copy()
        child_counts = np.zeros((M, C), dtype=np.int64)
        node_offsets = np.zeros(M, dtype=np.int64)
        return point_indices_next, child_counts, node_offsets

    # 计算每个节点的长度
    lens_cpu = (node_ends_cpu - node_starts_cpu).astype(np.int64)

    # 构造每个点对应的 node_id
    node_ids_cpu = np.repeat(np.arange(M, dtype=np.int32), lens_cpu)
    node_ids_gpu = cp.asarray(node_ids_cpu)

    # 子集 fine_idx[point_indices]
    fine_sub_gpu = fine_idx_gpu[point_indices_gpu]

    # 计算 child_id
    bits_gpu = (fine_sub_gpu >> bitpos) & 1
    shifts = cp.arange(dim - 1, -1, -1, dtype=np.int32)
    child_id_gpu = cp.sum(bits_gpu * (1 << shifts), axis=1)

    # 计算排序 key = node_id * C + child_id
    key_gpu = node_ids_gpu * C + child_id_gpu
    sort_idx_gpu = cp.argsort(key_gpu)

    # 重排 point_indices
    point_indices_next_gpu = point_indices_gpu[sort_idx_gpu]
    point_indices_next = cp.asnumpy(point_indices_next_gpu)

    # 计算 child_counts (CPU)
    child_counts = np.zeros((M, C), dtype=np.int64)
    sorted_node_ids = node_ids_cpu[cp.asnumpy(sort_idx_gpu)]
    sorted_child_ids = cp.asnumpy(child_id_gpu)[cp.asnumpy(sort_idx_gpu)]

    for i in range(N):
        node_id = sorted_node_ids[i]
        child_id = sorted_child_ids[i]
        child_counts[node_id, child_id] += 1

    # 计算 node_offsets (CPU)
    node_offsets = np.zeros(M, dtype=np.int64)
    offset = 0
    for i in range(M):
        node_offsets[i] = offset
        offset += lens_cpu[i]

    return point_indices_next, child_counts, node_offsets


class TreeHierarchy(BaseHierarchy):
    """
    CN: 基于空间划分树（2^n-Tree 或 KD-Tree）的层次结构。
    CN: 对应原 HALO 的 HypercubeHierarchy，支持 recursive 和 bucket 两种构建模式。
    EN: Hierarchy based on spatial partitioning tree (2^n-Tree or KD-Tree).
    EN: Corresponds to HALO's HypercubeHierarchy, supports recursive and bucket construction modes.
    """

    def __init__(
        self,
        num_levels: Optional[int] = None,
        split_mode: str = "2^n",  # '2^n' 或 'kdtree'
        build_mode: str = "bucket",  # 'recursive' 或 'bucket'
        target_coarse_size: int = 256,
        max_L0_L1_ratio: float = 4.0,
        k_neighbors: int = 10,
        use_gpu_bucket: bool = False,
        depth_min: Optional[int] = None,  # 最少层数
        depth_max: Optional[int] = None,  # 最多层数
        verbose: bool = False,
    ):
        """
        CN: 初始化 TreeHierarchy。
        EN: Initialize TreeHierarchy.
        """
        super().__init__([])
        self._build_root_level = num_levels or 3
        self._num_levels = num_levels
        self.split_mode = split_mode
        self.build_mode = build_mode  # 'recursive' 或 'bucket'
        self.target_coarse_size = target_coarse_size
        self.max_L0_L1_ratio = max_L0_L1_ratio
        self.k_neighbors = k_neighbors
        self.use_gpu_bucket = use_gpu_bucket
        self.depth_min = depth_min  # 最少层数
        self.depth_max = depth_max if depth_max is not None else 100  # 最多层数
        self.verbose = verbose

        # 内部状态
        self.dim: Optional[int] = None
        self._nodes_by_level: List[List[_Node]] = []
        self._global_node_index = 0
        self._l0_to_l1_map: Optional[np.ndarray] = None
        self.build_time: float = 0.0

        # Bucket 模式专用状态
        self._initial_points: Optional[np.ndarray] = None
        self._initial_masses: Optional[np.ndarray] = None
        self._fine_indices: Optional[np.ndarray] = None  # (N, dim) 网格索引
        self._fine_max_splits: int = 0
        self._permuted_point_indices: Optional[np.ndarray] = None
        self.layers_data: Dict = {}  # bucket 模式的层数据

    def _compute_node_geometry(self, bbox: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        CN: 计算节点的中心和半径。
        EN: Compute the center and radius of a node.
        """
        center = 0.5 * (bbox[0] + bbox[1])
        diff = bbox[1] - center
        radius = float(np.sqrt(np.dot(diff, diff)))
        return center, radius

    def _estimate_optimal_num_levels(
        self,
        points: np.ndarray,
        masses: np.ndarray,
        depth_min: int = 2,
        depth_max: int = 20,
    ) -> int:
        """
        CN: 自适应估计最优 num_levels。
        EN: Adaptively estimate the optimal number of levels.
        使用规则网格估算 L1 非空格子数，选择满足 max_L0_L1_ratio 的最小 K。
        """
        n0 = len(points)
        dim = self.dim

        # 计算包围盒
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        span = max_coords - min_coords
        span[span == 0.0] = 1.0

        def estimate_L1_for_K(K: int) -> int:
            """对给定的 num_levels=K，估算 L1 的非空格子数。"""
            if K <= 1:
                return 1

            num_splits = K - 1
            n_bins = 1 << num_splits

            # 量化坐标
            idx = np.floor((points - min_coords) * (n_bins / span)).astype(np.int64)
            np.clip(idx, 0, n_bins - 1, out=idx)

            # 坐标压缩
            bits_per_dim = num_splits
            if bits_per_dim * dim < 63:
                packed = np.zeros(idx.shape[0], dtype=np.int64)
                for d in range(dim):
                    if d > 0:
                        packed <<= bits_per_dim
                    packed |= idx[:, d]
                return len(np.unique(packed))
            else:
                return len(np.unique(idx, axis=0))

        best_K = None
        best_ratio = None

        for K in range(depth_min, depth_max + 1):
            n1_est = estimate_L1_for_K(K)
            ratio = n0 / max(n1_est, 1)

            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_K = K

            if ratio <= self.max_L0_L1_ratio:
                break

        if best_K is None:
            best_K = depth_max

        return best_K

    def build(
        self,
        initial_points: np.ndarray,
        initial_masses: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        CN: 构建树层次结构。
        EN: Build the tree hierarchy.

        Args:
            initial_points: (N, D) 点云坐标
            initial_masses: (N,) 质量分布，None 表示均匀分布
        """
        t0 = time.perf_counter()

        # 保存初始数据
        points = np.asarray(initial_points, dtype=np.float64)
        self.dim = points.shape[1]
        n_points = len(points)

        if initial_masses is None:
            masses = np.full(n_points, 1.0 / n_points, dtype=np.float64)
        else:
            masses = np.asarray(initial_masses, dtype=np.float64)
            masses /= masses.sum()

        # 自动确定层数（自适应计算）
        if self._num_levels is None:
            # [HALO 对齐] 使用构造函数传入的 depth_min/depth_max
            depth_min = self.depth_min if self.depth_min is not None else 2
            depth_max = self.depth_max if self.depth_max is not None else 100
            self._num_levels = self._estimate_optimal_num_levels(
                points, masses, depth_min, depth_max
            )
        self._build_root_level = self._num_levels

        # 初始化
        self._nodes_by_level = [[] for _ in range(self.num_levels + 1)]
        self._global_node_index = 0
        self._l0_to_l1_map = np.zeros(n_points, dtype=np.int32)

        # 计算根包围盒
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        box_tol = 1e-10
        root_bbox = np.stack([min_coords - box_tol, max_coords + box_tol], axis=0)

        # 根据 build_mode 选择构建方法
        build_mode = kwargs.get('build_mode', self.build_mode)

        if build_mode == "bucket" and self.split_mode == "2^n":
            # [HALO 对齐] 使用 bucket 模式
            self._build_bucket(points, masses, root_bbox)
        else:
            # 使用递归模式
            self._build_recursive(
                level=self.num_levels,
                parent_node_index=-1,
                point_indices=np.arange(n_points, dtype=np.int32),
                bbox=root_bbox,
                points=points,
                masses=masses,
            )

        # 创建公开层级
        self._create_public_levels(points, masses)

        # [HALO 对齐] 自动截断顶层：保留节点数 >= target_coarse_size 的层
        self._trim_to_target_size(target_coarse_size=self.target_coarse_size)

        # 计算 K-NN
        self._compute_knn()

        self.build_time = time.perf_counter() - t0
        if self.verbose:
            print(f"[TreeHierarchy] 构建完成: {self.num_levels} 层, 用时 {self.build_time:.3f}s")

        # 打印层次结构信息（与 HALO 一致）
        print("层次结构信息:")
        print(f"- 维度: {self.dim}")
        print(f"- 层数: {self.num_levels}")
        for level_idx in range(self.num_levels, -1, -1):
            if level_idx < len(self.levels):
                n_points = len(self.levels[level_idx].points)
                print(f"  - 层 {level_idx}:")
                print(f"    - 非空节点数: {n_points}")

    # =========================================================================
    # Bucket 模式实现 (与 HALO 完全对齐)
    # =========================================================================

    def _prepare_fine_grid_indices(self, root_bbox: np.ndarray):
        """
        CN: 预计算每个点在最细层 (L1) 的规则网格索引。
        EN: Pre-compute the regular grid index for each point at the finest level (L1).
        """
        K = self._build_root_level
        if K <= 1:
            self._fine_indices = None
            return

        pts = self._initial_points
        N, dim = pts.shape

        box_lo = root_bbox[0]
        box_hi = root_bbox[1]
        span = box_hi - box_lo
        span[span == 0.0] = 1.0

        max_splits = K - 1
        n_bins = 1 << max_splits

        # 归一化 + 量化到 [0, n_bins-1]
        idx = np.floor((pts - box_lo) * (n_bins / span)).astype(np.int64)
        np.clip(idx, 0, n_bins - 1, out=idx)

        self._fine_indices = idx.astype(np.int32)
        self._fine_max_splits = max_splits

    def _build_bucket(
        self,
        points: np.ndarray,
        masses: np.ndarray,
        root_bbox: np.ndarray
    ):
        """
        CN: Ultra-Fast V5 bucket 模式构建。
        EN: Ultra-Fast V5 bucket mode construction.
        """
        N = len(points)
        K = self._build_root_level
        dim = self.dim

        # 保存初始点/质量
        self._initial_points = points
        self._initial_masses = masses

        # 预处理网格索引
        self._prepare_fine_grid_indices(root_bbox)

        # 初始化层数据
        self.layers_data = {}

        # Root (L_K)
        root_meta = np.array([[K, -1, 0, 0, N]], dtype=np.int64)
        root_bbox_arr = root_bbox[np.newaxis, :, :].astype(np.float64)

        cur_indices = np.arange(N, dtype=np.int32)
        cur_meta = root_meta
        cur_bbox = root_bbox_arr

        self.layers_data[K] = {
            'meta': cur_meta,
            'bbox': cur_bbox,
            'counts': None
        }

        # Top-Down Loop (K -> 2)
        for level in range(K, 1, -1):
            cur_indices, next_meta, next_bbox, parent_child_counts = \
                self._build_next_level_bucket_fast_flat(
                    level, cur_indices, cur_meta, cur_bbox, dim
                )

            self.layers_data[level]['counts'] = parent_child_counts
            self.layers_data[level - 1] = {
                'meta': next_meta,
                'bbox': next_bbox,
                'counts': None
            }

            cur_meta = next_meta
            cur_bbox = next_bbox

        # L1 Mass 计算
        current_masses = self._initial_masses[cur_indices]
        mass_cumsum = np.zeros(len(current_masses) + 1, dtype=np.float64)
        np.cumsum(current_masses, out=mass_cumsum[1:])

        l1_meta = self.layers_data[1]['meta']
        n_l1 = len(l1_meta)
        l1_masses = np.empty(n_l1, dtype=np.float64)

        self._l0_to_l1_map[:] = -1
        _fill_l1_info_kernel(self._l0_to_l1_map, l1_masses, l1_meta, mass_cumsum)

        self.layers_data[1]['mass'] = l1_masses
        self._permuted_point_indices = cur_indices

        # 创建 flat 数组（bucket 模式需要）
        self._finalize_flat_arrays(N)

    def _build_next_level_bucket_fast_flat(
        self,
        level: int,
        point_indices: np.ndarray,
        specs_meta: np.ndarray,
        specs_bbox: np.ndarray,
        dim: int
    ):
        """
        CN: 构建下一层 (Level -> Level-1)。
        EN: Build the next level (Level -> Level-1).
        """
        M = len(specs_meta)
        N = len(point_indices)
        C = 1 << dim

        if N == 0 or M == 0:
            return point_indices.copy(), np.empty((0, 5), dtype=np.int64), \
                   np.empty((0, 2, dim), dtype=np.float64), np.zeros((0, 1 << dim), dtype=np.int64)

        # Bucket Split
        fine_idx = self._fine_indices
        max_splits = self._fine_max_splits

        # 计算当前层对应的 bit 位置
        bitpos = level - 2
        if bitpos < 0:
            bitpos = 0
        if bitpos >= max_splits:
            bitpos = max_splits - 1

        # 提取父节点的 start/end 范围
        node_starts = specs_meta[:, 3]
        node_ends = specs_meta[:, 4]

        # 执行 bucket split
        use_gpu = self.use_gpu_bucket
        if use_gpu:
            fine_idx_arg = self._fine_indices_gpu if hasattr(self, '_fine_indices_gpu') else fine_idx
            point_indices_next64, child_counts, node_offsets = _bucket_split_gpu(
                fine_idx=fine_idx_arg,
                point_indices=point_indices,
                node_starts=node_starts,
                node_ends=node_ends,
                dim=dim,
                bitpos=bitpos,
            )
            cp.cuda.Stream.null.synchronize()
        else:
            point_indices_next64, child_counts, node_offsets = _bucket_split_numba(
                fine_idx,
                point_indices.astype(np.int64),
                node_starts,
                node_ends,
                dim,
                bitpos,
            )
        point_indices_next = point_indices_next64.astype(np.int32)

        # 批量计算所有可能的子节点 BBox
        level_min = specs_bbox[:, 0, :]
        level_max = specs_bbox[:, 1, :]
        level_center = 0.5 * (level_min + level_max)

        pattern_min, pattern_max = _child_bbox_patterns(dim)
        child_bboxes_all = np.empty((M, C, 2, dim), dtype=np.float64)

        _compute_child_bboxes_all(
            level_min, level_max, level_center,
            pattern_min, pattern_max,
            child_bboxes_all,
        )

        # 扁平化下一层的 specs
        new_specs_meta, new_specs_bbox = _flatten_specs_kernel(
            level, M, C, dim, node_offsets, child_counts, child_bboxes_all
        )

        return point_indices_next, new_specs_meta, new_specs_bbox, child_counts

    # =========================================================================
    # 递归模式实现
    # =========================================================================

    def _build_recursive(
        self,
        level: int,
        parent_node_index: int,
        point_indices: np.ndarray,
        bbox: np.ndarray,
        points: np.ndarray,
        masses: np.ndarray,
    ) -> Optional[_Node]:
        """
        CN: 递归构建树。
        EN: Recursively build the tree.
        """
        if len(point_indices) == 0:
            return None

        if level < 0 or level >= len(self._nodes_by_level):
            print(f"[ERROR] level={level}, len(_nodes_by_level)={len(self._nodes_by_level)}")
            raise IndexError(f"level={level} out of range")

        # 创建节点
        my_node_index = self._global_node_index
        self._global_node_index += 1
        public_idx = len(self._nodes_by_level[level])

        center, radius = self._compute_node_geometry(bbox)
        node = _Node(
            level=level,
            bbox=bbox,
            parent_node_index=parent_node_index,
            global_node_index=my_node_index,
            public_idx=public_idx,
            center=center,
            radius=radius,
            point_indices=point_indices,
            mass=masses[point_indices].sum(),
        )

        self._nodes_by_level[level].append(node)

        # 停止条件（到达 L1）
        if level == 1:
            node.is_leaf = True
            self._l0_to_l1_map[point_indices] = public_idx
            return node

        # 递归划分
        child_bboxes = self._get_child_bboxes(bbox, center, level)
        point_partitions = self._partition_points(point_indices, points, child_bboxes)

        for child_bbox, child_indices in zip(child_bboxes, point_partitions):
            child_node = self._build_recursive(
                level=level - 1,
                parent_node_index=my_node_index,
                point_indices=child_indices,
                bbox=child_bbox,
                points=points,
                masses=masses,
            )
            if child_node is not None:
                node.children.append(child_node)

        return node

    def _get_child_bboxes(self, bbox: np.ndarray, center: np.ndarray, level: int) -> List[np.ndarray]:
        """
        CN: 获取子包围盒。
        EN: Get child bounding boxes.
        """
        if self.split_mode == "2^n":
            return self._get_child_bboxes_2n(bbox, center)
        elif self.split_mode == "kdtree":
            return self._get_child_bboxes_kdtree(bbox, center, level)
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

    def _get_child_bboxes_2n(self, bbox: np.ndarray, center: np.ndarray) -> List[np.ndarray]:
        """
        CN: 2^n 划分获取子包围盒。
        EN: Get child bounding boxes for 2^n split.
        """
        dim = self.dim
        min_corner, max_corner = bbox
        child_bboxes = []

        for pattern in range(1 << dim):
            child_min = min_corner.copy()
            child_max = max_corner.copy()

            for d in range(dim):
                if pattern & (1 << (dim - 1 - d)):
                    child_min[d] = center[d]
                else:
                    child_max[d] = center[d]

            child_bboxes.append(np.stack([child_min, child_max], axis=0))

        return child_bboxes

    def _get_child_bboxes_kdtree(self, bbox: np.ndarray, center: np.ndarray, level: int) -> List[np.ndarray]:
        """
        CN: KD-Tree 划分获取子包围盒。
        EN: Get child bounding boxes for KD-Tree split.
        """
        dim = self.dim
        depth = self._build_root_level - level
        split_dim = depth % dim

        min_corner, max_corner = bbox

        left_min = min_corner.copy()
        left_max = max_corner.copy()
        left_max[split_dim] = center[split_dim]
        left_bbox = np.stack([left_min, left_max], axis=0)

        right_min = min_corner.copy()
        right_max = max_corner.copy()
        right_min[split_dim] = center[split_dim]
        right_bbox = np.stack([right_min, right_max], axis=0)

        return [left_bbox, right_bbox]

    def _partition_points(
        self,
        point_indices: np.ndarray,
        points: np.ndarray,
        child_bboxes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        CN: 将点划分到子盒子中。
        EN: Partition points into child boxes.
        """
        dim = self.dim
        partitions = [[] for _ in child_bboxes]

        for idx in point_indices:
            point = points[idx]
            for i, bbox in enumerate(child_bboxes):
                tol = 1e-10
                if np.all(point >= bbox[0] - tol) and np.all(point <= bbox[1] + tol):
                    partitions[i].append(idx)
                    break

        return [np.array(p, dtype=np.int32) for p in partitions]

    # =========================================================================
    # 公开层级创建
    # =========================================================================

    def _create_public_levels(self, points: np.ndarray, masses: np.ndarray):
        """
        CN: 创建公开层级接口。
        EN: Create public level interface.
        支持 recursive 和 bucket 两种构建模式。
        """
        self.levels = []

        # L0: 原始点
        n_points = len(points)
        level_0 = HierarchyLevel(
            level_idx=0,
            points=points,
            masses=masses,
            cost_vec=None,
            parent_labels=self._l0_to_l1_map.astype(np.int32),
            radius=np.zeros(n_points, dtype=np.float64),
            knn_indices=None,
            children_offsets=None,
            children_indices=None,
        )
        self.levels.append(level_0)

        # 判断使用哪种构建模式的数据
        use_bucket = hasattr(self, 'layers_data') and self.layers_data

        if use_bucket:
            self._create_public_levels_bucket(points, masses)
        else:
            self._create_public_levels_recursive(points, masses)

    def _create_public_levels_bucket(self, points: np.ndarray, masses: np.ndarray):
        """
        CN: 从 bucket 模式的数据创建公开层级。
        EN: Create public levels from bucket mode data.
        """
        K = self._build_root_level

        for l in range(1, K + 1):
            if l not in self.layer_offsets:
                self.levels[l] = HierarchyLevel(
                    l, np.empty((0, self.dim)), np.empty(0), np.empty(0), None
                )
                continue

            start = self.layer_offsets[l]
            count = len(self.layers_data[l]['meta'])
            end = start + count

            # 切片
            pts = self.flat_centers[start:end]
            mass = self.flat_masses[start:end]
            rad = self.flat_radii[start:end]

            # Parent Labels
            if l < K and (l + 1) in self.layer_offsets:
                parent_offset = self.layer_offsets[l + 1]
                global_parents = self.flat_parent_indices[start:end]
                parent_labels = (global_parents - parent_offset).astype(np.int32)
            else:
                parent_labels = None

            # 计算 children CSR
            children_offsets, children_indices = self._compute_children_csr(l)

            # Internal nodes
            _internal_nodes = self._nodes_by_level[l] if l < len(self._nodes_by_level) else []

            level = HierarchyLevel(
                level_idx=l,
                points=pts,
                masses=mass,
                cost_vec=None,
                child_labels=None,
                radius=rad,
                parent_labels=parent_labels,
                knn_indices=None,
                children_offsets=children_offsets,
                children_indices=children_indices,
                _internal_nodes=_internal_nodes,
            )
            self.levels.append(level)

    def _create_public_levels_recursive(self, points: np.ndarray, masses: np.ndarray):
        """
        CN: 从 recursive 模式的数据创建公开层级。
        EN: Create public levels from recursive mode data.
        """
        for level_idx in range(1, self._build_root_level + 1):
            nodes = self._nodes_by_level[level_idx]
            if not nodes:
                continue

            n_nodes = len(nodes)
            level_points = np.array([n.center for n in nodes], dtype=np.float64)
            level_masses = np.array([n.mass for n in nodes], dtype=np.float64)
            level_radius = np.array([n.radius for n in nodes], dtype=np.float64)

            if level_idx < self._build_root_level:
                parent_nodes = self._nodes_by_level[level_idx + 1]
                global_to_public = {n.global_node_index: n.public_idx for n in parent_nodes}
                parent_labels_public = np.array([
                    global_to_public.get(n.parent_node_index, -1) for n in nodes
                ], dtype=np.int32)
            else:
                parent_labels_public = None

            children_offsets, children_indices = self._compute_children_csr(level_idx)

            level = HierarchyLevel(
                level_idx=level_idx,
                points=level_points,
                masses=level_masses,
                cost_vec=None,
                child_labels=None,
                radius=level_radius,
                parent_labels=parent_labels_public,
                knn_indices=None,
                children_offsets=children_offsets,
                children_indices=children_indices,
                _internal_nodes=nodes,
            )
            self.levels.append(level)

    @property
    def layer_offsets(self) -> Dict[int, int]:
        """
        CN: 计算每层在 flat 数组中的偏移量 (bucket 模式专用)。
        EN: Compute the offset of each layer in the flat array (bucket mode only).
        """
        if not hasattr(self, '_layer_offsets_cache'):
            if hasattr(self, 'layers_data') and self.layers_data:
                offsets = {}
                total = 0
                K = self._build_root_level
                for l in range(1, K + 1):
                    if l in self.layers_data:
                        offsets[l] = total
                        total += len(self.layers_data[l]['meta'])
                    elif l == K:
                        offsets[l] = total
                self._layer_offsets_cache = offsets
            else:
                self._layer_offsets_cache = {}
        return self._layer_offsets_cache

    def _finalize_flat_arrays(self, N: int):
        """
        CN: 从 layers_data 创建 flat 数组。
        EN: Create flat arrays from layers_data.
        """
        K = self._build_root_level
        dim = self.dim

        # 1. 计算层偏移
        self.layer_offsets  # 触发计算
        offsets = self._layer_offsets_cache

        total_nodes = 0
        for l in range(1, K + 1):
            if l in offsets:
                total_nodes += len(self.layers_data[l]['meta'])

        # 2. 分配 flat 数组
        self.flat_centers = np.empty((total_nodes, dim), dtype=np.float64)
        self.flat_radii = np.empty(total_nodes, dtype=np.float64)
        self.flat_masses = np.zeros(total_nodes, dtype=np.float64)
        self.flat_parent_indices = np.full(total_nodes, -1, dtype=np.int64)
        self.flat_levels = np.empty(total_nodes, dtype=np.int32)
        self.flat_is_leaf = np.zeros(total_nodes, dtype=np.int8)
        self.flat_public_idx = np.empty(total_nodes, dtype=np.int32)
        self.flat_child_ptr = np.zeros(total_nodes + 1, dtype=np.int32)

        # 3. Count total children links
        total_children_links = 0
        for lvl in range(2, K + 1):
            if lvl in self.layers_data:
                counts = self.layers_data[lvl]['counts']
                if counts is not None:
                    total_children_links += np.count_nonzero(counts)
        self.flat_child_indices = np.empty(total_children_links, dtype=np.int32)

        # 4. 从 layers_data 填充 flat 数组
        running_child_idx = 0
        for l in range(1, K + 1):
            if l not in offsets:
                continue

            start = offsets[l]
            data = self.layers_data[l]
            meta = data['meta']
            bbox = data['bbox']
            n_nodes = len(meta)

            end = start + n_nodes

            # centers 和 radii
            centers = 0.5 * (bbox[:, 0, :] + bbox[:, 1, :])
            diffs = bbox[:, 1, :] - centers
            radii = np.sqrt(np.sum(diffs ** 2, axis=1))

            self.flat_centers[start:end] = centers
            self.flat_radii[start:end] = radii
            self.flat_levels[start:end] = l
            if l == 1:
                self.flat_is_leaf[start:end] = 1
            self.flat_public_idx[start:end] = meta[:, 2]

            # masses: L1 直接使用叶层聚合质量；高层通过 children 反向聚合
            if 'mass' in data:
                self.flat_masses[start:end] = data['mass']

            # parent indices (global) = parent_rel_idx + parent_layer_offset
            if l < K and (l + 1) in offsets:
                parent_offset = offsets[l + 1]
                self.flat_parent_indices[start:end] = meta[:, 1] + parent_offset

            # children CSR + 质量聚合
            counts = data['counts']
            if counts is not None and l > 1:
                num_children = np.count_nonzero(counts, axis=1)
                child_layer_start = offsets.get(l - 1, 0)
                running_child_idx = _fill_csr_and_mass_kernel(
                    self.flat_child_ptr,
                    self.flat_child_indices,
                    self.flat_masses,
                    self.flat_masses,
                    start,
                    child_layer_start,
                    num_children,
                    running_child_idx,
                )
            else:
                # 没有子节点（根）或叶层（L1）
                self.flat_child_ptr[start : start + n_nodes + 1] = running_child_idx

        self.flat_child_ptr[total_nodes] = running_child_idx

        # 5. Leaf point ptr 和 points
        self.flat_leaf_point_ptr = np.zeros(total_nodes + 1, dtype=np.int32)
        if hasattr(self, '_permuted_point_indices'):
            self.flat_leaf_points = self._permuted_point_indices.astype(np.int32)
        else:
            self.flat_leaf_points = np.arange(N, dtype=np.int32)

        if 1 in self.layers_data:
            l1_start = offsets[1]
            l1_meta = self.layers_data[1]['meta']
            n_l1 = len(l1_meta)
            self.flat_leaf_point_ptr[l1_start : l1_start + n_l1] = l1_meta[:, 3]
            if n_l1 > 0:
                self.flat_leaf_point_ptr[l1_start + n_l1] = l1_meta[:, 4][-1]
            self.flat_leaf_point_ptr[l1_start + n_l1 + 1 :] = N

        # 6. Root indices
        if K in offsets:
            k_start = offsets[K]
            n_k = len(self.layers_data[K]['meta'])
            self.flat_root_indices = np.arange(k_start, k_start + n_k, dtype=np.int32)

        # 7. 重建 _nodes_by_level
        self._rebuild_nodes_from_layers_data()

    def _rebuild_nodes_from_layers_data(self):
        """
        CN: 从 layers_data 重建 _nodes_by_level (bucket 模式)。
        EN: Rebuild _nodes_by_level from layers_data (bucket mode).
        """
        K = self._build_root_level

        for l in range(1, K + 1):
            if l not in self.layers_data:
                continue

            data = self.layers_data[l]
            meta = data['meta']
            bbox = data['bbox']
            n_nodes = len(meta)

            offset = self.layer_offsets.get(l, 0)
            nodes = []

            for i in range(n_nodes):
                node_level = int(meta[i, 0])
                parent_idx = int(meta[i, 1])
                public_idx = int(meta[i, 2])
                global_idx = offset + i

                node = _Node(
                    level=node_level,
                    bbox=bbox[i],
                    parent_node_index=parent_idx,
                    global_node_index=global_idx,
                    public_idx=public_idx,
                )
                node.center = 0.5 * (bbox[i, 0, :] + bbox[i, 1, :])
                node.radius = np.sqrt(np.sum((bbox[i, 1, :] - node.center) ** 2))

                if l == 1:
                    start, end = int(meta[i, 3]), int(meta[i, 4])
                    node.point_indices = self._permuted_point_indices[start:end]
                    node.is_leaf = True

                nodes.append(node)

            if l < len(self._nodes_by_level):
                self._nodes_by_level[l] = nodes

    # =========================================================================
    # CSR 计算和 K-NN
    # =========================================================================

    def _compute_children_csr(self, level_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CN: 计算子节点的 CSR 格式。
        EN: Compute CSR format for child nodes.
        """
        if level_idx == 0:
            return np.array([0], dtype=np.int64), np.array([], dtype=np.int32)

        # 检查是否使用 bucket 模式
        use_bucket = hasattr(self, 'layers_data') and self.layers_data

        if use_bucket and level_idx in self.layer_offsets:
            return self._compute_children_csr_bucket(level_idx)
        else:
            return self._compute_children_csr_recursive(level_idx)

    def _compute_children_csr_bucket(self, level_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CN: 从 flat 数组计算 CSR (bucket 模式)。
        EN: Compute CSR from flat arrays (bucket mode).
        """
        if level_idx not in self.layer_offsets:
            return np.array([0], dtype=np.int64), np.array([], dtype=np.int32)

        start_idx = self.layer_offsets[level_idx]
        n_nodes = len(self.layers_data[level_idx]['meta'])  # 从 layers_data 获取节点数
        end_idx = start_idx + n_nodes

        if level_idx == 1:
            # L1 -> L0
            global_ptrs = self.flat_leaf_point_ptr[start_idx:end_idx + 1]
            offsets = (global_ptrs - global_ptrs[0]).astype(np.int64)

            idx_start = global_ptrs[0]
            idx_end = global_ptrs[-1]
            indices = self.flat_leaf_points[idx_start:idx_end].astype(np.int32)
        else:
            # Lk -> Lk-1
            global_ptrs = self.flat_child_ptr[start_idx:end_idx + 1]
            offsets = (global_ptrs - global_ptrs[0]).astype(np.int64)

            idx_start = global_ptrs[0]
            idx_end = global_ptrs[-1]
            global_indices = self.flat_child_indices[idx_start:idx_end]

            child_layer_offset = self.layer_offsets.get(level_idx - 1, 0)
            indices = (global_indices - child_layer_offset).astype(np.int32)

        return offsets, indices

    def _compute_children_csr_recursive(self, level_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        CN: 从 _nodes_by_level 计算 CSR (recursive 模式)。
        EN: Compute CSR from _nodes_by_level (recursive mode).
        """
        if level_idx == 0:
            return np.array([0], dtype=np.int64), np.array([], dtype=np.int32)

        parent_nodes = self._nodes_by_level[level_idx]

        if level_idx == 1:
            offsets = np.empty(len(parent_nodes) + 1, dtype=np.int64)
            offsets[0] = 0
            indices_list = []
            for i, node in enumerate(parent_nodes):
                indices_list.extend(node.point_indices)
                offsets[i + 1] = len(indices_list)
            return offsets, np.array(indices_list, dtype=np.int32)
        else:
            offsets = np.empty(len(parent_nodes) + 1, dtype=np.int64)
            offsets[0] = 0
            indices_list = []
            for i, node in enumerate(parent_nodes):
                for child in node.children:
                    if child is not None:
                        indices_list.append(child.public_idx)
                offsets[i + 1] = len(indices_list)
            return offsets, np.array(indices_list, dtype=np.int32)

    def _compute_knn(self):
        """
        CN: 计算每层节点的 K-NN。
        EN: Compute K-Nearest-Neighbors for each level's nodes.
        """
        for level_idx, level in enumerate(self.levels):
            nodes = level.points
            n_nodes = len(nodes)

            if n_nodes <= 1:
                level.knn_indices = np.empty((n_nodes, 0), dtype=np.int32)
                continue

            k = min(self.k_neighbors + 1, n_nodes)

            tree = cKDTree(nodes, compact_nodes=False, balanced_tree=False)
            _, indices = tree.query(nodes, k=k, workers=-1)

            if k > 1:
                knn = indices[:, 1:].astype(np.int32)
            else:
                knn = np.empty((n_nodes, 0), dtype=np.int32)

            level.knn_indices = knn

    # =========================================================================
    # Prolongate 和其他接口
    # =========================================================================

    def prolongate(
        self,
        coarse_potential: np.ndarray,
        coarse_level_idx: int,
        fine_level_idx: int,
    ) -> np.ndarray:
        """
        CN: 将粗层势能插值到细层。
        EN: Interpolate coarse level potential to fine level.
        """
        fine_level = self.levels[fine_level_idx]
        coarse_level = self.levels[coarse_level_idx]

        n_fine = len(fine_level.points)
        n_coarse = len(coarse_level.points)

        fine_potential = np.zeros(n_fine, dtype=coarse_potential.dtype)

        if fine_level.parent_labels is not None:
            valid_mask = fine_level.parent_labels >= 0
            fine_potential[valid_mask] = coarse_potential[
                fine_level.parent_labels[valid_mask]
            ]

        return fine_potential

    def get_level(self, level_idx: int) -> HierarchyLevel:
        """
        CN: 获取指定层级。
        EN: Get the specified level.
        """
        return self.levels[level_idx]

    @property
    def num_levels(self) -> int:
        return len(self.levels) - 1 if self.levels else (self._num_levels or 0)

    @num_levels.setter
    def num_levels(self, value: int):
        """
        CN: 设置层级数。
        EN: Set the number of levels.
        """
        self._num_levels = value

    def flatten_for_numba(self):
        """
        CN: 将树结构展平为数组以供 Numba 加速。
        EN: Flatten tree structure into arrays for Numba acceleration.
        在 build() 之后调用，若使用 shielding 则必须调用此方法。
        """
        # 基于当前可见层（可能已截断）而非原始 build 深度来构建扁平树
        available_levels = [
            lvl
            for lvl in range(1, len(self._nodes_by_level))
            if len(self._nodes_by_level[lvl]) > 0
        ]

        node_list = []
        for lvl in available_levels:
            for node in self._nodes_by_level[lvl]:
                node_list.append(node)

        n_nodes = len(node_list)
        if n_nodes == 0:
            self.flat_centers = np.empty((0, self.dim), dtype=np.float64)
            self.flat_radii = np.empty(0, dtype=np.float64)
            self.flat_levels = np.empty(0, dtype=np.int32)
            self.flat_is_leaf = np.empty(0, dtype=np.int8)
            self.flat_public_idx = np.empty(0, dtype=np.int32)
            self.flat_child_ptr = np.zeros(1, dtype=np.int32)
            self.flat_child_indices = np.empty(0, dtype=np.int32)
            self.flat_leaf_point_ptr = np.zeros(1, dtype=np.int32)
            self.flat_leaf_points = np.empty(0, dtype=np.int32)
            self.flat_root_indices = np.empty(0, dtype=np.int32)
            return

        node_to_idx = {id(node): i for i, node in enumerate(node_list)}

        self.flat_centers = np.empty((n_nodes, self.dim), dtype=np.float64)
        self.flat_radii = np.empty(n_nodes, dtype=np.float64)
        self.flat_levels = np.empty(n_nodes, dtype=np.int32)
        self.flat_is_leaf = np.zeros(n_nodes, dtype=np.int8)
        self.flat_public_idx = np.empty(n_nodes, dtype=np.int32)

        total_children = 0
        total_leaf_points = 0
        for node in node_list:
            total_children += len(node.children)
            if getattr(node, "is_leaf", False) and node.point_indices is not None:
                total_leaf_points += len(node.point_indices)

        self.flat_child_ptr = np.empty(n_nodes + 1, dtype=np.int32)
        self.flat_child_indices = np.empty(total_children, dtype=np.int32)
        self.flat_leaf_point_ptr = np.empty(n_nodes + 1, dtype=np.int32)
        self.flat_leaf_points = np.empty(total_leaf_points, dtype=np.int32)

        self.flat_child_ptr[0] = 0
        self.flat_leaf_point_ptr[0] = 0
        running_child = 0
        running_leaf = 0

        for i, node in enumerate(node_list):
            self.flat_centers[i, :] = node.center.astype(np.float64)
            self.flat_radii[i] = float(node.radius)
            self.flat_levels[i] = int(node.level)
            self.flat_public_idx[i] = int(node.public_idx)
            if getattr(node, "is_leaf", False):
                self.flat_is_leaf[i] = 1

            self.flat_child_ptr[i] = running_child
            for child in node.children:
                if child is not None:
                    idx_child = node_to_idx[id(child)]
                    self.flat_child_indices[running_child] = idx_child
                    running_child += 1

            self.flat_leaf_point_ptr[i] = running_leaf
            if getattr(node, "is_leaf", False) and node.point_indices is not None:
                for p in node.point_indices:
                    self.flat_leaf_points[running_leaf] = int(p)
                    running_leaf += 1

        self.flat_child_ptr[n_nodes] = running_child
        self.flat_leaf_point_ptr[n_nodes] = running_leaf

        max_level = max((int(node.level) for node in node_list), default=0)
        roots = []
        for i, node in enumerate(node_list):
            if node.level == max_level and node.parent_node_index == -1:
                roots.append(i)
        if len(roots) == 0:
            for i, node in enumerate(node_list):
                if node.level == max_level:
                    roots.append(i)
        self.flat_root_indices = np.array(roots, dtype=np.int32)

    def _trim_to_target_size(self, target_coarse_size: int = 128):
        """
        CN: 自动截断顶层，保留节点数 >= target_coarse_size 的层。
        EN: Automatically truncate the top level, keeping levels with node count >= target_coarse_size.
        从最粗层开始，找到第一个节点数 >= target_coarse_size 的层，
        截断掉比这个层更粗的所有层。

        Args:
            target_coarse_size: 粗层节点数阈值
        """
        if self.num_levels <= 0:
            return

        # 计算每层的节点数
        num_nodes_per_level = []
        for l in range(1, self._build_root_level + 1):
            if l < len(self.levels):
                num_nodes_per_level.append(len(self.levels[l].points))
            else:
                num_nodes_per_level.append(0)

        if not num_nodes_per_level:
            return

        # 从最粗层开始，找到第一个节点数 >= target_coarse_size 的层
        chosen_level = None
        for l in range(1, self._build_root_level + 1):
            # num_nodes_per_level 索引是 l-1
            if num_nodes_per_level[l - 1] >= target_coarse_size:
                chosen_level = l
            else:
                break

        if chosen_level is None:
            # 所有层节点数都 < target_coarse_size，使用完整树
            if self.verbose:
                print(f"   [Hierarchy] 所有层节点数 < {target_coarse_size}，使用完整树")
            return

        coarsest_depth = self._build_root_level - chosen_level
        if coarsest_depth > 0:
            if self.verbose:
                print(f"   [Hierarchy] 自动选择公开根 L{chosen_level} (节点数={num_nodes_per_level[chosen_level-1]})")
                print(f"   [Hierarchy] 截断顶层：丢弃 L{chosen_level+1}..L{self._build_root_level}")

            # 截断 levels
            self.levels = self.levels[0: chosen_level + 1]
            self.num_levels = chosen_level

            # 新的最粗层不再有 parent_labels
            if self.levels:
                self.levels[chosen_level].parent_labels = None


def align_hierarchy_depths(
    hX: "TreeHierarchy",
    hY: "TreeHierarchy",
    verbose: bool = True,
) -> None:
    """
    CN: 把 hX, hY 的层级数对齐到同一个 num_levels。
    EN: Align the number of levels between hX and hY to the same num_levels.
    """
    Lx = hX.num_levels
    Ly = hY.num_levels
    L_common = min(Lx, Ly)

    if verbose:
        print(
            f"[align_hierarchy_depths] "
            f"X 有 {Lx+1} 层, Y 有 {Ly+1} 层, "
            f"对齐到 {L_common+1} 层。"
        )

    def _trim(h: "TreeHierarchy", L_target: int):
        if h.num_levels > L_target:
            h.levels = h.levels[0 : L_target + 1]
            h.num_levels = L_target
            if h.levels:
                h.levels[L_target].parent_labels = None

    _trim(hX, L_common)
    _trim(hY, L_common)

    if verbose:
        print(f"   对齐后: X 和 Y 都有 {L_common + 1} 层 (L0..L{L_common})")


def print_hierarchy_info(hierarchy: "TreeHierarchy", all: bool = False):
    """
    CN: 打印层次结构信息。
    EN: Print hierarchy information.

    Args:
        hierarchy: TreeHierarchy 实例
        all: 是否打印详细信息
    """
    print(f"层次结构信息:")
    print(f"- dim: {hierarchy.dim}")
    print(f"- num_levels: {hierarchy.num_levels + 1}")
    for level_index in reversed(range(hierarchy.num_levels + 1)):
        level = hierarchy.levels[level_index]
        print(f"  - level {level_index}:")
        print(f"    - num_level_points: {len(level.points)}")

def check_level_mass_balance(
    level_s: HierarchyLevel,
    level_t: HierarchyLevel,
    tol: float = 1e-12,
) -> Dict[str, float]:
    """
    CN: 诊断某一层 source/target 的总质量是否一致。
    EN: Diagnose whether the total mass of source/target at a certain level is consistent.
    """
    sum_s = float(np.sum(level_s.masses))
    sum_t = float(np.sum(level_t.masses))
    diff = sum_s - sum_t
    return {
        "sum_source": sum_s,
        "sum_target": sum_t,
        "diff": diff,
        "ok": abs(diff) <= tol,
    }

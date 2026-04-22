# hierarchical_ot/core/base.py

from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

# 通用数组类型 (numpy array or torch tensor)
ArrayType = Union[np.ndarray, Any]


@dataclass
class HierarchyLevel:
    """
    描述单一层级的数据结构。
    统一支持 Cluster (HELLO), Tree (HALO), 和 Grid (MGPD) 三种模式。
    """
    level_idx: int
    points: ArrayType        # (N, D) 坐标/特征
    masses: ArrayType        # (N,)   质量/权重
    cost_vec: Optional[ArrayType] = None  # 低秩模式下的 ||x||^2

    # ========== 父子映射关系 (Cluster/Grid/Tree 共用) ==========
    # child_labels[i] = p 表示第 i 个点属于上一层的第 p 个点
    child_labels: Optional[ArrayType] = None
    
    # ========== Tree (HALO) 特有字段 ==========
    # 每个节点的半径 (用于 shielding)
    radius: Optional[ArrayType] = None
    
    # 父节点标签：parent_labels[i] = p 表示第 i 个点属于下一层 (更粗层) 的第 p 个点
    # 注意：这是 child_labels 的反方向映射
    parent_labels: Optional[ArrayType] = None
    
    # K-NN 索引 (用于 shielding 的邻居搜索)
    knn_indices: Optional[ArrayType] = None
    
    # 子节点 CSR 格式 (用于 refine_duals)
    # children_offsets[i]: 第 i 个节点的子节点在 children_indices 中的起始位置
    # children_indices: 扁平化的子节点索引列表
    children_offsets: Optional[ArrayType] = None
    children_indices: Optional[ArrayType] = None
    
    # ========== 内部节点 (HALO Tree 特有) ==========
    # 用于 shielding 树搜索的内部节点列表
    _internal_nodes: List[Any] = field(default_factory=list, repr=False)


class BaseHierarchy(ABC):
    """
    层级结构的基类。
    """

    def __init__(self, levels: List[HierarchyLevel]):
        self.levels = []

    @property
    def finest_level(self) -> HierarchyLevel:
        return self.levels[0]

    @property
    def coarsest_level(self) -> HierarchyLevel:
        return self.levels[-1]

    @abstractmethod
    def prolongate(self,
                   coarse_potential: ArrayType,
                   coarse_level_idx: int,
                   fine_level_idx: int) -> ArrayType:
        """
        [核心统一接口] 将上一层(Coarse)的势能/解，插值/广播到当前层(Fine)。
        """
        pass


class BaseStrategy(ABC):
    """
    筛选策略（Pricing Strategy）的基类。
    """
    @abstractmethod
    def generate(
        self,
        T_prev: Any,
        duals_prev: Tuple[ArrayType, ArrayType],
        level_cache: Dict[str, Any],
        **kwargs
    ) -> Tuple[ArrayType, ArrayType]:
        """
        生成候选用 active set。

        Args:
            T_prev: 上一轮的传输计划 (通常是稀疏矩阵)
            duals_prev: (u, v) 对偶变量
            level_cache: 当前层级的缓存数据 (S, T, cost_vec 等)
            beta: 探索系数

        Returns:
            (rows, cols): 新增的活跃边索引
        """
        pass

    def close(self) -> None:
        """清理资源 (如 Faiss 索引)"""
        pass


class ActiveSupportStrategy(ABC):
    """
    Active Support (活跃变量集) 策略的抽象基类。
    
    这是 Pricing/Cleaning 策略的更底层抽象，用于管理:
    1. 活跃变量集的初始化 (从上一层继承或全新构建)
    2. 活跃变量集的更新 (基于对偶解的变化)
    
    适用于 HALO (Shielding) 和 HELLO (MIPS) 两种架构。
    """
    
    @abstractmethod
    def initialize_support(
        self,
        level_s: 'HierarchyLevel',
        level_t: 'HierarchyLevel',
        x_init: Optional[ArrayType] = None,
        **kwargs
    ) -> Dict[str, ArrayType]:
        """
        初始化活跃变量集。
        
        Args:
            level_s: 源层级
            level_t: 目标层级
            x_init: 初始对偶势 (可选)
            
        Returns:
            Dict 包含:
                - 'keep': 1D 扁平索引 (N_X * N_Y)
                - 'keep_coord': 结构体数组 [('idx1', 'i4'), ('idx2', 'i4')]
                - 'y_init': 初始对偶解 (可选)
        """
        pass
    
    @abstractmethod
    def update_active_support(
        self,
        x_solution: ArrayType,
        y_solution_last: Dict[str, ArrayType],
        level_s: 'HierarchyLevel',
        level_t: 'HierarchyLevel',
        hierarchy_s: 'BaseHierarchy',
        hierarchy_t: 'BaseHierarchy',
        **kwargs
    ) -> Dict[str, ArrayType]:
        """
        更新活跃变量集 (内迭代调用)。
        使用 shielding: y_last + shield + yhat。

        Args:
            x_solution: 当前对偶解 (u, v)
            y_solution_last: 上一次的原始解 {'y': ..., 'keep': ...}
            level_s: 当前源层级
            level_t: 当前目标层级
            hierarchy_s: 完整源层次结构
            hierarchy_t: 完整目标层次结构

        Returns:
            Dict 包含更新后的活跃集信息
            调用方可通过额外 kwarg 请求仅返回最小必要字段。
        """
        pass
    
    def close(self) -> None:
        """清理资源"""
        pass


class ActiveSupport:
    """
    管理列生成过程中的活跃集 (Active Support / Working Set)。
    存储当前所有被激活的 (row, col) 对及其对应的 Cost 和 Primal Solution。
    """

    def __init__(self, level_cache: Dict[str, Any], track_creation: bool = False):
        # 核心数据结构 (Numpy Arrays)
        self.rows = np.empty(0, dtype=np.int32)
        self.cols = np.empty(0, dtype=np.int32)
        self.c_vec = np.empty(0, dtype=np.float32)
        self.x_prev = np.empty(0, dtype=np.float32)  # Warm start primal
        self.keys = np.empty(0, dtype=np.int64)      # 用于快速查找/去重的唯一键

        # 辅助信息
        self.level_cache = level_cache
        self.track_creation = track_creation
        self.creation_iteration = np.empty(
            0, dtype=np.int32) if track_creation else None

        # 预计算 n_target，用于生成 key
        # key = row * n_target + col
        if 'b' in level_cache:
            self.n_target = level_cache['b'].shape[0]
        else:
            self.n_target = level_cache['T'].shape[0]

    @property
    def size(self) -> int:
        return self.rows.size

    def add_pairs(self, new_rows: np.ndarray, new_cols: np.ndarray, new_costs: np.ndarray, iter_idx: int = -1):
        """添加新的候选列到活跃集"""
        if len(new_rows) == 0:
            return

        # 1. 拼接基础数组
        self.rows = np.concatenate([self.rows, new_rows])
        self.cols = np.concatenate([self.cols, new_cols])
        self.c_vec = np.concatenate([self.c_vec, new_costs])

        # 2. 扩展 x_prev (新加入的变量初始值为 0)
        zeros = np.zeros(len(new_rows), dtype=np.float32)
        self.x_prev = np.concatenate([self.x_prev, zeros])

        # 3. 更新 Keys
        new_keys = new_rows.astype(np.int64) * \
            self.n_target + new_cols.astype(np.int64)
        self.keys = np.concatenate([self.keys, new_keys])

        # 4. 追踪创建迭代 (用于 AgeCleaning)
        if self.track_creation:
            iter_arr = np.full(len(new_rows), iter_idx, dtype=np.int32)
            self.creation_iteration = np.concatenate(
                [self.creation_iteration, iter_arr])

    def prune(self, mask: np.ndarray):
        """根据布尔掩码保留活跃集中的元素 (用于 Cleaning)"""
        self.rows = self.rows[mask]
        self.cols = self.cols[mask]
        self.c_vec = self.c_vec[mask]
        self.x_prev = self.x_prev[mask]
        self.keys = self.keys[mask]

        if self.track_creation:
            self.creation_iteration = self.creation_iteration[mask]

    def __getitem__(self, item):
        """兼容性接口：允许像字典一样访问 (用于兼容旧版 CleaningStrategy)"""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"ActiveSet has no attribute {item}")


__all__ = [
    "ArrayType",
    "HierarchyLevel",
    "BaseHierarchy",
    "BaseStrategy",
    "ActiveSupportStrategy",
    "ActiveSupport",
]

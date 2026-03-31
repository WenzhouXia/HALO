

from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

ArrayType = Union[np.ndarray, Any]

@dataclass
class HierarchyLevel:
    
    level_idx: int
    points: ArrayType        
    masses: ArrayType        
    cost_vec: Optional[ArrayType] = None  

    
    
    child_labels: Optional[ArrayType] = None
    
    
    
    radius: Optional[ArrayType] = None
    
    
    
    parent_labels: Optional[ArrayType] = None
    
    
    knn_indices: Optional[ArrayType] = None
    
    
    
    
    children_offsets: Optional[ArrayType] = None
    children_indices: Optional[ArrayType] = None
    
    
    
    _internal_nodes: List[Any] = field(default_factory=list, repr=False)

class BaseHierarchy(ABC):
    

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
        
        pass

class BaseStrategy(ABC):
    
    @abstractmethod
    def generate(
        self,
        T_prev: Any,
        duals_prev: Tuple[ArrayType, ArrayType],
        level_cache: Dict[str, Any],
        **kwargs
    ) -> Tuple[ArrayType, ArrayType]:
        
        pass

    def close(self) -> None:
        
        pass

class ActiveSupportStrategy(ABC):
    
    
    @abstractmethod
    def initialize_support(
        self,
        level_s: 'HierarchyLevel',
        level_t: 'HierarchyLevel',
        x_init: Optional[ArrayType] = None,
        **kwargs
    ) -> Dict[str, ArrayType]:
        
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
        
        pass
    
    def close(self) -> None:
        
        pass

class ActiveSupport:
    

    def __init__(self, level_cache: Dict[str, Any], track_creation: bool = False):
        
        self.rows = np.empty(0, dtype=np.int32)
        self.cols = np.empty(0, dtype=np.int32)
        self.c_vec = np.empty(0, dtype=np.float32)
        self.x_prev = np.empty(0, dtype=np.float32)  
        self.keys = np.empty(0, dtype=np.int64)      

        
        self.level_cache = level_cache
        self.track_creation = track_creation
        self.creation_iteration = np.empty(
            0, dtype=np.int32) if track_creation else None

        
        
        if 'b' in level_cache:
            self.n_target = level_cache['b'].shape[0]
        else:
            self.n_target = level_cache['T'].shape[0]

    @property
    def size(self) -> int:
        return self.rows.size

    def add_pairs(self, new_rows: np.ndarray, new_cols: np.ndarray, new_costs: np.ndarray, iter_idx: int = -1):
        
        if len(new_rows) == 0:
            return

        
        self.rows = np.concatenate([self.rows, new_rows])
        self.cols = np.concatenate([self.cols, new_cols])
        self.c_vec = np.concatenate([self.c_vec, new_costs])

        
        zeros = np.zeros(len(new_rows), dtype=np.float32)
        self.x_prev = np.concatenate([self.x_prev, zeros])

        
        new_keys = new_rows.astype(np.int64) *            self.n_target + new_cols.astype(np.int64)
        self.keys = np.concatenate([self.keys, new_keys])

        
        if self.track_creation:
            iter_arr = np.full(len(new_rows), iter_idx, dtype=np.int32)
            self.creation_iteration = np.concatenate(
                [self.creation_iteration, iter_arr])

    def prune(self, mask: np.ndarray):
        
        self.rows = self.rows[mask]
        self.cols = self.cols[mask]
        self.c_vec = self.c_vec[mask]
        self.x_prev = self.x_prev[mask]
        self.keys = self.keys[mask]

        if self.track_creation:
            self.creation_iteration = self.creation_iteration[mask]

    def __getitem__(self, item):
        
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"ActiveSet has no attribute {item}")

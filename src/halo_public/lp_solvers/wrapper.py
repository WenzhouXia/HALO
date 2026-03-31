from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from typing import Dict

@dataclass
class SolverResult:
    
    success: bool
    x: np.ndarray = None        
    y: np.ndarray = None        
    obj_val: float = 0.0
    dual_obj_val: float = None
    duration: float = 0.0
    iterations: int = 0
    peak_mem: float = 0.0       
    termination_reason: str = None
    
    primal_feas: float = None   
    dual_feas: float = None     
    gap: float = None           
    solver_diag: Dict = None    

class LPSolver(ABC):
    
    @abstractmethod
    def solve(
        self,
        c: np.ndarray,            
        A_csc: sparse.csc_matrix,  
        b_eq: np.ndarray,         
        lb: np.ndarray,           
        ub: np.ndarray,           
        n_eqs: int,
        warm_start_primal=None,
        warm_start_dual=None,
        tolerance: Dict[str, float] = {
            'objective': 1e-6, 'primal': 1e-6, 'dual': 1e-6},
        verbose: int = 0,
        **kwargs
    ) -> SolverResult:
        pass

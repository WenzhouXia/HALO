import numpy as np
from scipy import sparse
import hprlp

def main():
    print("=" * 70)
    print("HPRLP Example: Direct LP from Arrays - Python")
    print("=" * 70)
    
    # Problem: minimize -3*x1 - 5*x2
    # subject to x1 + 2*x2 <= 10
    #            3*x1 + x2 <= 12
    #            x1, x2 >= 0

    # Define the LP problem
    # Objective coefficients (c)
    c = np.array([-3.0, -5.0])

    # Constraint matrix (A) in CSR format
    A = sparse.csr_matrix([
        [1.0, 2.0],  # x1 + 2*x2 <= 10
        [3.0, 1.0]   # 3*x1 + x2 <= 12
    ])
    
    # Constraint bounds (AL and AU)
    AL = np.array([-np.inf, -np.inf])  # Lower bounds
    AU = np.array([10.0, 12.0])         # Upper bounds
    
    # Variable bounds (l and u)
    l = np.array([0.0, 0.0])           # x1, x2 >= 0
    u = np.array([np.inf, np.inf])     # Unbounded above
    
    # Step 1: Create the model from arrays
    print("Creating model from arrays...")
    model = hprlp.Model.from_arrays(A, AL, AU, l, u, c)
    print(f"Model created: {model.m} constraints, {model.n} variables")
    
    # Step 2: Set solver parameters
    param = hprlp.Parameters()
    param.stop_tol = 1e-9
    param.device_number = 0
    param.use_bc_scaling = False
    param.use_Ruiz_scaling = False
    # Step 3: Solve the model
    print("Solving the model...")
    result = model.solve(param)
    
    # Step 4: Display results
    print("=" * 70)
    print("Solution Summary")
    print("=" * 70)
    print(f"Status: {result.status}")
    print(f"Iterations: {result.iter}")
    print(f"Time: {result.time:.2f} seconds")
    print(f"Primal Objective: {result.primal_obj:.12e}")
    print(f"Residual: {result.residuals:.12e}")
    
    print("Primal solution:")
    print(f"  x1 = {result.x[0]:.6f}")
    print(f"  x2 = {result.x[1]:.6f}")
    
    print("=" * 70)
    
    # Step 5: Free the model
    model.free()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install HPRLP first:")
        print("  cd bindings/python")
        print("  python -m pip install .")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

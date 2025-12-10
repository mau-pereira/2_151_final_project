import numpy as np

def controllability_matrix(A, B):
    """
    Build controllability matrix:
    C = [B, AB, A^2 B, ..., A^{n-1} B]
    and return (C, rank).
    """
    n = A.shape[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    rank = np.linalg.matrix_rank(C)
    return C, rank


def analyze_system(name, A, B):
    n = A.shape[0]
    print(f"\n=== {name.upper()} configuration ===")

    #controllability
    C, rank_C = controllability_matrix(A, B)
    print(f"Controllability rank: {rank_C} / {n}")
    if rank_C == n:
        print("  -> (A,B) is fully controllable.")
    else:
        print("  -> (A,B) is NOT fully controllable.")

    #open-loop eigenvalues
    eig_A = np.linalg.eigvals(A)
    print("Open-loop eigenvalues of A:")
    print(eig_A)

    if np.all(np.real(eig_A) < 0):
        print("  -> Open-loop system is asymptotically stable (Hurwitz).")
    elif np.any(np.real(eig_A) > 0):
        print("  -> Open-loop system is unstable.")
    else:
        print("  -> Open-loop system is marginally stable (some Re(λ) = 0).")

#matrices for each height configuration
A_low = np.array([
    [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  1.        ],
    [14.7363243 , -1.92269375,  0.34938238,  0.        ,  0.        ,  0.        ],
    [-37.5025373, -25.17099183, 0.11829813,  0.        ,  0.        ,  0.        ],
    [19.54346897, 70.46441524, -9.54627958, 0.        ,  0.        ,  0.        ],
])

B_low = np.array([
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.79161343, -0.61486476, -2.38802913],
    [-0.61486476,  2.62395205, -3.96866928],
    [-2.38802913, -3.96866928, 74.80793658],
])

A_med = np.array([
    [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        , 1.        , 0.        ],
    [0.        , 0.        , 0.        , 0.        , 0.        , 1.        ],
    [26.6301566 , 0.43136622, -0.02761779, 0.        , 0.        , 0.        ],
    [-39.445433 , -3.37875519, 1.34019919, 0.        , 0.        , 0.        ],
    [17.8932841 , 2.4016591 , -13.638819 , 0.        , 0.        , 0.        ],
])

B_med = np.array([
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.7517243 , -1.03965173,  0.45103043],
    [-1.03965173,  4.18083851, -11.21404607],
    [ 0.45103043, -11.21404607, 92.96197751],
])

A_high = np.array([
    [ 0.        ,  0.        ,  0.        ,  1.        , 0.        , 0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        , 1.        , 0.        ],
    [ 0.        ,  0.        ,  0.        ,  0.        , 0.        , 1.        ],
    [29.63771194, -11.74026758,  0.49202434, 0.        , 0.        , 0.        ],
    [-47.9448276 ,  41.92672936, -6.21430362, 0.        , 0.        , 0.        ],
    [22.48078316, -49.55028039, 54.75450916, 0.        , 0.        , 0.        ],
])

B_high = np.array([
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 1.35448011, -2.95293271,  2.36507837],
    [-2.95293271,  9.17414954, -15.69925031],
    [ 2.36507837, -15.69925031, 96.55511022],
])

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    analyze_system("low", A_low, B_low)
    analyze_system("medium", A_med, B_med)
    analyze_system("high", A_high, B_high)

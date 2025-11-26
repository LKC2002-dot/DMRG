import torch

# 横场ising模型的mpo(OBC)
# 左端点mpo: shape (1, D, d, d)
# 1×3的行向量:[-hSx , -J Sz , I]

# 中间mpo张量: shape (D, D, d, d)
# [   I    0      0 ]
# [  Sz    0      0 ]
# [ -hSx -J Sz    I ]

# 右端点mpo张量:shape (D, 1, d, d)
# 3×1 的“列向量”:
# [  I   ]
# [  Sz  ]
# [ -hSx ]


# 定义PAULI矩阵
def spin_ops(dtype=torch.float64, device="cpu"):
    I = torch.eye(2, dtype=dtype, device=device)
    Sx = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    Sz = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype, device=device)
    return I, Sx, Sz


def build_tfim_mpo(L, J, h, dtype=torch.float64, device="cpu"):

    I, Sx, Sz = spin_ops(dtype=dtype, device=device)
    d = 2
    D = 3
    mpo = []

    W_L = torch.zeros(1, D, d, d, dtype=dtype, device=device)
    W_L[0, 0, :, :] = -h * Sx
    W_L[0, 1, :, :] = -J * Sz
    W_L[0, 2, :, :] = I
    mpo.append(W_L)

    for _ in range(1, L - 1):
        W = torch.zeros(D, D, d, d, dtype=dtype, device=device)
        W[0, 0, :, :] = I
        W[1, 0, :, :] = Sz
        W[2, 0, :, :] = -h * Sx
        W[2, 1, :, :] = -J * Sz
        W[2, 2, :, :] = I
        mpo.append(W)

    W_R = torch.zeros(D, 1, d, d, dtype=dtype, device=device)
    W_R[0, 0, :, :] = I
    W_R[1, 0, :, :] = Sz
    W_R[2, 0, :, :] = -h * Sx
    mpo.append(W_R)

    return mpo

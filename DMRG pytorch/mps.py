import torch


# 开放边界条件mps
def obc_mps(L, d, D, dtype=torch.float64, device="cpu"):

    mps = []
    for n in range(L):
        if n == 0:
            A = torch.randn(1, d, D, dtype=dtype, device=device)
        elif n == L - 1:
            A = torch.randn(D, d, 1, dtype=dtype, device=device)
        else:
            A = torch.randn(D, d, D, dtype=dtype, device=device)
        mps.append(A)
    return mps

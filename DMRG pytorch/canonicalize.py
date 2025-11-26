# 左正交,右正交,中心正交化MPS

import torch


def left_canonicalize(mps, normalize=False, full_matrices=False):
    mps_svd = [A.clone() for A in mps]
    L = len(mps_svd)

    for n in range(L - 1):
        A = mps_svd[n]
        Dl, d, Dr = A.shape
        A_mat = A.reshape(Dl * d, Dr)
        U, S, Vh = torch.linalg.svd(A_mat, full_matrices=full_matrices)

        if normalize:
            norm_factor = torch.norm(S)
            U = U / norm_factor
            S = S / norm_factor

        Dr_new = U.shape[1]
        mps_svd[n] = U.reshape(Dl, d, Dr_new)

        R = torch.diag(S) @ Vh
        A_next = mps_svd[n + 1]
        A_next = torch.tensordot(R, A_next, dims=([1], [0]))
        mps_svd[n + 1] = A_next

    return mps_svd


def right_canonicalize(mps, normalize=False, full_matrices=False):
    mps_svd = [A.clone() for A in mps]
    L = len(mps_svd)

    for n in reversed(range(1, L)):
        A = mps_svd[n]
        Dl, d, Dr = A.shape
        A_mat = A.reshape(Dl, d * Dr)
        U, S, Vh = torch.linalg.svd(A_mat, full_matrices=full_matrices)

        if normalize:
            norm_factor = torch.norm(S)
            U = U / norm_factor
            S = S / norm_factor

        Dl_new = Vh.shape[0]
        mps_svd[n] = Vh.reshape(Dl_new, d, Dr)

        Lmat = U @ torch.diag(S)
        A_prev = mps_svd[n - 1]
        A_prev = torch.tensordot(A_prev, Lmat, dims=([2], [0]))
        mps_svd[n - 1] = A_prev

    return mps_svd


def center_canonicalize(mps, center, normalize=False, full_matrices=False):
    mps_svd = [A.clone() for A in mps]
    L = len(mps_svd)
    assert 0 <= center < L, "The center must be within the range [0, L-1]."

    for n in range(center):
        A = mps_svd[n]
        Dl, d, Dr = A.shape
        A_mat = A.reshape(Dl * d, Dr)
        U, S, Vh = torch.linalg.svd(A_mat, full_matrices=full_matrices)

        if normalize:
            norm_factor = torch.norm(S)
            U = U / norm_factor
            S = S / norm_factor

        Dr_new = U.shape[1]
        mps_svd[n] = U.reshape(Dl, d, Dr_new)

        R = torch.diag(S) @ Vh
        A_next = mps_svd[n + 1]
        A_next = torch.tensordot(R, A_next, dims=([1], [0]))
        mps_svd[n + 1] = A_next

    for n in range(L - 1, center, -1):
        A = mps_svd[n]
        Dl, d, Dr = A.shape
        A_mat = A.reshape(Dl, d * Dr)
        U, S, Vh = torch.linalg.svd(A_mat, full_matrices=full_matrices)

        if normalize:
            norm_factor = torch.norm(S)
            U = U / norm_factor
            S = S / norm_factor

        Dl_new = Vh.shape[0]
        mps_svd[n] = Vh.reshape(Dl_new, d, Dr)

        Lmat = U @ torch.diag(S)
        A_prev = mps_svd[n - 1]
        A_prev = torch.tensordot(A_prev, Lmat, dims=([2], [0]))
        mps_svd[n - 1] = A_prev

    return mps_svd

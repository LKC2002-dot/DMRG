# 这部分代码用来检查一些函数的正确性，不会参与实际的dmrg计算


import torch


def calculate_mps_norm_and_normalize(mps):

    device = mps[0].device
    dtype = mps[0].dtype

    v = torch.ones(1, 1, device=device, dtype=dtype)
    v_list = []

    for n, A in enumerate(mps):
        v_tilde = torch.einsum("ab,asc,bsd->cd", v, A.conj(), A)
        v_n = v_tilde.norm()
        v_list.append(v_n)

        # 归一化
        v = v_tilde / v_n
        mps[n] = A / torch.sqrt(v_n)

    mps_norm = torch.prod(torch.stack(v_list))

    # 返回未归一化mps的模方
    return mps_norm


def check_left_canonical(mps, atol=1e-10):

    max_errors = []
    for n, A in enumerate(mps):
        Dl, d, Dr = A.shape

        G = torch.einsum("asb,asC->bC", A.conj(), A)

        I = torch.eye(Dr, dtype=A.dtype, device=A.device)
        err = (G - I).abs().max().item()
        max_errors.append(err)
        print(f"[left] site {n}: max |G-I| = {err:.5e}")

    return max_errors


def check_right_canonical(mps, atol=1e-10):

    max_errors = []
    for n, A in enumerate(mps):
        Dl, d, Dr = A.shape

        H = torch.einsum("asb,Asb->aA", A.conj(), A)

        I = torch.eye(Dl, dtype=A.dtype, device=A.device)
        err = (H - I).abs().max().item()
        max_errors.append(err)
        print(f"[right] site {n}: max |H-I| = {err:.5e}")

    return max_errors


def check_center_canonical(mps, center, atol=1e-10):
    L = len(mps)
    assert 0 <= center < L

    print(f"=== check mixed canonical with center = {center} ===")

    # 检查左正交
    print("checking left part 0..center-1:")
    check_left_canonical(mps[:center], atol=atol)

    # 检查右正交
    print("checking right part center+1..L-1:")
    right_errors = check_right_canonical(mps[center + 1 :], atol=atol)

    # 检查中心正交
    print("checking center site:")
    A = mps[center]
    Dl, d, Dr = A.shape

    G = torch.einsum("asb,asC->bC", A.conj(), A)  # 左正交的 G
    H = torch.einsum("asb,Asb->aA", A.conj(), A)  # 右正交的 H

    I_r = torch.eye(Dr, dtype=A.dtype, device=A.device)
    I_l = torch.eye(Dl, dtype=A.dtype, device=A.device)
    err_left_center = (G - I_r).abs().max().item()
    err_right_center = (H - I_l).abs().max().item()

    print(f"[center] site {center}: as left-ortho max |G-I| = {err_left_center:.5e}")
    print(f"[center] site {center}: as right-ortho max |H-I| = {err_right_center:.5e}")

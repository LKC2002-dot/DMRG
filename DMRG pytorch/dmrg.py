# dmrg主程序,包括左右环境的构建,以及sweep的实现

import torch as tc


# 计算<psi|H|psi>/<psi|psi>,用于每次sweep后能量计算
def calculate_energy(mps, mpo):
    dtype = mps[0].dtype
    device = mps[0].device

    L_env = tc.ones(1, 1, 1, dtype=dtype, device=device)
    for A, W in zip(mps, mpo):
        L_env = tc.einsum("ame, asb, mrst, etf -> brf", L_env, A.conj(), W, A)
    num = L_env.squeeze().real

    A0 = mps[0]
    denom = tc.einsum("asb, asb ->", A0.conj(), A0).real

    return num / denom


# mps归一化,使得<psi|psi>=1
def normalize_mps(mps):

    device = mps[0].device
    dtype = mps[0].dtype
    v = tc.ones(1, 1, device=device, dtype=dtype)
    v_list = []

    for n, A in enumerate(mps):
        v_tilde = tc.einsum("ab,asc,bsd->cd", v, A.conj(), A)
        v_n = v_tilde.norm()
        v_list.append(v_n)
        v = v_tilde / v_n
        mps[n] = A / tc.sqrt(v_n)

    return mps


# 构建左环境,返回左环境列表:L_env,l_env[n]对应第n个site左侧的环境,为一个三阶张量。
def build_left_envs(mps, mpo):

    L = len(mps)
    dtype = mps[0].dtype
    device = mps[0].device
    L_env = [None] * (L + 1)
    L_env[0] = tc.ones(1, 1, 1, dtype=dtype, device=device)

    for n in range(L):
        A = mps[n]  # (Dl, d, Dr)
        W = mpo[n]  # (Ml, Mr, d, d)
        L_prev = L_env[n]
        L_next = tc.einsum("ame, asb, mrst, etf -> brf", L_prev, A.conj(), W, A)
        L_env[n + 1] = L_next

    return L_env


# 构建右环境,返回右环境列表:R_env, R_env[n]对应第n - 1个site右侧的环境,为一个三阶张量。
def build_right_envs(mps, mpo):

    L = len(mps)
    dtype = mps[0].dtype
    device = mps[0].device
    R_env = [None] * (L + 1)
    R_env[L] = tc.ones(1, 1, 1, dtype=dtype, device=device)

    for n in reversed(range(L)):
        A = mps[n]  # (Dl, d, Dr)
        W = mpo[n]  # (Ml, Mr, d, d)
        R_next = R_env[n + 1]  # (Dr, Mr, Dr)
        R_curr = tc.einsum("asb, mrst, etc, brc -> ame", A, W, A.conj(), R_next)
        R_env[n] = R_curr

    return R_env


# 从左到右half-sweep
def two_site_update_L2R(mps, mpo, D_max):

    L = len(mps)
    dtype = mps[0].dtype
    device = mps[0].device
    R_env = build_right_envs(mps, mpo)
    L_env_curr = tc.ones(1, 1, 1, dtype=dtype, device=device)  # 初始左环境L_env[0]

    for n in range(L - 1):
        A = mps[n]
        B = mps[n + 1]
        Dl, d, Dm = A.shape  # (Dl, d, Dm)
        Dm_B, d_B, Dr = B.shape  # (Dm, d, Dr)
        assert Dm_B == Dm and d_B == d

        W1 = mpo[n]
        W2 = mpo[n + 1]
        L_env_n = L_env_curr
        R_env_np2 = R_env[n + 2]

        # 构造Heff[astb;euvf]
        Heff = tc.einsum("ale, lmsu, mrtv, brf -> astbeuvf", L_env_n, W1, W2, R_env_np2)
        dim = Dl * d * d * Dr
        assert Heff.numel() == dim * dim
        Heff = Heff.reshape(dim, dim)

        _, evecs = tc.linalg.eigh(Heff)
        theta_mat = evecs[:, 0].reshape(Dl * d, d * Dr)
        U, S, Vh = tc.linalg.svd(theta_mat, full_matrices=False)

        chi = min(D_max, S.numel())
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        A_new = U.reshape(Dl, d, chi)
        B_new = (tc.diag(S) @ Vh).reshape(chi, d, Dr)

        mps[n] = A_new
        mps[n + 1] = B_new

        # 更新当前左环境
        L_env_curr = tc.einsum(
            "ame, asb, mrst, etf -> brf", L_env_n, A_new.conj(), W1, A_new
        )

    return mps


# 从右到左half-sweep
def two_site_update_R2L(mps, mpo, D_max):

    L = len(mps)
    dtype = mps[0].dtype
    device = mps[0].device
    L_env = build_left_envs(mps, mpo)
    R_env_curr = tc.ones(1, 1, 1, dtype=dtype, device=device)

    for n in reversed(range(L - 1)):
        A = mps[n]
        B = mps[n + 1]
        Dl, d, Dm = A.shape
        Dm_B, d_B, Dr = B.shape
        assert Dm_B == Dm and d_B == d

        W1 = mpo[n]
        W2 = mpo[n + 1]
        L_env_n = L_env[n]
        R_env_np2 = R_env_curr

        # 构造Heff
        Heff = tc.einsum("ale, lmsu, mrtv, brf -> astbeuvf", L_env_n, W1, W2, R_env_np2)
        dim = Dl * d * d * Dr
        assert Heff.numel() == dim * dim
        Heff = Heff.reshape(dim, dim)

        _, evecs = tc.linalg.eigh(Heff)
        theta = evecs[:, 0].reshape(Dl, d, d, Dr)
        theta_mat = theta.reshape(Dl * d, d * Dr)
        U, S, Vh = tc.linalg.svd(theta_mat, full_matrices=False)
        chi = min(D_max, S.numel())
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]

        A_new = (U @ tc.diag(S)).reshape(Dl, d, chi)
        B_new = Vh.reshape(chi, d, Dr)
        mps[n] = A_new
        mps[n + 1] = B_new

        # 更新当前右环境
        R_env_curr = tc.einsum(
            "asb, mrst, etc, brc -> ame", B_new, W2, B_new.conj(), R_env_np2
        )

    return mps


# two-site DMRG 主循环
def run_two_site_dmrg(mps, mpo, D_max, n_sweeps, verbose=True):

    L = len(mps)
    energies = []

    for sweep in range(n_sweeps):
        mps = two_site_update_L2R(mps, mpo, D_max)
        mps = two_site_update_R2L(mps, mpo, D_max)
        E = calculate_energy(mps, mpo).item()
        energies.append(E)

        if verbose:
            print(f"sweep {sweep+1:2d} : E/L = {E / L:.15f}")

    return mps, energies

# 1D TFIM exact diagonalization benchmark (gpt5写的)

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def build_tfim_hamiltonian(L, J=1.0, h=1.0):

    dim = 1 << L  # Hilbert 空间维度 = 2^L

    rows = []
    cols = []
    data = []

    for state in range(dim):
        # 对角项：-J σ^z_i σ^z_{i+1}
        diag_E = 0.0
        for i in range(L - 1):  # 开放边界，只到 L-2
            bit_i = (state >> i) & 1
            bit_j = (state >> (i + 1)) & 1
            s_i = 1 if bit_i == 1 else -1
            s_j = 1 if bit_j == 1 else -1
            diag_E += -J * s_i * s_j

        rows.append(state)
        cols.append(state)
        data.append(diag_E)

        # 非对角项：-h σ^x_i 翻转第 i 个自旋
        for i in range(L):
            flipped_state = state ^ (1 << i)  # 翻转第 i 位
            rows.append(flipped_state)
            cols.append(state)
            data.append(-h)

    # 构造稀疏矩阵并转换为 CSR 格式（适合 eigsh）
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    return H


def main():
    L = 16
    J = 1.0
    h = 1.0

    print(f"构造 L = {L}, J = {J}, h = {h} 的 1D TFIM 哈密顿量（开放边界条件）...")
    H = build_tfim_hamiltonian(L, J, h)

    print(f"Hilbert 空间维度 = 2^{L} = {H.shape[0]}")
    print("开始用稀疏对角化求最低若干本征能...")

    # 求最小的4个本征值
    k = 4
    vals, vecs = eigsh(H, k=k, which="SA")

    # 排序
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    print(f"\n最低 {k} 个本征能：")
    for i, E in enumerate(vals):
        print(f"E[{i}] = {E:.8f}")

    print(f"\n基态能量 E0 = {vals[0]:.8f}")
    print(f"基态能量密度 e0 = E0 / L = {vals[0] / L:.15f}")


if __name__ == "__main__":
    main()

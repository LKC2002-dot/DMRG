import torch
from mpo import build_tfim_mpo
from mps import obc_mps
from dmrg import run_two_site_dmrg, normalize_mps


if __name__ == "__main__":

    # 参数设置
    L = 16
    d = 2
    D = 10
    D_max = 20  # 最大截断维度
    J = 1.0
    h = 1.0
    dtype = torch.float64
    device = "cpu"
    n_sweeps = 6

    mps = obc_mps(L, d, D, dtype=dtype, device=device)
    mps = normalize_mps(mps)
    mpo = build_tfim_mpo(L, J, h, dtype=dtype, device=device)

    mps_gs, energies = run_two_site_dmrg(
        mps, mpo, D_max=D_max, n_sweeps=n_sweeps, verbose=True
    )

    E_final = energies[-1]
    print(f"\nDMRG final E/L = {E_final / L:.15f}")

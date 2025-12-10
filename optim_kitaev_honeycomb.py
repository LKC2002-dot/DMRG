import torch
import context
import config as cfg
from ipeps.ipeps import IPEPS

# 返回一个rank-5的随机张量
def rand_site(d, Du, Dl, Dd, Dr):

    return torch.randn(
        d, Du, Dl, Dd, Dr,
        dtype=cfg.global_args.torch_dtype,
        device=cfg.global_args.device
    )

def make_kitaev_honeycomb_brickwall_ipeps(D_active=2, d_phys=2):

    D = D_active
    d = d_phys
    Du_even, Dd_even = D, 1
    Du_odd,  Dd_odd  = 1, D

    # A (0,0): 偶数行
    A = rand_site(
        d,
        Du=Du_even,
        Dl=D,
        Dd=Dd_even,
        Dr=D
    )

    # B (1,0): 偶数行
    B = rand_site(
        d,
        Du=Du_odd,
        Dl=D,
        Dd=Dd_odd,
        Dr=D
    )

    # C (0,1): 奇数行
    C = rand_site(
        d,
        Du=Du_odd,
        Dl=D,
        Dd=Dd_odd,
        Dr=D
    )

    # D (1,1): 奇数行
    D_tensor = rand_site(
        d,
        Du=Du_even,
        Dl=D,
        Dd=Dd_even,
        Dr=D
    )

    # 2×2 unit cell: A, B, C, D
    sites = {
        (0, 0): A,        # A
        (1, 0): B,        # B
        (0, 1): C,        # C
        (1, 1): D_tensor  # D
    }

    state = IPEPS(
        sites=sites,
        lX=2,
        lY=2,
        peps_args=cfg.peps_args,
        global_args=cfg.global_args
    )

    return state


if __name__ == "__main__":

    torch.manual_seed(0)
    cfg.global_args.device = "cpu" 
    cfg.global_args.torch_dtype = torch.float64
    D = 10
    wfc = make_kitaev_honeycomb_brickwall_ipeps(D_active=D, d_phys=2)

    print(wfc)

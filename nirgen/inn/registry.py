"""Biophysical species and device-type registries."""

import math

RHO = 2.30
D_POOL_NM = 100_000.0
ION1 = math.floor(RHO * math.pi * D_POOL_NM**2 / 4)
PHI_T = 25.7
Z_ION = 1
NERNST_EPS_IONS = 1e-6
NERNST_EXP_CLIP = 20.0
INVENTORY_JITTER = 2.0
INVENTORY_JITTER_REL = 0.05
RTH_MIN, RQ_MIN, VTH_MIN, VQ_MIN = 5, 2, 5, 2

DEVICE_TYPES = {
    "Human": {"rth": 8, "rq": 4, "vth": 4, "vq": 4},
    "Mouse": {"rth": 8, "rq": 4, "vth": 4, "vq": 4},
    "Drosophila": {"rth": 4, "rq": 4, "vth": 2, "vq": 4},
    "C. elegans": {"rth": 4, "rq": 2, "vth": 2, "vq": 2},
}

SPECIES = {
    "Human": dict(d_endo=20_000.0, d_ecto=20_080.0, rho=RHO, f=0.05,
                   C_endo=723_000_000, C_ecto=5_800_000, S_Bendo_0=36_000_000,
                   S_Becto_0=290_000, D=36_290_000, delta=2e-3, gamma=8e-3,
                   alpha=0.5, beta=0.5, sigma=1.0, f_rth=0.10, f_rq=0.05,
                   f_vth=0.10, f_vq=0.05, two_cmp=True),
    "Mouse": dict(d_endo=15_000.0, d_ecto=15_040.0, rho=RHO, f=0.05,
                   C_endo=408_000_000, C_ecto=2_200_000, S_Bendo_0=20_000_000,
                   S_Becto_0=110_000, D=20_110_000, delta=1.1e-3, gamma=5.3e-3,
                   alpha=0.5, beta=0.5, sigma=1.0, f_rth=0.10, f_rq=0.05,
                   f_vth=0.10, f_vq=0.05, two_cmp=True),
    "Drosophila": dict(d_endo=4_000.0, d_ecto=4_040.0, rho=RHO, f=0.05,
                   C_endo=28_900_000, C_ecto=580_000, S_Bendo_0=1_500_000,
                   S_Becto_0=29_000, D=1_529_000, delta=8.2e-5, gamma=2e-2,
                   alpha=0.5, beta=0.5, sigma=1.0, f_rth=0.10, f_rq=0.05,
                   f_vth=0.10, f_vq=0.05, two_cmp=True),
    "C. elegans": dict(d_endo=2_850.0, d_ecto=2_890.0, rho=RHO, f=0.05,
                   C_endo=14_700_000, C_ecto=420_000, S_Bendo_0=730_000,
                   S_Becto_0=21_000, D=751_000, delta=4.2e-5, gamma=2.7e-2,
                   alpha=0.5, beta=0.5, sigma=1.0, f_rth=0.10, f_rq=0.05,
                   f_vth=0.10, f_vq=0.05, two_cmp=True),
}
CAL_MODES = ("pool", "fixed")
TYPE_PRESETS = {
    "paper": None,
    "low": {"rth": 2, "rq": 2, "vth": 2, "vq": 2},
    "mid": {"rth": 4, "rq": 4, "vth": 4, "vq": 4},
    "high": {"rth": 8, "rq": 8, "vth": 8, "vq": 8},
}


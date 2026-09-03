"""Core six-operation discrete-ion neural network."""

import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .mechanics import ste_floor
from .registry import (
    CAL_MODES, DEVICE_TYPES, ION1, INVENTORY_JITTER, INVENTORY_JITTER_REL,
    NERNST_EPS_IONS, NERNST_EXP_CLIP, PHI_T, RTH_MIN, RQ_MIN, SPECIES,
    VTH_MIN, VQ_MIN, Z_ION,
)


def equal_fracs(size):
    return np.full(size, 1.0 / size, dtype=np.float64)


def view_global(total, fractions):
    return total * fractions


def _norm_cal_mode(cal_mode):
    if cal_mode in ("pool", "patch"):
        return "pool"
    if cal_mode in ("fixed", "global"):
        return "fixed"
    raise ValueError("cal_mode must be 'pool'/'fixed' (aliases patch/global)")


def init_inn(species, n, m, ion1=ION1, w_in_fracs=None, w_out_fracs=None,
             cal_mode="fixed", type_preset=None):
    """Build the INN's immutable biophysical configuration."""
    cal_mode = _norm_cal_mode(cal_mode)
    sp = SPECIES[species]
    if not sp["two_cmp"]:
        raise ValueError(f"{species} is single-compartment.")
    wi = np.asarray(w_in_fracs if w_in_fracs is not None else equal_fracs(m), dtype=np.float64)
    wo = np.asarray(w_out_fracs if w_out_fracs is not None else equal_fracs(n), dtype=np.float64)
    if not np.isclose(wi.sum(), 1.0) or not np.isclose(wo.sum(), 1.0):
        raise ValueError("Input/output patch fractions must each sum to 1.")
    sli_e = view_global(sp["S_Becto_0"], wo); sli_n = view_global(sp["S_Bendo_0"], wo)
    slo_e = view_global(sp["S_Becto_0"], wi); slo_n = view_global(sp["S_Bendo_0"], wi)
    cli_e = view_global(sp["C_ecto"], wo); cli_n = view_global(sp["C_endo"], wo)
    clo_e = view_global(sp["C_ecto"], wi); clo_n = view_global(sp["C_endo"], wi)
    f_li_e = math.floor(sp["alpha"] * sli_e[0]); a_li_e = sli_e[0] - f_li_e
    f_lo_n = math.floor(sp["beta"] * slo_n[0]); a_lo_n = slo_n[0] - f_lo_n
    f_lo_e = math.floor(sp["beta"] * slo_e[0])
    if cal_mode == "pool":
        rth=max(RTH_MIN, math.floor(sp["f_rth"]*a_li_e)); rq=max(RQ_MIN, math.floor(sp["f_rq"]*f_li_e))
        vth=max(VTH_MIN, math.floor(sp["f_vth"]*a_lo_n)); vq=max(VQ_MIN, math.floor(sp["f_vq"]*f_lo_e))
        type_source="pool_fraction"
    else:
        labels = dict(DEVICE_TYPES[species] if type_preset is None else type_preset)
        rth=max(1,int(labels["rth"])); rq=max(1,int(labels["rq"]))
        vth=max(1,int(labels["vth"])); vq=max(1,int(labels["vq"]))
        type_source="fixed"
    r0=max(1.0, 0.5*a_li_e/rth); v0=max(1.0, 0.5*a_lo_n/vth)
    t2=int(a_li_e/rth); t3=int(sp["sigma"]*f_li_e/rq); tv=int(a_lo_n/vth)
    return dict(species=species,n=n,m=m,cal_mode=cal_mode,type_source=type_source,
        C_endo=sp["C_endo"],C_ecto=sp["C_ecto"],Cli_endo=cli_n,Cli_ecto=cli_e,
        Clo_endo=clo_n,Clo_ecto=clo_e,S_Bendo_0=sp["S_Bendo_0"],S_Becto_0=sp["S_Becto_0"],
        Sli_endo_0=sli_n,Sli_ecto_0=sli_e,Slo_endo_0=slo_n,Slo_ecto_0=slo_e,
        D=sp["D"],delta=sp["delta"],gamma=sp["gamma"],w_in=wi.astype(np.float32),w_out=wo.astype(np.float32),
        r0=r0,v0=v0,x_sf=float(rq)/sp["alpha"],y_sf=float(rq)/sp["alpha"],alpha=sp["alpha"],beta=sp["beta"],sigma=sp["sigma"],
        rth=float(rth),rq=float(rq),rq_abs=float(rq),vth=float(vth),vq=float(vq),
        f_rth=sp["f_rth"],f_rq=sp["f_rq"],f_vth=sp["f_vth"],f_vq=sp["f_vq"],
        A_li_ecto_anchor=float(a_li_e),F_li_ecto_anchor=float(f_li_e),A_lo_endo_anchor=float(a_lo_n),
        t2_rest=t2,t3_rest=t3,tv_rest=tv,feasible=bool(t2>=1 and t3>=1 and tv>=1),bind_r_rest="inventory")

@dataclass
class INNConfig:
    species: str
    n: int
    m: int
    readout: int = 1
    cal_mode: str = "fixed"

class INN(nn.Module):
    """Single-cell INN implementing the notebook's six-operation schedule."""
    def __init__(self, species, n, m, readout=1, device=None, w_in_fracs=None, w_out_fracs=None, cal_mode="fixed"):
        super().__init__(); device=device or torch.device("cpu")
        self.species,self.n,self.m,self.readout=species,n,m,readout; self.cal_mode=_norm_cal_mode(cal_mode)
        p=init_inn(species,n,m,w_in_fracs=w_in_fracs,w_out_fracs=w_out_fracs,cal_mode=self.cal_mode)
        def sb(v): return torch.as_tensor(float(v),dtype=torch.float32,device=device)
        def ba(v): return torch.as_tensor(v,dtype=torch.float32,device=device)
        for k in ("alpha","beta","sigma","rth","rq","rq_abs","vth","vq"): self.register_buffer(k,sb(p[k]))
        for k in ("Cli_ecto","Cli_endo","Clo_ecto","Clo_endo","Sli_ecto_0","Sli_endo_0","Slo_ecto_0","Slo_endo_0","w_in","w_out"): self.register_buffer(k,ba(p[k]))
        for k,v in (("x_sf",p["x_sf"]),("y_sf",p["y_sf"]),("phi_t",PHI_T),("z_ion",Z_ION)): self.register_buffer(k,sb(v))
        ambient=float(ION1)-float(p["D"]); self.register_buffer("ambient_total",sb(ambient)); self.register_buffer("pool_in",sb(ambient/n)); self.register_buffer("pool_out",sb(ambient/m)); self.register_buffer("S_Becto_0_scalar",sb(p["S_Becto_0"]))
        r0,v0=p["r0"],p["v0"]; jr=max(INVENTORY_JITTER,INVENTORY_JITTER_REL*r0); jv=max(INVENTORY_JITTER,INVENTORY_JITTER_REL*v0)
        self.r_tilde=nn.Parameter(torch.empty(n,device=device).uniform_(max(.5,r0-jr),r0+jr)); self.v_tilde=nn.Parameter(torch.empty(m,device=device).uniform_(max(.5,v0-jv),v0+jv))
        self.register_buffer("y_rest",torch.zeros(m,device=device)); self.register_buffer("r2_scale",torch.ones((),device=device)); self._refresh_rest_calibration()
    def r_int(self): return torch.clamp(ste_floor(self.r_tilde),min=0).round()
    def v_int(self): return torch.clamp(ste_floor(self.v_tilde),min=0).round()
    def _state(self,B):
        exp=lambda t:t.unsqueeze(0).expand(B,-1).clone()
        return exp(self.Sli_ecto_0),exp(self.Sli_endo_0),torch.zeros(B,self.m,device=self.pool_in.device),torch.zeros(B,self.m,device=self.pool_in.device),torch.zeros(B,self.n,device=self.pool_in.device),torch.zeros(B,self.m,device=self.pool_in.device)
    def _split(self,S,c): F=ste_floor(c*S); return F,S-F
    def _o0_voltage(self,S,x):
        V0=self.phi_t/self.z_ion*torch.log(torch.clamp(self.pool_in/torch.clamp(S,min=NERNST_EPS_IONS),min=NERNST_EPS_IONS)); e=torch.clamp(-self.z_ion*(V0+x)/self.phi_t,min=-NERNST_EXP_CLIP,max=NERNST_EXP_CLIP); return torch.minimum(torch.clamp(ste_floor(self.pool_in*torch.exp(e)),min=1),self.Cli_ecto.unsqueeze(0))
    def _o2(self,e,n): return (*self._split(e,self.alpha),*self._split(n,self.beta))
    def _o3(self,e,n,b,Fe,Ae,Fn,An):
        r=self.r_int().unsqueeze(0).expand_as(e); t2=ste_floor(Ae/torch.clamp(self.rth,min=1)); t3=ste_floor((self.sigma*Fe+(1-self.sigma)*Fn)/torch.clamp(self.rq_abs,min=1)); hw=torch.clamp((1-self.sigma)*(self.Cli_ecto-e)+self.sigma*(self.Cli_endo-n),min=0); t4=ste_floor(hw/torch.clamp(self.rq_abs,min=1)); re=torch.clamp(torch.minimum(torch.minimum(r,t2),torch.minimum(t3,t4)),min=0); b=b+re*self.rth; e=torch.clamp(e-re*self.rq-re*self.rth,min=0); n=torch.clamp(n+re*self.rq,min=0); return e,n,b,*self._o2(e,n)
    def _o4(self,e,n,oe,on): d_e=n.sum(1,keepdim=True)*self.w_in.unsqueeze(0); d_n=e.sum(1,keepdim=True)*self.w_in.unsqueeze(0); return torch.clamp(d_n,min=0),torch.clamp(d_e,min=0),torch.zeros_like(e),torch.zeros_like(n)
    def _o5(self,n): return self._split(n,self.beta)
    def _o6(self,e,n,b,Fn,An): v=self.v_int().unsqueeze(0).expand_as(n); t2=ste_floor(An/torch.clamp(self.vth,min=1)); hw=torch.clamp(self.Clo_ecto-e,min=0); t3=ste_floor(torch.minimum(Fn,hw)/torch.clamp(self.vq,min=1)); ve=torch.clamp(torch.minimum(torch.minimum(v,t2),t3),min=0); e=torch.clamp(e+ve*self.vq,min=0); n=torch.clamp(n-ve*self.vq-ve*self.vth,min=0); b=b+ve*self.vth; Fn,An=self._o5(n); return e,n,b,Fn,An
    def _core(self,x):
        x=x.reshape(-1,self.n).to(device=self.pool_in.device,dtype=torch.float32); B=x.shape[0]; e,n,oe,on,ib,ob=self._state(B); e=self._o0_voltage(e,x); Fe,Ae,Fn,An=self._o2(e,n); e,n,ib,Fe,Ae,Fn,An=self._o3(e,n,ib,Fe,Ae,Fn,An); oe,on,_,_=self._o4(e,n,oe,on); Fn,An=self._o5(on); oe,on,ob,_,_=self._o6(oe,on,ob,Fn,An); return ste_floor(oe).round(),on
    def _nernst(self,e): return self.phi_t/self.z_ion*torch.log(torch.clamp(self.pool_out/torch.clamp(e,min=NERNST_EPS_IONS),min=NERNST_EPS_IONS))
    def _raw_readout(self,e,n): return e-n if int(self.readout)==2 else self._nernst(e)
    @torch.no_grad()
    def _refresh_rest_calibration(self):
        e,n=self._core(torch.zeros(1,self.n,device=self.pool_in.device)); self.y_rest.copy_(self._raw_readout(e,n).squeeze(0)); self.r2_scale.copy_(torch.clamp(self.y_rest.abs().mean(),min=1))
    def forward(self,x):
        e,n=self._core(x); y=self._raw_readout(e,n)-self.y_rest.unsqueeze(0); return y/self.r2_scale if int(self.readout)==2 else y
    def copy_inventories_from(self,other):
        with torch.no_grad(): self.r_tilde.copy_(other.r_tilde); self.v_tilde.copy_(other.v_tilde); self.y_rest.copy_(other.y_rest); self.r2_scale.copy_(other.r2_scale)
    def reg(self,lam_r,lam_v): return lam_r*self.r_int().sum()+lam_v*self.v_int().sum()

    @torch.no_grad()
    def conservation_check(self, x):
        """Return total ion counts after each operation for the notebook audit."""
        x=x.reshape(-1,self.n).to(device=self.pool_in.device,dtype=torch.float32); B=x.shape[0]
        total=lambda *parts: float(sum(part.sum(1) for part in parts).mean())
        e,n,oe,on,ib,ob=self._state(B); log={"t0_init":total(e,n,oe,on,ib,ob)}
        e=self._o0_voltage(e,x); log["t1_after_o0_voltage"]=total(e,n,oe,on,ib,ob)
        Fe,Ae,Fn,An=self._o2(e,n); log["t2_after_o2"]=total(e,n,oe,on,ib,ob)
        e,n,ib,Fe,Ae,Fn,An=self._o3(e,n,ib,Fe,Ae,Fn,An); log["t3_after_o3"]=total(e,n,oe,on,ib,ob)
        oe,on,_,_=self._o4(e,n,oe,on); e=torch.zeros_like(e); n=torch.zeros_like(n); log["t4_after_o4"]=total(oe,on,e,n,ib,ob)
        Fn,An=self._o5(on); log["t5_after_o5"]=total(oe,on,e,n,ib,ob)
        oe,on,ob,Fn,An=self._o6(oe,on,ob,Fn,An); log["t6_after_o6"]=total(oe,on,e,n,ib,ob)
        return log


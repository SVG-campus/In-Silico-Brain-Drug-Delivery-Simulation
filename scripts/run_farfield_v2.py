#!/usr/bin/env python3
import os, math, json, argparse, datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Callable

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------- Configs
@dataclass
class DiffusionConfig:
    nx:int=64; ny:int=64; Lx:float=0.032; Ly:float=0.032
    dt:float=10.0; t_final:float=48*3600.0; save_dt:float=300.0
    bctype:str="neumann"
    D_white:float=5e-10; D_grey:float=1.0e-9; k_decay:float=1e-4
    src_x_frac:float=0.25; src_y_frac:float=0.50
    tgt_x_frac:float=0.75; tgt_y_frac:float=0.50
    source_radius_frac:float=0.02; target_radius_frac:float=0.10
    # dosing
    point_injection:bool=False
    dose_amount:float=1.0
    infusion_rate:float=2e-3         # v2: lower rate
    infusion_t_on:float=0.0
    infusion_t_off:float=12*3600.0   # v2: longer duration
    ramp_first_hours:float=2.0       # 0 = no ramp
    therapeutic_thresh:float=0.01; toxicity_thresh:float=0.5

@dataclass
class PKPDConfig:
    three_compartment:bool=True
    k_blood_to_brain:float=8e-4; k_brain_to_blood:float=5e-4
    k_elim_blood:float=3e-4; k_deg_brain:float=2e-4; k_blood_to_clear:float=0.0
    V_blood:float=5.0; V_brain:float=1.2; V_clear:float=2.0
    boluses:List[Tuple[float,float]]=None
    infusions:List[Tuple[float,float,float]]=None
    t_final:float=24*3600.0
    EC50:float=0.2; Emax:float=1.0; hill_n:float=2.0
    therapeutic_thresh:float=0.05; toxicity_thresh:float=1.0

def default_configs():
    diff = DiffusionConfig()
    pk = PKPDConfig(
        boluses=[(0.0,0.6),(6*3600.0,0.6)],
        infusions=[(2*3600.0,12*3600.0,2e-4)]  # v2: longer infusion
    )
    return diff, pk

# ------------- Geometry / helpers
def make_masks(nx,ny,Lx,Ly,src_r,tgt_r,sxf,syf,txf,tyf):
    x=np.linspace(0,Lx,nx); y=np.linspace(0,Ly,ny); X,Y=np.meshgrid(x,y,indexing="ij")
    cx,cy=Lx/2,Ly/2; a,b=0.35*Lx,0.35*Ly
    gm=((X-cx)**2/a**2+(Y-cy)**2/b**2)<=1.0; wm=~gm
    sx,sy=sxf*Lx,syf*Ly; tx,ty=txf*Lx,tyf*Ly; diag=math.hypot(Lx,Ly)
    r_src=src_r*diag; r_tgt=tgt_r*diag
    src=(X-sx)**2+(Y-sy)**2<=r_src**2; tgt=(X-tx)**2+(Y-ty)**2<=r_tgt**2
    return gm,wm,src,tgt

def Dmap(gm,wm,Dg,Dw): return np.where(gm,Dg,Dw).astype(np.float64)

def div_D_grad(C,D,dx,dy,bctype="neumann"):
    if bctype=="neumann":
        Cpad=np.pad(C,((1,1),(1,1)),mode="edge"); Dpad=np.pad(D,((1,1),(1,1)),mode="edge")
    else:
        Cpad=np.pad(C,((1,1),(1,1)),mode="constant"); Dpad=np.pad(D,((1,1),(1,1)),mode="edge")
    Dxp=0.5*(Dpad[2:,1:-1]+Dpad[1:-1,1:-1]); Dxm=0.5*(Dpad[1:-1,1:-1]+Dpad[0:-2,1:-1])
    Dyp=0.5*(Dpad[1:-1,2:]+Dpad[1:-1,1:-1]); Dym=0.5*(Dpad[1:-1,1:-1]+Dpad[1:-1,0:-2])
    dCxp=(Cpad[2:,1:-1]-Cpad[1:-1,1:-1]); dCxm=(Cpad[1:-1,1:-1]-Cpad[0:-2,1:-1])
    dCyp=(Cpad[1:-1,2:]-Cpad[1:-1,1:-1]); dCym=(Cpad[1:-1,1:-1]-Cpad[1:-1,0:-2])
    return (Dxp*dCxp-Dxm*dCxm)/dx**2+(Dyp*dCyp-Dym*dCym)/dy**2

def dose_shape(t, rate, t_on, t_off, ramp_h):
    if not (t_on <= t <= t_off): return 0.0
    if ramp_h <= 0: return rate
    ramp_s = ramp_h*3600.0
    if t - t_on <= ramp_s:
        return rate * (t - t_on) / ramp_s
    return rate

# ------------- Diffusion (checkpoint + KPIs)
def run_diffusion(diff:DiffusionConfig,outdir:str,ckpt:str):
    os.makedirs(outdir,exist_ok=True)
    nx,ny=diff.nx,diff.ny; dx,dy=diff.Lx/(nx-1),diff.Ly/(ny-1)
    gm,wm,src,tgt=make_masks(nx,ny,diff.Lx,diff.Ly,diff.source_radius_frac,diff.target_radius_frac,
                             diff.src_x_frac,diff.src_y_frac,diff.tgt_x_frac,diff.tgt_y_frac)
    D=Dmap(gm,wm,diff.D_grey,diff.D_white)
    dt_stab=0.2*min(dx,dy)**2/(4.0*D.max()+1e-30); dt=min(diff.dt,dt_stab)

    # resume if ckpt present
    C=np.zeros((nx,ny)); t0=0.0
    times=[]; avg=[]; cov=[]; od=[]; mass=[]
    if os.path.exists(ckpt):
        s=np.load(ckpt,allow_pickle=True).item()
        C=s["C"]; t0=s["t"]; times=s["times"]; avg=s["avg"]; cov=s["cov"]; od=s["od"]; mass=s["mass"]
        print(f"[resume] t={t0/3600:.2f} h")

    if diff.point_injection and C.sum()==0 and src.any():
        C[src]+=diff.dose_amount

    nsteps=int(np.ceil(diff.t_final/dt)); save_every=max(1,int(round(diff.save_dt/dt)))
    first_cover_t=None; last_cover_t=None

    for step in range(int(t0/dt), nsteps+1):
        t=step*dt
        if step%save_every==0 or step==nsteps:
            tgt_vals=C[tgt]
            times.append(t); avg.append(float(tgt_vals.mean()))
            c=float((tgt_vals>=diff.therapeutic_thresh).mean()); o=float((C>=diff.toxicity_thresh).mean())
            cov.append(c); od.append(o); mass.append(float(C.mean()))
            # track first/last crossing of coverage>0
            if c>0 and first_cover_t is None: first_cover_t=t
            if c>0: last_cover_t=t
            pd.DataFrame({"time_s":times,"avg_target":avg,"coverage_target":cov,
                          "frac_overdose":od,"total_mass_proxy":mass}).to_csv(
                os.path.join(outdir,"diffusion_timeseries.csv"), index=False)
            np.save(ckpt,{"C":C,"t":t,"times":times,"avg":avg,"cov":cov,"od":od,"mass":mass})

        if step==nsteps: break
        S = 0.0 if diff.point_injection else dose_shape(t, diff.infusion_rate, diff.infusion_t_on, diff.infusion_t_off, diff.ramp_first_hours)
        Cn = C + dt*(div_D_grad(C,D,dx,dy,diff.bctype) - diff.k_decay*C)
        if S!=0.0: Cn[src]+=S*dt
        C=np.maximum(Cn,0.0)

    # plots
    df=pd.read_csv(os.path.join(outdir,"diffusion_timeseries.csv"))
    plt.figure(); 
    plt.plot(df["time_s"]/3600.0,df["avg_target"],label="Avg target")
    plt.plot(df["time_s"]/3600.0,df["coverage_target"],label="Coverage ≥ thr")
    plt.plot(df["time_s"]/3600.0,df["frac_overdose"],label="Overdose frac")
    plt.xlabel("Time (h)"); plt.legend(); plt.title("Diffusion metrics (far-field v2)")
    plt.savefig(os.path.join(outdir,"diffusion_metrics_timeseries.png"),bbox_inches="tight"); plt.close()

    # KPIs
    t=np.array(df["time_s"]); dt_arr=np.diff(t,prepend=t[0])
    coverage=np.array(df["coverage_target"]); overdose=np.array(df["frac_overdose"])
    kpis = {
        "t_end_h": float(t[-1]/3600.0),
        "coverage_time_h": float(np.sum((coverage>0)*dt_arr)/3600.0),
        "coverage_auc_hxfraction": float(np.trapz(coverage,t)/3600.0),
        "max_coverage_fraction": float(np.max(coverage)),
        "max_overdose_fraction": float(np.max(overdose)),
        "first_coverage_h": None if first_cover_t is None else float(first_cover_t/3600.0),
        "last_coverage_h":  None if last_cover_t  is None else float(last_cover_t/3600.0),
    }
    return kpis

# ------------- PK/PD + PD
def make_dose_fn(bol,inf):
    bol=bol or []; inf=inf or []
    def dose(t):
        r=0.0
        for tb,amt in bol:
            if tb<=t<tb+1.0: r+=amt
        for a,b,rt in inf:
            if a<=t<=b: r+=rt
        return r
    return dose

def pkpd_rhs(t,y,p,dose):
    Cb,Cbr=y[0],y[1]; u=dose(t)
    dCb=-(p.k_blood_to_brain+p.k_elim_blood+(p.k_blood_to_clear if p.three_compartment else 0.0))*Cb + p.k_brain_to_blood*Cbr + u/max(p.V_blood,1e-12)
    dCbr=p.k_blood_to_brain*Cb-(p.k_brain_to_blood+p.k_deg_brain)*Cbr
    if p.three_compartment:
        Cl=y[2]; dCl=p.k_blood_to_clear*Cb; return [dCb,dCbr,dCl]
    return [dCb,dCbr]

def run_pkpd(pk:PKPDConfig,outdir:str):
    os.makedirs(outdir,exist_ok=True)
    y0=[0.0,0.0] if not pk.three_compartment else [0.0,0.0,0.0]
    t_eval=np.linspace(0,pk.t_final,1400)
    rhs=lambda t,y: pkpd_rhs(t,y,pk,make_dose_fn(pk.boluses,pk.infusions))
    sol=solve_ivp(rhs,[0,pk.t_final],y0,t_eval=t_eval,rtol=1e-6,atol=1e-9)
    Cb, Cbr = sol.y[0], sol.y[1]
    E = pk.Emax*(Cbr**pk.hill_n)/(pk.EC50**pk.hill_n + Cbr**pk.hill_n)

    pd.DataFrame({"time_s":sol.t,"Cb":Cb,"Cbrain":Cbr,"Effect":E}).to_csv(
        os.path.join(outdir,"pkpd_timeseries.csv"), index=False)

    plt.figure(); plt.plot(sol.t/3600.0,Cb,label="Blood"); plt.plot(sol.t/3600.0,Cbr,label="Brain")
    plt.axhline(pk.therapeutic_thresh,ls="--",lw=0.7,label="Therapeutic"); plt.axhline(pk.toxicity_thresh,ls="--",lw=0.7,label="Toxicity")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration"); plt.legend(); plt.title("PK across compartments (v2)")
    plt.savefig(os.path.join(outdir,"pkpd_compartments.png"),bbox_inches="tight"); plt.close()
    plt.figure(); plt.plot(sol.t/3600.0,E); plt.xlabel("Time (h)"); plt.ylabel("Effect"); plt.title("PD effect (v2)")
    plt.savefig(os.path.join(outdir,"pkpd_effect.png"),bbox_inches="tight"); plt.close()

    dt=np.diff(sol.t,prepend=sol.t[0])
    return {
        "t_end_h": float(sol.t[-1]/3600.0),
        "time_in_therapeutic_h": float(np.sum((Cbr>=pk.therapeutic_thresh)*dt)/3600.0),
        "frac_time_in_therapeutic": float(np.sum((Cbr>=pk.therapeutic_thresh)*dt)/(sol.t[-1]-sol.t[0])),
        "time_above_toxic_h": float(np.sum((Cbr>=pk.toxicity_thresh)*dt)/3600.0),
        "mean_Cbrain": float(np.trapz(Cbr,sol.t)/(sol.t[-1]-sol.t[0])),
        "max_Cbrain": float(np.max(Cbr)),
    }

# ------------- CLI + README
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir",default=None,help="Output folder")
    ap.add_argument("--resume",action="store_true",help="Resume diffusion from checkpoint if present")
    args=ap.parse_args()

    ts=datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    outdir = args.outdir or f"outputs/farfield_v2_{ts}"
    os.makedirs(outdir,exist_ok=True)
    ckpt=os.path.join(outdir,"diffusion_ckpt.npy")

    diff, pk = default_configs()
    if not args.resume and os.path.exists(ckpt): os.remove(ckpt)

    diff_kpis = run_diffusion(diff, outdir, ckpt)
    pk_kpis   = run_pkpd(pk, outdir)

    summary = {"diffusion": diff_kpis, "pkpd": pk_kpis,
               "config": {"diffusion": asdict(diff), "pkpd": asdict(pk)}}
    with open(os.path.join(outdir,"run_summary.json"),"w") as f: json.dump(summary,f,indent=2)

    # lightweight README for provenance
    with open(os.path.join(outdir,"README.md"),"w") as f:
        f.write(f"# Far-field v2 run\n\n")
        f.write(f"- Date (UTC): {ts}\n- Outdir: `{outdir}`\n\n")
        f.write("## KPIs\n")
        for k,v in {**{f'diffusion.{k}':v for k,v in diff_kpis.items()},
                    **{f'pkpd.{k}':v for k,v in pk_kpis.items()}}.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Config\n")
        f.write("```json\n"+json.dumps({"diffusion":asdict(diff),"pkpd":asdict(pk)},indent=2)+"\n```\n")
    print("All done. Files in:", outdir)

if __name__=="__main__":
    main()

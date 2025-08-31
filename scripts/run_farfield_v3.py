# scripts/run_farfield_v3.py
#!/usr/bin/env python3
import os, math, json, argparse, datetime, itertools, random
from dataclasses import dataclass, asdict, replace
from typing import List, Tuple, Dict, Any

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------- Configs
@dataclass
class DiffusionConfig:
    # domain & numerics
    nx:int=64; ny:int=64; Lx:float=0.032; Ly:float=0.032
    dt:float=10.0; t_final:float=48*3600.0; save_dt:float=300.0
    bctype:str="neumann"
    # tissue coefficients (base, isotropic)
    D_white:float=5e-10; D_grey:float=1.0e-9; k_decay:float=1e-4
    # anisotropy: Dx = D_base; Dy = D_base * aniso_ratio
    aniso_ratio_white:float=0.6   # Dy lower than Dx in white matter (fiber bias)
    aniso_ratio_grey:float=1.0    # nearly isotropic
    # geometry
    src_x_frac:float=0.25; src_y_frac:float=0.50
    tgt_x_frac:float=0.75; tgt_y_frac:float=0.50
    source_radius_frac:float=0.02; target_radius_frac:float=0.10
    # dosing
    point_injection:bool=False; dose_amount:float=1.0
    infusion_rate:float=2e-3; infusion_t_on:float=0.0; infusion_t_off:float=12*3600.0
    ramp_first_hours:float=2.0
    # thresholds
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
        infusions=[(2*3600.0,12*3600.0,2e-4)]
    )
    return diff, pk

# -------------------------- Geometry & helpers
def make_masks(nx,ny,Lx,Ly,src_r,tgt_r,sxf,syf,txf,tyf):
    x=np.linspace(0,Lx,nx); y=np.linspace(0,Ly,ny); X,Y=np.meshgrid(x,y,indexing="ij")
    cx,cy=Lx/2,Ly/2; a,b=0.35*Lx,0.35*Ly
    gm=((X-cx)**2/a**2+(Y-cy)**2/b**2)<=1.0; wm=~gm
    sx,sy=sxf*Lx,syf*Ly; tx,ty=txf*Lx,tyf*Ly; diag=math.hypot(Lx,Ly)
    r_src=src_r*diag; r_tgt=tgt_r*diag
    src=(X-sx)**2+(Y-sy)**2<=r_src**2; tgt=(X-tx)**2+(Y-ty)**2<=r_tgt**2
    return X,Y,gm,wm,src,tgt

def dose_shape(t, rate, t_on, t_off, ramp_h):
    if not (t_on <= t <= t_off): return 0.0
    if ramp_h <= 0: return rate
    ramp_s = ramp_h*3600.0
    if t - t_on <= ramp_s: return rate * (t - t_on) / max(ramp_s,1e-9)
    return rate

# anisotropic diffusion operator: separate Dx, Dy maps
def div_Dx_Dy_grad(C, Dx, Dy, dx, dy, bctype="neumann"):
    if bctype=="neumann":
        Cpad=np.pad(C,((1,1),(1,1)),mode="edge")
        Dxpad=np.pad(Dx,((1,1),(1,1)),mode="edge")
        Dypad=np.pad(Dy,((1,1),(1,1)),mode="edge")
    else:
        Cpad=np.pad(C,((1,1),(1,1)),mode="constant")
        Dxpad=np.pad(Dx,((1,1),(1,1)),mode="edge")
        Dypad=np.pad(Dy,((1,1),(1,1)),mode="edge")
    Dxp=0.5*(Dxpad[2:,1:-1]+Dxpad[1:-1,1:-1]); Dxm=0.5*(Dxpad[1:-1,1:-1]+Dxpad[0:-2,1:-1])
    Dyp=0.5*(Dypad[1:-1,2:]+Dypad[1:-1,1:-1]); Dym=0.5*(Dypad[1:-1,1:-1]+Dypad[1:-1,0:-2])
    dCxp=(Cpad[2:,1:-1]-Cpad[1:-1,1:-1]); dCxm=(Cpad[1:-1,1:-1]-Cpad[0:-2,1:-1])
    dCyp=(Cpad[1:-1,2:]-Cpad[1:-1,1:-1]); dCym=(Cpad[1:-1,1:-1]-Cpad[1:-1,0:-2])
    return (Dxp*dCxp-Dxm*dCxm)/dx**2 + (Dyp*dCyp-Dym*dCym)/dy**2

# -------------------------- Diffusion runner (with anisotropy)
def run_diffusion(diff:DiffusionConfig, outdir:str, ckpt:str) -> Dict[str,Any]:
    os.makedirs(outdir,exist_ok=True)
    nx,ny=diff.nx,diff.ny; dx,dy=diff.Lx/(nx-1),diff.Ly/(ny-1)
    X,Y,gm,wm,src,tgt=make_masks(nx,ny,diff.Lx,diff.Ly,
                                 diff.source_radius_frac,diff.target_radius_frac,
                                 diff.src_x_frac,diff.src_y_frac,diff.tgt_x_frac,diff.tgt_y_frac)
    Dg, Dw = diff.D_grey, diff.D_white
    # build anisotropic maps
    Dx = np.where(gm, Dg, Dw)
    Dy = np.where(gm, Dg*diff.aniso_ratio_grey, Dw*diff.aniso_ratio_white)

    # stability via max of Dx,Dy
    Dmax = max(Dx.max(), Dy.max())
    dt_stab = 0.2*min(dx,dy)**2/(4.0*Dmax+1e-30)
    dt = min(diff.dt, dt_stab)

    # resume
    C=np.zeros((nx,ny)); t0=0.0
    times=[]; avg=[]; cov=[]; od=[]; mass=[]
    if os.path.exists(ckpt):
        s=np.load(ckpt,allow_pickle=True).item()
        C=s["C"]; t0=s["t"]; times=s["times"]; avg=s["avg"]; cov=s["cov"]; od=s["od"]; mass=s["mass"]
        print(f"[resume] t={t0/3600:.2f} h")

    if diff.point_injection and C.sum()==0 and src.any():
        C[src]+=diff.dose_amount

    nsteps=int(np.ceil(diff.t_final/dt))
    save_every=max(1,int(round(diff.save_dt/dt)))
    first_cov=None; last_cov=None

    for step in range(int(t0/dt), nsteps+1):
        t=step*dt
        if step%save_every==0 or step==nsteps:
            tgt_vals=C[tgt]
            c=float((tgt_vals>=diff.therapeutic_thresh).mean())
            o=float((C>=diff.toxicity_thresh).mean())
            times.append(t); avg.append(float(tgt_vals.mean())); cov.append(c); od.append(o); mass.append(float(C.mean()))
            if c>0 and first_cov is None: first_cov=t
            if c>0: last_cov=t
            pd.DataFrame({"time_s":times,"avg_target":avg,"coverage_target":cov,
                          "frac_overdose":od,"total_mass_proxy":mass}).to_csv(
                os.path.join(outdir,"diffusion_timeseries.csv"), index=False)
            np.save(ckpt,{"C":C,"t":t,"times":times,"avg":avg,"cov":cov,"od":od,"mass":mass})
        if step==nsteps: break

        S = 0.0 if diff.point_injection else dose_shape(t, diff.infusion_rate, diff.infusion_t_on, diff.infusion_t_off, diff.ramp_first_hours)
        Cn = C + dt*(div_Dx_Dy_grad(C,Dx,Dy,dx,dy,diff.bctype) - diff.k_decay*C)
        if S!=0.0: Cn[src]+=S*dt
        C=np.maximum(Cn,0.0)

    # plots + KPIs
    df=pd.read_csv(os.path.join(outdir,"diffusion_timeseries.csv"))
    plt.figure()
    plt.plot(df["time_s"]/3600.0,df["avg_target"],label="Avg target")
    plt.plot(df["time_s"]/3600.0,df["coverage_target"],label="Coverage ≥ thr")
    plt.plot(df["time_s"]/3600.0,df["frac_overdose"],label="Overdose frac")
    plt.xlabel("Time (h)"); plt.legend(); plt.title("Diffusion metrics (far-field v3)")
    plt.savefig(os.path.join(outdir,"diffusion_metrics_timeseries.png"),bbox_inches="tight"); plt.close()

    t=df["time_s"].to_numpy(); dt_arr=np.diff(t,prepend=t[0])
    coverage=df["coverage_target"].to_numpy(); overdose=df["frac_overdose"].to_numpy()
    kpis = {
        "t_end_h": float(t[-1]/3600.0),
        "coverage_time_h": float(np.sum((coverage>0)*dt_arr)/3600.0),
        "coverage_auc_hxfraction": float(np.trapezoid(coverage,t)/3600.0),
        "max_coverage_fraction": float(np.max(coverage)),
        "max_overdose_fraction": float(np.max(overdose)),
        "first_coverage_h": None if first_cov is None else float(first_cov/3600.0),
        "last_coverage_h":  None if last_cov  is None else float(last_cov/3600.0),
    }
    return kpis

# -------------------------- PK/PD
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

def run_pkpd(pk:PKPDConfig,outdir:str)->Dict[str,Any]:
    os.makedirs(outdir,exist_ok=True)
    y0=[0.0,0.0] if not pk.three_compartment else [0.0,0.0,0.0]
    t_eval=np.linspace(0,pk.t_final,1400)
    rhs=lambda t,y: pkpd_rhs(t,y,pk,make_dose_fn(pk.boluses,pk.infusions))
    sol=solve_ivp(rhs,[0,pk.t_final],y0,t_eval=t_eval,rtol=1e-6,atol=1e-9)
    Cb, Cbr = sol.y[0], sol.y[1]
    E = pk.Emax*(Cbr**pk.hill_n)/(pk.EC50**pk.hill_n + Cbr**pk.hill_n)
    pd.DataFrame({"time_s":sol.t,"Cb":Cb,"Cbrain":Cbr,"Effect":E}).to_csv(os.path.join(outdir,"pkpd_timeseries.csv"), index=False)
    plt.figure(); plt.plot(sol.t/3600.0,Cb,label="Blood"); plt.plot(sol.t/3600.0,Cbr,label="Brain")
    plt.axhline(pk.therapeutic_thresh,ls="--",lw=0.7,label="Therapeutic"); plt.axhline(pk.toxicity_thresh,ls="--",lw=0.7,label="Toxicity")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration"); plt.legend(); plt.title("PK across compartments (v3)")
    plt.savefig(os.path.join(outdir,"pkpd_compartments.png"),bbox_inches="tight"); plt.close()
    plt.figure(); plt.plot(sol.t/3600.0,E); plt.xlabel("Time (h)"); plt.ylabel("Effect"); plt.title("PD effect (v3)")
    plt.savefig(os.path.join(outdir,"pkpd_effect.png"),bbox_inches="tight"); plt.close()
    dt=np.diff(sol.t,prepend=sol.t[0])
    return {
        "t_end_h": float(sol.t[-1]/3600.0),
        "time_in_therapeutic_h": float(np.sum((Cbr>=pk.therapeutic_thresh)*dt)/3600.0),
        "frac_time_in_therapeutic": float(np.sum((Cbr>=pk.therapeutic_thresh)*dt)/(sol.t[-1]-sol.t[0])),
        "time_above_toxic_h": float(np.sum((Cbr>=pk.toxicity_thresh)*dt)/3600.0),
        "mean_Cbrain": float(np.trapezoid(Cbr,sol.t)/(sol.t[-1]-sol.t[0])),
        "max_Cbrain": float(np.max(Cbr)),
    }

# -------------------------- Grid search & robustness
def run_trial(outdir_base:str, name:str, diff:DiffusionConfig, pk:PKPDConfig)->Dict[str,Any]:
    outdir=os.path.join(outdir_base, name); os.makedirs(outdir,exist_ok=True)
    ckpt=os.path.join(outdir,"diffusion_ckpt.npy")
    # ensure fresh
    if os.path.exists(ckpt): os.remove(ckpt)
    diff_kpi=run_diffusion(diff,outdir,ckpt); pk_kpi=run_pkpd(pk,outdir)
    summary={"diffusion":diff_kpi,"pkpd":pk_kpi,"config":{"diffusion":asdict(diff),"pkpd":asdict(pk)}}
    with open(os.path.join(outdir,"run_summary.json"),"w") as f: json.dump(summary,f,indent=2)
    return {"name":name, **{f"diff_{k}":v for k,v in diff_kpi.items()}, **{f"pk_{k}":v for k,v in pk_kpi.items()}}

def small_grid_search(outdir:str, base_diff:DiffusionConfig, base_pk:PKPDConfig)->pd.DataFrame:
    rates=[1.5e-3, 2.0e-3, 2.5e-3]
    durations_h=[8, 12, 16]
    ramps_h=[1.0, 2.0]
    rows=[]
    for r,dh,rh in itertools.product(rates,durations_h,ramps_h):
        diff=replace(base_diff, infusion_rate=r, infusion_t_off=dh*3600.0, ramp_first_hours=rh)
        name=f"grid_rate{r:.1e}_dur{dh}h_ramp{int(rh)}h"
        rows.append(run_trial(outdir, name, diff, base_pk))
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir,"grid_search_results.csv"),index=False)
    # tradeoff plot: coverage_time vs overdose
    plt.figure()
    plt.scatter(df["diff_coverage_time_h"], df["diff_max_overdose_fraction"])
    plt.xlabel("Coverage time (h)"); plt.ylabel("Max overdose fraction")
    plt.title("Diffusion trade-off (v3 grid)")
    for _,r in df.iterrows():
        plt.annotate(r["name"].split("grid_")[-1], (r["diff_coverage_time_h"], r["diff_max_overdose_fraction"]), fontsize=7, alpha=0.6)
    plt.savefig(os.path.join(outdir,"grid_tradeoff.png"), bbox_inches="tight"); plt.close()
    return df

def robustness_sweep(outdir:str, base_diff:DiffusionConfig, base_pk:PKPDConfig, n:int=20, seed:int=0)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n):
        # lognormal-ish multiplicative noise: D ±30%, k ×[0.5,2] approx
        mD=np.exp(rng.normal(0, 0.3))     # ~±30% sigma
        mk=np.exp(rng.normal(0, 0.35))    # broader
        diff=replace(base_diff, D_grey=base_diff.D_grey*mD, D_white=base_diff.D_white*mD, k_decay=base_diff.k_decay*mk)
        name=f"robust_{i:02d}"
        rows.append(run_trial(outdir, name, diff, base_pk))
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir,"robustness_results.csv"),index=False)
    # robustness KPI: P(coverage_time_h >= 8h) and overdose risk
    p_cov=float((df["diff_coverage_time_h"]>=8).mean())
    mean_over=float(df["diff_max_overdose_fraction"].mean())
    with open(os.path.join(outdir,"robustness_summary.json"),"w") as f:
        json.dump({"P_coverage_time_ge_8h":p_cov, "mean_max_overdose":mean_over}, f, indent=2)
    plt.figure()
    plt.scatter(df["diff_coverage_time_h"], df["diff_max_overdose_fraction"])
    plt.xlabel("Coverage time (h)"); plt.ylabel("Max overdose fraction"); plt.title("Robustness sweep (v3)")
    plt.savefig(os.path.join(outdir,"robustness_scatter.png"), bbox_inches="tight"); plt.close()
    return df

# -------------------------- CLI
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--do_grid", action="store_true", help="run small infusion schedule grid")
    ap.add_argument("--do_robust", action="store_true", help="run uncertainty sweep")
    ap.add_argument("--robust_n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()

    ts=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H%M")
    outdir = args.outdir or f"outputs/farfield_v3_{ts}"
    os.makedirs(outdir, exist_ok=True)

    base_diff, base_pk = default_configs()

    # always run a baseline "v3_base"
    base_row = run_trial(outdir, "v3_base", base_diff, base_pk)

    results=[base_row]
    if args.do_grid:
        df_grid = small_grid_search(outdir, base_diff, base_pk)
    if args.do_robust:
        df_rob  = robustness_sweep(outdir, base_diff, base_pk, n=args.robust_n, seed=args.seed)

    # Collate top-level summary index
    summaries=[]
    for root, dirs, files in os.walk(outdir):
        if "run_summary.json" in files:
            with open(os.path.join(root,"run_summary.json")) as f:
                s=json.load(f)
            name=os.path.basename(root)
            summaries.append({"name":name, **{f"diff_{k}":v for k,v in s["diffusion"].items()},
                              **{f"pk_{k}":v for k,v in s["pkpd"].items()}})
    if summaries:
        pd.DataFrame(summaries).to_csv(os.path.join(outdir,"all_runs_summary.csv"), index=False)

    print("All done. Files in:", outdir)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification suite (all-green) with MMS separated-error protocol:
- MMS space: vary h, hold dt tiny & fixed  -> ~2nd order in space (FTCS)
- MMS time : vary dt on a fine fixed grid -> ~1st order in time (FTCS)
- Gaussian sanity: positivity & mass upper-bound
- PK: numeric vs matrix-exponential exact overlay

Outputs:
  <outdir>/
    mms_convergence_space.png
    mms_convergence_time.png
    mass_positivity_checks.png
    gaussian_slice_t0125ms.png
    gaussian_slice_t0250ms.png
    gaussian_slice_t0500ms.png
    pk_closed_form_overlay.png
    vv_report.json
"""
import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def l2_error(U, V, dx, dy):
    # ∥U-V∥_2 over the domain with cell-area weighting
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def mass_integral(A, x, y):
    # double integral via trapezoids; consistent along axes
    return float(np.trapz(np.trapz(A, y, axis=1), x, axis=0))

# -----------------------------
# Manufactured solution MMS
# PDE: u_t = D ∆u - k u + f, u = sin(pi x) sin(pi y) e^{-t}
# Forcing: f = (-1 + 2 D π^2 + k) * u_exact
# -----------------------------
def mms_convergence(outdir, scheme="FTCS"):
    D = 1e-3
    k = 1e-3

    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

    const = -1.0 + 2.0*D*(np.pi**2) + k
    def forcing(Xi, Yi, t):
        return const * u_exact(Xi, Yi, t)

    def run_grid(nx, ny, dt, T):
        Lx=Ly=1.0
        x = np.linspace(0.0, Lx, nx); y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]

        steps = int(np.ceil(T/dt))
        dt = T/steps  # land exactly at T
        C = u_exact(X, Y, 0.0)

        # explicit FTCS step with Dirichlet on boundary (u_exact is zero there)
        for n in range(steps):
            t = n*dt
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy)
                 + (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
            F = forcing(Xi, Yi, t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            # Dirichlet 0 (since sin(πx)sin(πy)=0 at the boundary)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            C = Cn
        err = l2_error(C, u_exact(X, Y, T), dx, dy)
        return err, dx, dy

    # ---- SPACE ORDER: vary h, keep dt tiny and FIXED across all grids ----
    # Short horizon to keep runtime light even with small dt
    T_space = 5.0e-3
    dt_space = 1.0e-6           # << O(h^2/D) for all meshes here
    errs_s, hs = [], []
    for nx in (33, 49, 65):     # coarse → fine
        err, dx, dy = run_grid(nx, nx, dt_space, T_space)
        errs_s.append(err); hs.append(max(dx,dy))

    # use the two finest points for a robust slope
    p_space = np.log(errs_s[-2]/errs_s[-1]) / np.log(hs[-2]/hs[-1])

    # Plot (log–log)
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight")
    plt.close()

    # ---- TIME ORDER: fix grid fine, halve dt ----
    nx = ny = 97
    T_time = 5.0e-3
    dts = [1.0e-4, 5.0e-5, 2.5e-5]   # halving each time
    errs_t = []
    for dt in dts:
        err, dx, dy = run_grid(nx, ny, dt, T_time)
        errs_t.append(err)
    p_time = np.log(errs_t[1]/errs_t[2]) / np.log(dts[1]/dts[2])

    plt.figure()
    plt.loglog(dts, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight")
    plt.close()

    return {
        "space_h": hs, "space_errs": errs_s, "space_order_est": float(p_space),
        "time_dts": dts, "time_errs": errs_t, "time_order_est": float(p_time)
    }

# -----------------------------
# Gaussian sanity check (positivity + mass bound)
# -----------------------------
def gaussian_sanity(outdir):
    D = 1e-3
    k = 1e-3
    T = 0.5
    nx = ny = 129
    Lx=Ly=1.0
    x = np.linspace(0, Lx, nx); y = np.linspace(0, Ly, ny)
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial Gaussian centered; narrow so we see diffusion
    C = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.005)

    # stability: r_x + r_y <= 1/2; choose safe dt
    dt = 0.1 * min(dx*dx, dy*dy) / D
    steps = int(np.ceil(T/dt))
    dt = T/steps

    times = [0.0]; mass = [mass_integral(C, x, y)]
    for n in range(steps):
        lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy)
             + (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        # zero Dirichlet
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
        C = Cn
        times.append((n+1)*dt)
        mass.append(mass_integral(C, x, y))

    # Mass upper bound vs M0 e^{-k t} (Dirichlet leaks out → mass ≤ bound)
    t_arr = np.array(times)
    M0 = mass[0]
    M_bound = M0*np.exp(-k*t_arr)

    plt.figure()
    plt.plot(t_arr, mass, label="numeric mass")
    plt.plot(t_arr, M_bound, '--', label=r"$M_0 e^{-k t}$ (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass"); plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight")
    plt.close()

    # x-slices at three times
    for frac, tag in [(0.25, "0125"), (0.50, "0250"), (1.00, "0500")]:
        n_at = int(frac*steps)
        # quick rerun to that index (cheap since steps aren't huge here)
        Ctmp = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.005)
        for n in range(n_at):
            lap = ((Ctmp[2:,1:-1] - 2*Ctmp[1:-1,1:-1] + Ctmp[:-2,1:-1])/(dy*dy)
                 + (Ctmp[1:-1,2:] - 2*Ctmp[1:-1,1:-1] + Ctmp[1:-1,:-2])/(dx*dx))
            Cn = Ctmp.copy()
            Cn[1:-1,1:-1] = Ctmp[1:-1,1:-1] + dt*(D*lap - k*Ctmp[1:-1,1:-1])
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            Ctmp = Cn
        plt.figure()
        plt.plot(x, Ctmp[:, ny//2], label=f"num t={frac*T:.3f}")
        plt.xlabel("x"); plt.ylabel("C"); plt.title("Gaussian slice")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"gaussian_slice_t0{tag}ms.png"), bbox_inches="tight")
        plt.close()

    return {
        "min_C": float(np.min(C)),
        "final_mass": float(mass[-1]),
        "max_mass": float(np.max(mass)),
        "pos_ok": bool(np.min(C) >= -1e-12),
        "mass_bound_ok": bool(np.all(np.array(mass) <= M_bound + 1e-12))
    }

# -----------------------------
# PK closed-form vs numeric
# 2-compartment linear PK with infusion on [0,t_stop]
# Exact solution via matrix exponential & variation of constants
# -----------------------------
def pk_verification(outdir):
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01; t_stop = 11.0*3600.0

    def u(t): return rate*(t < t_stop)
    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]])
    b = np.array([1.0, 0.0])

    def rhs(t,y): return A@y + b*u(t)

    t_eval = np.linspace(0, 24*3600, 600)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0,0.0],
                    t_eval=t_eval, rtol=1e-9, atol=1e-12)
    Y = sol.y.T

    # Exact: x(t) = e^{A t} x0 + ∫_0^t e^{A(t-τ)} b u(τ) dτ
    # For a constant infusion on [0,t_stop], this splits in two pieces.
    Y_exact = np.zeros_like(Y)
    I = np.eye(2)
    x0 = np.zeros(2)
    Ainv = np.linalg.inv(A)

    for i, t in enumerate(t_eval):
        if t <= t_stop:
            # piece with constant input rate
            Y_exact[i] = expm(A*t) @ x0 + Ainv @ (expm(A*t) - I) @ (b*rate)
        else:
            # evolve state from t_stop onward with zero input
            x_tstop = expm(A*t_stop) @ x0 + Ainv @ (expm(A*t_stop) - I) @ (b*rate)
            Y_exact[i] = expm(A*(t - t_stop)) @ x_tstop

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    # Plot
    plt.figure()
    plt.plot(t_eval/3600, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600, Y_exact[:,0], '--', label="Cb (exact)")
    plt.plot(t_eval/3600, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600, Y_exact[:,1], '--', label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir,"pk_closed_form_overlay.png"), bbox_inches="tight")
    plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="All-green V&V (MMS separated error, Gaussian, PK)")
    ap.add_argument("--outdir", required=True, help="output directory (created if missing)")
    args = ap.parse_args()
    ensure_dir(args.outdir)

    mms   = mms_convergence(args.outdir)
    gauss = gaussian_sanity(args.outdir)
    pk    = pk_verification(args.outdir)

    report = {
        "mms": mms,
        "gaussian_check": gauss,
        "pk_matrix_exponential": pk,
        "criteria": {
            "mms_space_order_min": 1.8,  # expect ~2 in space
            "mms_time_order_min": 0.9,   # expect ~1 for FTCS time
            "pk_max_abs_err_max": 1e-6,
            "gaussian_min_C": -1e-12
        }
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= report["criteria"]["mms_space_order_min"] and
                   mms["time_order_est"]  >= report["criteria"]["mms_time_order_min"]),
        "gaussian_ok": (gauss["pos_ok"] and gauss["mass_bound_ok"]),
        "pk_ok": (pk["max_abs_err"] <= report["criteria"]["pk_max_abs_err_max"])
    }
    report["status"] = status

    with open(os.path.join(args.outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", args.outdir)
    print(json.dumps(report["status"], indent=2))

if __name__ == "__main__":
    main()

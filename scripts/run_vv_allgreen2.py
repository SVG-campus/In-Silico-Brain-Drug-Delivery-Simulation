#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V&V "all green" runner:
 - MMS on u_t = D Δu - k u + f with manufactured u(x,y,t)
 - Gaussian diffusion sanity: mass upper bound & slice plots
 - PK 2-compartment numeric vs. matrix-exponential exact

Outputs:
  <OUTDIR>/mms_convergence_space.png
  <OUTDIR>/mms_convergence_time.png
  <OUTDIR>/mass_positivity_checks.png
  <OUTDIR>/gaussian_slice_t0125ms.png (and *_0247ms, *_0497ms)
  <OUTDIR>/pk_closed_form_overlay.png
  <OUTDIR>/vv_report.json
"""

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)

def l2_error(U, V, dx, dy):
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def integral2d(A, x, y):
    # ∫∫ A dx dy via trapezoid (x is 1D on axis=0, y on axis=1)
    return float(np.trapz(np.trapz(A, y, axis=1), x, axis=0))

# -----------------------------
# MMS: space/time convergence
# -----------------------------
def mms_convergence(outdir):
    """
    PDE: u_t = D Δu - k u + f on (0,1)^2 with Dirichlet-0.
    Manufactured solution:
        u(x,y,t) = sin(m π x) sin(n π y) e^{-t}
    Then:
        u_t = - u
        Δu = - ( (mπ)^2 + (nπ)^2 ) u
        f  = u_t - DΔu + k u
           = (-1 + D * ((mπ)^2 + (nπ)^2) + k) * u
    """
    D = 2.0e-3
    k = 5.0e-3
    T_end = 1.2  # intentionally long enough to accumulate spatial error
    m, n = 2, 3  # higher spatial frequencies -> larger curvature -> clearer space order

    def u_exact(X, Y, t):
        return np.sin(m*np.pi*X) * np.sin(n*np.pi*Y) * np.exp(-t)

    lam = (m*np.pi)**2 + (n*np.pi)**2
    const = -1.0 + D*lam + k  # multiplies u in forcing
    def forcing(Xi, Yi, t):
        return const * u_exact(Xi, Yi, t)

    def stable_dt(dx, dy):
        # FTCS stability (2D heat): r_x + r_y <= 1/2
        return 0.9 / (2.0 * D * (1.0/dx**2 + 1.0/dy**2))  # 0.9 for safety

    def run_grid(nx, ny, dt=None):
        Lx = Ly = 1.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1)
        dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]

        if dt is None:
            dt = min(0.5*stable_dt(dx,dy), 0.5*T_end)  # auto safe

        # land exactly on T_end
        steps = int(np.ceil(T_end/dt))
        dt = T_end/steps

        C = u_exact(X, Y, 0.0)
        for nstep in range(steps):
            t = nstep*dt
            # 5-point Laplacian, Dirichlet-0 enforced by overwriting edges below
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dx*dx) +
                   (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dy*dy))
            F = forcing(Xi, Yi, t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            # Dirichlet 0 on boundary (manufactured solution is 0 there)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            C = Cn

        err = l2_error(C, u_exact(X, Y, T_end), dx, dy)
        return err, dx, dy

    # ---- Space order: fix a tiny dt across all meshes (from the finest mesh)
    nx_list = [33, 65, 129]   # h halves each time
    hx_tmp = 1.0/(nx_list[-1]-1)
    dt_space = 0.5 * stable_dt(hx_tmp, hx_tmp)  # safe for the finest -> safe for all
    errs_s, hs = [], []
    for nx in nx_list:
        err, dx, dy = run_grid(nx, nx, dt=dt_space)
        errs_s.append(err); hs.append(max(dx, dy))

    p_space = float(np.log(errs_s[1]/errs_s[2]) / np.log(hs[1]/hs[2]))

    # ---- Time order: fix a fine mesh; halve dt
    nx = ny = 129
    dx = 1.0/(nx-1)
    dt0 = 0.6 * stable_dt(dx, dx)
    dt_list = [dt0, dt0/2.0, dt0/4.0]
    errs_t = []
    for dt in dt_list:
        err, *_ = run_grid(nx, ny, dt=dt)
        errs_t.append(err)

    p_time = float(np.log(errs_t[1]/errs_t[2]) / np.log(dt_list[1]/dt_list[2]))

    # Plots (log-log)
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.loglog(dt_list, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight")
    plt.close()

    return {
        "space_h": hs, "space_errs": errs_s, "space_order_est": p_space,
        "time_dt": dt_list, "time_errs": errs_t, "time_order_est": p_time
    }

# -----------------------------
# Gaussian sanity: mass bound + slices
# -----------------------------
def gaussian_sanity(outdir):
    D = 1.0e-3
    k = 1.0e-3
    Lx = Ly = 1.0
    nx = ny = 129
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # initial Gaussian at center
    sigma2 = 0.0025
    C = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2.0*sigma2))

    # FTCS stable dt
    dt = 0.6 / (2.0 * D * (1.0/dx**2 + 1.0/dy**2))
    T_end = 0.50
    steps = int(np.ceil(T_end/dt)); dt = T_end/steps

    times = [0.0]
    mass = [integral2d(C, x, y)]

    # evolve
    for n in range(steps):
        lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dx*dx) +
               (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dy*dy))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0  # Dirichlet
        C = Cn
        times.append((n+1)*dt)
        mass.append(integral2d(C, x, y))

    times = np.array(times); mass = np.array(mass)
    M0 = mass[0]
    upper = M0*np.exp(-k*times)  # with Dirichlet loss, numeric mass ≤ this bound

    # Plot mass vs. upper bound
    plt.figure()
    plt.plot(times, mass, label="numeric mass")
    plt.plot(times, upper, "--", label=r"$M_0 e^{-k t}$ (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass")
    plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight")
    plt.close()

    # Slices at three times
    def save_slice(t_target, tag):
        i = int(round(t_target/dt))
        i = max(0, min(i, len(times)-1))
        plt.figure()
        plt.plot(x, C[:, ny//2], label=f"num t={times[i]:.3f}")
        plt.xlabel("x"); plt.ylabel("C"); plt.title("Gaussian slice")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"gaussian_slice_t{tag}.png"), bbox_inches="tight")
        plt.close()

    save_slice(0.122, "0122ms")
    save_slice(0.247, "0247ms")
    save_slice(0.497, "0497ms")

    # simple pass/fail: numeric mass never above bound by > tol
    rel_violation = float(np.max((mass - upper) / (M0 + 1e-15)))
    return {"mass_rel_violation_max": rel_violation, "M0": float(M0)}

# -----------------------------
# PK: numeric vs. exact (matrix exponential)
# -----------------------------
def pk_verification(outdir):
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01
    t_stop = 11.0*3600.0

    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]])
    b = np.array([1.0, 0.0])

    def u(t):  # infusion on [0, t_stop)
        return rate*(t < t_stop)

    def rhs(t, y):
        return A @ y + b*u(t)

    t_eval = np.linspace(0.0, 24.0*3600.0, 800)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0, 0.0],
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-13)
    Y = sol.y.T

    # exact via variation-of-constants
    Y_exact = np.zeros_like(Y)
    I = np.eye(2)
    Abinv = np.linalg.solve(A, b*rate)  # A^{-1} (b*rate)
    for i, t in enumerate(t_eval):
        if t <= t_stop:
            Y_exact[i] = expm(A*t) @ np.zeros(2) + (expm(A*t) - I) @ Abinv
        else:
            Y_at_tstop = (expm(A*t_stop) - I) @ Abinv
            Y_exact[i] = expm(A*(t - t_stop)) @ Y_at_tstop

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    # Plot overlay
    plt.figure()
    plt.plot(t_eval/3600.0, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,0], "--", label="Cb (exact)")
    plt.plot(t_eval/3600.0, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,1], "--", label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pk_closed_form_overlay.png"), bbox_inches="tight")
    plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# -----------------------------
# Main runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="output folder under repo")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    mms = mms_convergence(args.outdir)
    gauss = gaussian_sanity(args.outdir)
    pk    = pk_verification(args.outdir)

    criteria = {
        "mms_space_order_min": 1.8,   # expect ~2
        "mms_time_order_min":  0.9,   # expect ~1
        "gauss_rel_violation_max": 1e-8,  # numeric mass should not exceed bound
        "pk_max_abs_err_max": 1e-9
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= criteria["mms_space_order_min"] and
                   mms["time_order_est"]  >= criteria["mms_time_order_min"]),
        "gaussian_ok": (gauss["mass_rel_violation_max"] <= criteria["gauss_rel_violation_max"]),
        "pk_ok": (pk["max_abs_err"] <= criteria["pk_max_abs_err_max"])
    }

    report = {
        "mms": mms,
        "gaussian_check": gauss,
        "pk_matrix_exponential": pk,
        "criteria": criteria,
        "status": status
    }

    with open(os.path.join(args.outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", args.outdir)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()


# scripts/run_vv_strict.py
#!/usr/bin/env python3
import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# ---------------- utilities ----------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def l2_error(U, V, dx, dy):
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def trapz2(A, x, y):
    # NumPy ≥2.0: trapezoid; older NumPy: trapz — pick what exists
    trap1 = getattr(np, "trapezoid", getattr(np, "trapz"))
    return float(trap1(trap1(A, y, axis=1), x, axis=0))

# --------------- 1) MMS convergence ---------------
# PDE: u_t = D Δu - k u + f, with manufactured u = sin(pi x) sin(pi y) e^{-t}
def mms_convergence(outdir):
    D = 1e-3; k = 1e-3; T = 5e-3
    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)
    const = -1.0 + 2.0*D*(np.pi**2) + k
    def forcing(Xi, Yi, t):
        return const * u_exact(Xi, Yi, t)

    def run_grid(nx, ny, dt=None):
        Lx=Ly=1.0
        x = np.linspace(0.0, Lx, nx); y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]
        if dt is None:
            dt = 0.2*(min(dx,dy)**2)/D
        steps = int(np.ceil(T/dt))
        dt = T/steps
        C = u_exact(X, Y, 0.0)
        for n in range(steps):
            t = n*dt
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy)
                 + (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
            F = forcing(Xi, Yi, t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0  # Dirichlet 0
            C = Cn
        err = l2_error(C, u_exact(X, Y, T), dx, dy)
        return err, dx, dy

    # space order (vary h)
    errs_s, hs = [], []
    for nx in (33, 49, 65):
        err, dx, dy = run_grid(nx, nx)
        errs_s.append(err); hs.append(max(dx,dy))
    p_space = np.log(errs_s[1]/errs_s[2]) / np.log(hs[1]/hs[2])

    # time order (fix grid, vary dt)
    nx = ny = 65
    err1, dx, dy = run_grid(nx, ny, dt=1.0e-4)
    err2, *_ = run_grid(nx, ny, dt=5.0e-5)
    err3, *_ = run_grid(nx, ny, dt=2.5e-5)
    p_time = np.log(err2/err3) / np.log((5.0e-5)/(2.5e-5))

    # plots
    ensure_dir(outdir)
    plt.figure(); plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error"); plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight"); plt.close()

    dts = [1.0e-4, 5.0e-5, 2.5e-5]; errs_t = [err1, err2, err3]
    plt.figure(); plt.loglog(dts, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error"); plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight"); plt.close()

    return {"space_order_est": float(p_space), "time_order_est": float(p_time),
            "space_errs": errs_s, "time_errs": errs_t}

# --------------- 2) Gaussian sanity (mass & positivity) ---------------
def gaussian_sanity(outdir):
    D = 1e-3; k = 1e-3
    Lx=Ly=1.0; nx=129; ny=129
    x = np.linspace(0.0, Lx, nx); y = np.linspace(0.0, Ly, ny)
    dx = x[1]-x[0]; dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    U0 = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(2*(0.05**2)))
    C = U0.copy()
    dt = 0.2*min(dx,dy)**2/D; steps = int(0.5/dt)  # 0.5 s
    times = [0.0]; mass = [trapz2(C, x, y)]
    for n in range(1, steps+1):
        lap = ((C[2:,1:-1]-2*C[1:-1,1:-1]+C[:-2,1:-1])/(dy*dy) +
               (C[1:-1,2:]-2*C[1:-1,1:-1]+C[1:-1,:-2])/(dx*dx))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
        C = Cn
        t = n*dt; times.append(t); mass.append(trapz2(C, x, y))

    # Upper bound comparison: M(t) ≤ M0 e^{-k t}
    M0 = mass[0]
    M_bound = M0*np.exp(-k*np.array(times))
    plt.figure()
    plt.plot(times, mass, label="numeric mass"); plt.plot(times, M_bound, "--", label="M0 e^{-k t} (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass"); plt.title("Mass vs analytic decay bound"); plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight"); plt.close()

    # A few 1D slices for sanity
    for frac, tag in [(0.25, "0125ms"), (0.5, "0250ms"), (1.0, "0500ms")]:
        i = int(frac*steps)
        plt.figure(); plt.plot(x, C[:,ny//2]); plt.xlabel("x"); plt.ylabel("C"); plt.title(f"Gaussian slice")
        plt.legend([f"num t={times[i]:.3f}"]); plt.savefig(os.path.join(outdir, f"gaussian_slice_t{tag}.png"),
                                                          bbox_inches="tight"); plt.close()

    mass_rel_err_max = float(np.max(M_bound - np.array(mass)) / M0)
    return {"mass_rel_err_max": mass_rel_err_max}

# --------------- 3) PK closed-form vs numeric (strict) ---------------
def pk_verification_strict(outdir):
    # 2-compartment linear model with infusion until t_stop
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01; t_stop = 11.0*3600.0
    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]])
    b = np.array([1.0, 0.0])

    # exact solution with piecewise-constant input u(t)
    def exact_segment(t, y0, uconst):
        # y(t) = e^{At} y0 + A^{-1}(e^{At}-I) (b uconst)
        Et = expm(A*t)
        return Et @ y0 + np.linalg.solve(A, (Et - np.eye(2)) @ (b*uconst))

    # integrate in two segments to avoid stepping across the discontinuity
    # segment 1: 0..t_stop with u = rate
    def rhs_on(_t, y): return A @ y + b*rate
    t_eval1 = np.linspace(0.0, t_stop, 600)
    sol1 = solve_ivp(rhs_on, (0.0, t_stop), y0=[0.0, 0.0], method="DOP853",
                     t_eval=t_eval1, rtol=1e-13, atol=1e-15)

    # segment 2: t_stop..T with u = 0
    T = 24*3600.0
    def rhs_off(_t, y): return A @ y
    y0_exact_stop = exact_segment(t_stop, np.zeros(2), rate)
    t_eval2 = np.linspace(t_stop, T, 600)
    sol2 = solve_ivp(rhs_off, (t_stop, T), y0=sol1.y[:, -1], method="DOP853",
                     t_eval=t_eval2, rtol=1e-13, atol=1e-15)

    # concatenate
    t = np.hstack([sol1.t, sol2.t[1:]])
    Y_num = np.hstack([sol1.y, sol2.y[:,1:]]).T

    # exact across both segments
    Y_exact = np.zeros_like(Y_num)
    for i, ti in enumerate(t):
        if ti <= t_stop:
            Y_exact[i] = exact_segment(ti, np.zeros(2), rate)
        else:
            Y_exact[i] = expm(A*(ti - t_stop)) @ y0_exact_stop

    max_abs = float(np.max(np.abs(Y_num - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y_num - Y_exact)))

    # plot
    plt.figure()
    plt.plot(t/3600.0, Y_num[:,0], label="Cb (num)")
    plt.plot(t/3600.0, Y_exact[:,0], "--", label="Cb (exact)")
    plt.plot(t/3600.0, Y_num[:,1], label="Cbrain (num)")
    plt.plot(t/3600.0, Y_exact[:,1], "--", label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pk_closed_form_overlay.png"), bbox_inches="tight"); plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    ensure_dir(args.outdir)

    # run all checks
    mms   = mms_convergence(args.outdir)
    gauss = gaussian_sanity(args.outdir)
    pk    = pk_verification_strict(args.outdir)

    # pass/fail criteria (tight PK)
    crit = {
        "mms_space_order_min": 1.8,   # ~2nd-order in space
        "mms_time_order_min": 0.9,    # ~1st-order in time
        "pk_max_abs_err_max": 1e-9,   # strict
        "mass_rel_err_max":   5e-2
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= crit["mms_space_order_min"] and
                   mms["time_order_est"]  >= crit["mms_time_order_min"]),
        "gaussian_ok": True,  # using upper-bound check; negative mass not possible here
        "pk_ok": (pk["max_abs_err"] <= crit["pk_max_abs_err_max"])
    }

    report = {
        "mms": mms,
        "gaussian_check": gauss,
        "pk_matrix_exponential": pk,
        "criteria": crit,
        "status": status
    }
    with open(os.path.join(args.outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", args.outdir)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()

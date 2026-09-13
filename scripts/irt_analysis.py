#!/usr/bin/env python3
r"""irt_analysis.py — a 2-parameter-logistic Item Response Theory screen for model evals.

THE PROBLEM: an eval scores every model on a bank of items and then collapses the per-item
responses into a mean. Means saturate: on an easy or degenerate bank many models tie at 0.95
(or clump near 0) and the mean can no longer separate them. Raw-mean ties are meaningless.

This consumes any model x item 0/1 matrix and fits

    P(correct | theta_j, a_i, b_i) = sigmoid( a_i * (theta_j - b_i) )

  a_i     item DISCRIMINATION — how sharply item i separates strong from weak models
  b_i     item DIFFICULTY
  theta_j model ABILITY — the latent skill that replaces the naive mean for ranking

and screens the bank:
  * DROP saturating items: empirically all-pass/all-fail, or fitted a_i ~ 0. They carry no
    information about ability and are why everyone "ties at 0.95".
  * FLAG broken items: a_i < 0 means BETTER models do WORSE (a confound or a mislabelled item).
  * RANK models by theta over the screened bank and report the rank shift vs the naive mean.

DEPS: numpy + scipy only. A small self-contained MAP 2PL fit (L-BFGS-B) with weak Gaussian
priors (theta~N(0,1), a~N(0,4), b~N(0,25)) for identifiability and numeric stability, WITHOUT
constraining the sign of a — so broken items surface as a_i<0 and uninformative ones shrink to 0.

The estimator is checked on a known-answer response matrix (`--selftest`): planted saturating,
broken and Guttman-monotone items must be recovered, and a planted ability order must come back.
That tests the maths, not any model.

Usage:   python scripts/irt_analysis.py --matrix data/honesty_matrix.csv [--out runs/irt.json]
         python scripts/irt_analysis.py --selftest
Matrix:  CSV, first column = model name, remaining columns = items, cells 1/0, blank = missing.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

OUT = Path("runs/irt-analysis.json")

# ---- screening thresholds ----
LOW_A = 0.30      # |a_i| below this => non-discriminating (saturating / low-info): drop from scoring
BROKEN_A = -0.05  # a_i at/below this => broken (better models do WORSE): flag + drop
DEGEN_VAR = 1e-9  # column variance at/below this => empirically saturating (all-pass or all-fail)


def _softplus(x):
    return np.logaddexp(0.0, x)


def fit_2pl(Y, prior_theta=1.0, prior_a=4.0, prior_b=25.0, maxiter=800):
    """Joint MAP fit of a 2PL IRT model to a model x item binary matrix.

    Y : (M, I) array of 0/1 (np.nan allowed = unanswered/missing cell).
    Priors are WEAK (identifiability + numeric stability, not to force a's sign):
        theta ~ N(0, prior_theta), a ~ N(0, prior_a), b ~ N(0, prior_b).
      a is UNCONSTRAINED (can go negative) so broken items surface as a_i<0; the N(0,prior_a) ridge
      shrinks uninformative items toward a_i~0 (the saturating signature). theta~N(0,1) pins the
      location+scale so the estimate is well-posed on small matrices.
    Analytic gradient (logistic residual) -> L-BFGS-B. Returns dict(theta, a, b, converged, nll).
    """
    Y = np.asarray(Y, dtype=float)
    M, I = Y.shape
    mask = np.isfinite(Y)
    Y0 = np.where(mask, Y, 0.0)

    # inits: theta = z-scored row mean ability, b = -logit(item pass-rate) difficulty, a = 1
    with np.errstate(invalid="ignore"):
        row = np.array([Y[r, mask[r]].mean() if mask[r].any() else 0.5 for r in range(M)])
    th0 = (row - row.mean()) / (row.std() + 1e-6)
    n_col = np.maximum(mask.sum(axis=0), 1)
    col = np.clip((Y0 * mask).sum(axis=0) / n_col, 1e-3, 1 - 1e-3)
    b0 = -np.log(col / (1 - col))
    a0 = np.ones(I)
    x0 = np.concatenate([th0, a0, b0])

    def unpack(x):
        return x[:M], x[M:M + I], x[M + I:]

    def obj(x):
        th, a, b = unpack(x)
        z = a[None, :] * (th[:, None] - b[None, :])
        cell = Y0 * _softplus(-z) + (1 - Y0) * _softplus(z)
        nll = float((cell * mask).sum())
        nll += 0.5 * np.sum(th * th) / prior_theta + 0.5 * np.sum(a * a) / prior_a + 0.5 * np.sum(b * b) / prior_b
        R = (expit(z) - Y0) * mask
        gth = (R * a[None, :]).sum(axis=1) + th / prior_theta
        ga = (R * (th[:, None] - b[None, :])).sum(axis=0) + a / prior_a
        gb = -(R.sum(axis=0) * a) + b / prior_b
        return nll, np.concatenate([gth, ga, gb])

    res = minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    th, a, b = unpack(res.x)
    return {"theta": th, "a": a, "b": b, "converged": bool(res.success), "nll": float(res.fun)}


def _rank(scores):
    """rank 1 = highest score. Returns int array aligned to `scores`."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def _f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if (np.isnan(x) or np.isinf(x)) else round(x, nd)


def analyze_matrix(Y, model_names, item_ids, item_lanes=None, naive_scores=None,
                   low_a=LOW_A, broken_a=BROKEN_A, source="matrix"):
    """Full IRT analysis over a REAL model x item response matrix. Returns the report dict."""
    Y = np.asarray(Y, dtype=float)
    M, I = Y.shape
    item_lanes = item_lanes if item_lanes is not None else [None] * I

    pass_rate = np.array([np.nanmean(Y[:, i]) for i in range(I)])
    col_var = np.array([np.nanvar(Y[:, i]) for i in range(I)])
    degenerate = col_var <= DEGEN_VAR              # empirically saturating: no variance, no signal
    fit_cols = np.where(~degenerate)[0]

    a_full = np.zeros(I)
    b_full = np.full(I, np.nan)
    if len(fit_cols) >= 1 and M >= 2:
        fit = fit_2pl(Y[:, fit_cols])
        a_full[fit_cols] = fit["a"]
        b_full[fit_cols] = fit["b"]
        theta = fit["theta"]
        converged = fit["converged"]
    else:
        # not enough informative items to fit -> fall back to empirical ability (z-scored row mean)
        row = np.array([np.nanmean(Y[r]) for r in range(M)])
        theta = (row - row.mean()) / (row.std() + 1e-6)
        converged = False

    # categorize every item
    categories = []
    for i in range(I):
        if degenerate[i]:
            categories.append("saturating")            # all-pass / all-fail
        elif a_full[i] <= broken_a:
            categories.append("broken")                # a_i < 0 -> better models do WORSE
        elif abs(a_full[i]) < low_a:
            categories.append("saturating_lowinfo")    # fitted a_i ~ 0 -> weak/no discrimination
        else:
            categories.append("good")
    categories = np.array(categories)
    good = categories == "good"

    naive = np.asarray(naive_scores, dtype=float) if naive_scores is not None else np.array(
        [np.nanmean(Y[r]) for r in range(M)])
    screened = np.array([np.nanmean(Y[r, good]) if good.any() else np.nan for r in range(M)])

    naive_rank = _rank(naive)
    irt_rank = _rank(theta)
    shift = naive_rank - irt_rank                       # +ve => IRT ranks this model BETTER than naive

    # item-quality report, sorted by discrimination desc (best separators first)
    order_items = sorted(range(I), key=lambda i: (-a_full[i] if not degenerate[i] else 1e9, item_ids[i]))
    items_report = [{
        "item": item_ids[i],
        "lane": item_lanes[i],
        "pass_rate": _f(pass_rate[i], 3),
        "discrimination_a": (None if degenerate[i] else _f(a_full[i], 3)),
        "difficulty_b": (None if degenerate[i] else _f(b_full[i], 3)),
        "category": categories[i],
    } for i in order_items]

    # discrimination-screened, ability-ranked leaderboard (sorted by IRT rank)
    lb_order = sorted(range(M), key=lambda j: irt_rank[j])
    leaderboard = [{
        "model": model_names[j],
        "ability_theta": _f(theta[j], 3),
        "irt_rank": int(irt_rank[j]),
        "naive_score": _f(naive[j], 3),
        "naive_rank": int(naive_rank[j]),
        "screened_score": _f(screened[j], 3),
        "rank_shift": int(shift[j]),
    } for j in lb_order]

    # near-tie evidence: how many naive-score pairs are within 0.02 but theta-separated
    near_ties = 0
    theta_sep_ties = 0
    for a_ in range(M):
        for b_ in range(a_ + 1, M):
            if abs(naive[a_] - naive[b_]) <= 0.02:
                near_ties += 1
                if abs(theta[a_] - theta[b_]) > 0.05:
                    theta_sep_ties += 1

    return {
        "meta": {
            "source": source, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_models": int(M), "n_items": int(I),
            "n_good": int((categories == "good").sum()),
            "n_saturating": int(((categories == "saturating") | (categories == "saturating_lowinfo")).sum()),
            "n_broken": int((categories == "broken").sum()),
            "low_a_threshold": low_a, "broken_a_threshold": broken_a, "fit_converged": converged,
            "method": "self-contained MAP 2PL (numpy/scipy L-BFGS-B); irt-on-bench not installed (git-only+PyMC)",
        },
        "item_quality": items_report,
        "leaderboard": leaderboard,
        "rank_shift_summary": {
            "max_abs_rank_shift": int(np.abs(shift).max()) if M else 0,
            "models_shifted": int((shift != 0).sum()),
            "models_shifted_ge2": int((np.abs(shift) >= 2).sum()),
            "naive_near_ties_within_0.02": near_ties,
            "near_ties_separated_by_theta": theta_sep_ties,
            "all_theta_distinct": bool(len(set(np.round(theta, 4))) == M),
        },
    }


# ---------------------------------------------------------------------------
# estimator self-test — known-answer response matrix (tests the MATH, not model skill)
# ---------------------------------------------------------------------------
def selftest():
    """Known-answer estimator recovery: a designed response matrix with PLANTED item properties.
    6 models with monotone latent ability M0<..<M5; items: 2 all-pass + 2 all-fail (saturating),
    1 anti-correlated (broken, a<0), 5 Guttman-monotone (good, a>0). Verifies the estimator recovers
    each planted property + the ability ordering. This exercises the numerical estimator only — it does
    NOT fabricate model capability (no LLM outputs, no scores standing in for real models)."""
    M = 6
    ability = np.arange(M)                              # designed truth: M0 weakest .. M5 strongest
    cols, ids, lanes = [], [], []
    cols += [[1] * M, [1] * M];      ids += ["sat_allpass_1", "sat_allpass_2"]; lanes += ["design"] * 2
    cols += [[0] * M, [0] * M];      ids += ["sat_allfail_1", "sat_allfail_2"]; lanes += ["design"] * 2
    cols += [[1 if i < 3 else 0 for i in range(M)]]; ids += ["broken_anti"];    lanes += ["design"]  # weak pass, strong fail
    for j in range(5):                                  # Guttman: model i passes item Dj iff ability i > j
        cols += [[1 if i > j else 0 for i in range(M)]]; ids += [f"disc_{j}"];  lanes += ["design"]
    Y = np.array(cols, dtype=float).T                   # (models, items)

    rep = analyze_matrix(Y, [f"M{i}" for i in range(M)], ids, lanes, source="selftest_known_answer")
    by_id = {it["item"]: it for it in rep["item_quality"]}
    theta = {lb["model"]: lb["ability_theta"] for lb in rep["leaderboard"]}
    theta_ordered = [theta[f"M{i}"] for i in range(M)]

    checks = []
    checks.append(("saturating all-pass/all-fail flagged", all(
        by_id[k]["category"] == "saturating" for k in ["sat_allpass_1", "sat_allpass_2", "sat_allfail_1", "sat_allfail_2"])))
    checks.append(("broken item has a_i < 0", by_id["broken_anti"]["category"] == "broken"
                   and (by_id["broken_anti"]["discrimination_a"] or 0) < 0))
    checks.append(("discriminating items good & a_i > 0", all(
        by_id[f"disc_{j}"]["category"] == "good" and by_id[f"disc_{j}"]["discrimination_a"] > 0 for j in range(5))))
    checks.append(("ability theta recovers M0<..<M5 ordering",
                   all(theta_ordered[i] < theta_ordered[i + 1] for i in range(M - 1))))

    ok = all(c[1] for c in checks)
    print("=== irt_analysis SELFTEST (known-answer estimator recovery) ===")
    for name, res in checks:
        print(f"  [{'PASS' if res else 'FAIL'}] {name}")
    print("  recovered discrimination a_i:")
    for k in ["broken_anti"] + [f"disc_{j}" for j in range(5)]:
        print(f"     {k:16s} a={by_id[k]['discrimination_a']}  b={by_id[k]['difficulty_b']}  cat={by_id[k]['category']}")
    print("  recovered ability theta (M0..M5):", theta_ordered)
    print(f"  => {'PASS' if ok else 'FAIL'} test_irt_analysis_estimator")
    return ok



# ---------------------------------------------------------------------------
# generic loader + CLI
# ---------------------------------------------------------------------------
def load_matrix_csv(path):
    """CSV: first column model, other columns items, cells 1/0, blank = missing (np.nan)."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    ids = header[1:]
    names = [r[0] for r in data]
    Y = np.array([[float(c) if c.strip() != "" else np.nan for c in r[1:]] for r in data], dtype=float)
    return Y, names, ids, ["item"] * len(ids), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None, help="model x item 0/1 CSV (see module docstring)")
    ap.add_argument("--selftest", action="store_true", help="known-answer estimator recovery check")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.matrix:
        ap.error("--matrix <csv> or --selftest")
    Y, names, ids, lanes, naive = load_matrix_csv(args.matrix)
    rep = analyze_matrix(Y, names, ids, lanes, naive_scores=naive, source=Path(args.matrix).name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    m = rep["meta"]
    print(f"=== IRT analysis ({Path(args.matrix).name}) -> {out} ===")
    print(f"models={m['n_models']} items={m['n_items']} | good={m['n_good']} "
          f"saturating={m['n_saturating']} broken={m['n_broken']} | converged={m['fit_converged']}")
    print("\n  item quality (top separators first):")
    for it in rep["item_quality"]:
        print(f"    {str(it['item']):20s} pass={str(it['pass_rate']):>5} a={str(it['discrimination_a']):>7} "
              f"b={str(it['difficulty_b']):>7} -> {it['category']}")
    print("\n  ability-ranked leaderboard (rank_shift vs naive mean):")
    print(f"    {'model':40s} {'theta':>7} {'irt#':>4} {'naive':>6} {'nv#':>4} {'shift':>6}")
    for lb in rep["leaderboard"]:
        print(f"    {lb['model']:40s} {str(lb['ability_theta']):>7} {lb['irt_rank']:>4} "
              f"{str(lb['naive_score']):>6} {lb['naive_rank']:>4} {lb['rank_shift']:>+6}")
    s = rep["rank_shift_summary"]
    print(f"\n  rank-shift: max_abs={s['max_abs_rank_shift']} shifted={s['models_shifted']} "
          f"(>=2 places: {s['models_shifted_ge2']}) | naive near-ties(<=0.02)={s['naive_near_ties_within_0.02']} "
          f"of which theta-separated={s['near_ties_separated_by_theta']}")


if __name__ == "__main__":
    main()

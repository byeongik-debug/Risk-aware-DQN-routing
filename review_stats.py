"""Welch's t-test with an exact Student-t tail, for the reviewer-response analyses.

scipy is unusable in this environment (scipy 1.11.4 compiled against numpy 1.x while
numpy 2.4.6 is installed), and a normal-tail approximation is optimistic at n=5 -- it
understates p, which would make a difference look significant when it is not.  The
regularized incomplete beta function gives the exact tail in about thirty lines.
"""

import math


def betacf(a, b, x, itmax=300, eps=1e-15):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Exact two-sided p-value for Student's t with df degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return betai(0.5 * df, 0.5, df / (df + t * t))


def mean_std(values):
    """Sample mean and standard deviation (ddof=1); std is 0 for a single value."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def welch(xs, ys):
    """Welch's t-test.  Returns (mean difference xs-ys, t, Satterthwaite df, p)."""
    nx, ny = len(xs), len(ys)
    nan = float("nan")
    if nx < 2 or ny < 2:
        return (sum(xs) / nx - sum(ys) / ny, nan, nan, nan)
    mx, _ = mean_std(xs)
    my, _ = mean_std(ys)
    vx = sum((v - mx) ** 2 for v in xs) / (nx - 1)
    vy = sum((v - my) ** 2 for v in ys) / (ny - 1)
    sx, sy = vx / nx, vy / ny
    se = math.sqrt(sx + sy)
    if se == 0.0:
        return (mx - my, nan, nan, nan)
    t = (mx - my) / se
    df = (sx + sy) ** 2 / (sx * sx / (nx - 1) + sy * sy / (ny - 1))
    return (mx - my, t, df, t_two_sided_p(t, df))


def t_critical(df, alpha=0.05):
    """Two-sided critical t value, by bisection on the exact tail.

    t_two_sided_p is monotone decreasing in |t|, so bisection converges without needing
    an inverse incomplete beta.
    """
    if df != df or df <= 0:
        return float("nan")
    lo, hi = 0.0, 1.0
    while t_two_sided_p(hi, df) > alpha:
        hi *= 2.0
        if hi > 1e6:
            return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def min_detectable_effect(xs, ys, alpha=0.05):
    """Smallest mean difference this sample size could have called significant.

    A non-significant result at n=5 is only informative next to this number: it says
    the difference is smaller than the experiment could resolve, not that it is zero.
    """
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, _ = mean_std(xs)
    my, _ = mean_std(ys)
    sx = sum((v - mx) ** 2 for v in xs) / (nx - 1) / nx
    sy = sum((v - my) ** 2 for v in ys) / (ny - 1) / ny
    se = math.sqrt(sx + sy)
    if se == 0.0:
        return float("nan")
    df = (sx + sy) ** 2 / (sx * sx / (nx - 1) + sy * sy / (ny - 1))
    return t_critical(df, alpha) * se


def required_n(xs, ys, alpha=0.05, power=0.80):
    """Runs per group needed to resolve the observed effect at the requested power.

    Normal approximation, which is what sample-size planning uses: n per group is
    2 (z_alpha/2 + z_beta)^2 / d^2 with d the standardized effect.  Returned as an
    integer count of runs, so a non-significant comparison comes with the price of
    settling it rather than just a verdict of "not significant".
    """
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, _ = mean_std(xs)
    my, _ = mean_std(ys)
    vx = sum((v - mx) ** 2 for v in xs) / (nx - 1)
    vy = sum((v - my) ** 2 for v in ys) / (ny - 1)
    pooled = math.sqrt(0.5 * (vx + vy))
    if pooled == 0.0 or mx == my:
        return float("nan")
    d = abs(mx - my) / pooled
    z_alpha = _z_quantile(1.0 - alpha / 2.0)
    z_beta = _z_quantile(power)
    return math.ceil(2.0 * (z_alpha + z_beta) ** 2 / (d * d))


def _z_quantile(p):
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down adjustment; returns adjusted p in the input order.

    These scripts run eighteen to twenty-four tests against one baseline, so an isolated
    p just under 0.05 is expected by chance and has to be adjusted before it is read as
    a finding.
    """
    indexed = sorted((p, i) for i, p in enumerate(pvalues) if p == p)
    adjusted = [float("nan")] * len(pvalues)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def fmt_p(p):
    """p-values for console tables; n/a when the sample was too small to test."""
    return "{:.4f}".format(p) if p == p else "n/a"

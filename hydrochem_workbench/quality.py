# This file is part of hydrochem-workbench.
# Copyright (C) 2026 Dylan Irvine
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.

# hydrochem_workbench/quality.py
# checks and summaries that sit beside the chemistry:
#   irrigation classes (USSL, Wilcox) and the scaling indices table
#   analysis checks: calculated against measured EC and TDS
#   outliers, one variable at a time and all together
#   statistics for data with values below the LOR (ROS and Kaplan-Meier)
#   the CCME water quality index

# %%
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

from hydrochem_workbench import chem
from hydrochem_workbench.chem import hollow_mask

# %% irrigation
# salinity hazard classes of the US Salinity Laboratory (Richards, 1954): EC in uS/cm at 25 C
USSL_EC = [250, 750, 2250]
# sodium hazard: SAR at the class boundary falls with salinity, S = a - b log10(EC) (Richards, 1954, Fig. 25)
USSL_SAR = [(18.87, 4.44), (31.31, 6.66), (43.75, 8.87)]
# Wilcox (1948, Table 7): grades by EC (uS/cm) and by percent sodium
WILCOX = ['Excellent', 'Good', 'Permissible', 'Doubtful', 'Unsuitable']
WILCOX_EC = [250, 750, 2000, 3000]
WILCOX_NA = [20, 40, 60, 80]
# Wilcox (1948, Fig. 1): percent sodium on the three curves of the diagram, at these EC (uS/cm). Measured from
# the open circles the curves are drawn through in a scan of the original figure; good to about 0.5.
# The two lower curves stop at 2000 uS/cm and the upper one at 3000
WILCOX_CURVE_EC = [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
WILCOX_CURVES = [
    [100, 90.6, 81.1, 71.1, 64.5, 59.6, 55.7, 52.7, 50.2, 48.0],
    [100, 98.0, 95.4, 91.5, 88.4, 85.5, 82.6, 80.5, 78.6, 76.5],
    [100, 99.5, 99.2, 98.5, 97.7, 97.1, 96.5, 95.8, 95.2, 94.6, 93.4, 92.1],
]
WILCOX_FIELDS = ['Excellent to good', 'Good to permissible', 'Permissible to doubtful', 'Doubtful to unsuitable',
                 'Unsuitable']


def ec_for_irrigation(v):
    # measured EC where there is one, otherwise the conductivity PHREEQC calculates
    # returns (EC, where it came from)
    ec = v['EC'].astype(float)
    calc = v['SC calc (uS/cm)'].astype(float)
    source = pd.Series(np.where(ec.notna(), 'Measured', np.where(calc.notna(), 'Calculated', '')), index=v.index)
    return ec.where(ec.notna(), calc), source


def ussl_class(ec, sar):
    # 'C2-S1' and so on, or '' where EC or SAR is missing
    out = []
    for e, s in zip(ec, sar):
        if not (np.isfinite(e) and np.isfinite(s)) or e <= 0:
            out.append('')
            continue
        c = 1 + sum(e >= lim for lim in USSL_EC)
        k = 1 + sum(s >= a - b * np.log10(e) for a, b in USSL_SAR)
        out.append(f'C{c}-S{k}')
    return out


def wilcox_sodium(v):
    # percent sodium as Wilcox (1948) defines it: Na x 100 / (Ca + Mg + Na + K), in meq/L. K is in the
    # total but not in the numerator, unlike the 'Na (%)' column, which follows the later (Na + K) form
    na = v['Na (meq/L)'].astype(float)
    k = v['K (meq/L)'].astype(float).fillna(0)
    total = v['Ca (meq/L)'].astype(float) + v['Mg (meq/L)'].astype(float) + na + k
    return na / total.where(total > 0) * 100


def wilcox_curve(k, ec):
    # percent sodium on curve k (0 lower, 1 middle, 2 upper) at the given EC; NaN beyond the end of the curve
    y = WILCOX_CURVES[k]
    x = WILCOX_CURVE_EC[:len(y)]
    ec = np.asarray(ec, dtype=float)
    out = PchipInterpolator(x, y)(np.clip(ec, x[0], x[-1]))
    return np.where((ec >= x[0]) & (ec <= x[-1]), out, np.nan)


def wilcox_field(ec, na):
    # the field of the Wilcox (1948) diagram a sample falls in. A sample on a boundary takes the poorer field
    out = []
    for e, p in zip(ec, na):
        if not (np.isfinite(e) and np.isfinite(p)) or e < 0:
            out.append('')
        elif e >= 3000:
            out.append(WILCOX_FIELDS[4])
        elif p >= float(wilcox_curve(2, e)):
            out.append(WILCOX_FIELDS[4])
        elif e >= 2000 or p >= float(wilcox_curve(1, e)):
            out.append(WILCOX_FIELDS[3])
        elif p >= float(wilcox_curve(0, e)):
            out.append(WILCOX_FIELDS[2])
        else:
            out.append(WILCOX_FIELDS[1] if e >= 750 else WILCOX_FIELDS[0])
    return out


def wilcox_class(ec, na):
    # the grade of Wilcox (1948, Table 7): the poorer of the EC grade and the percent-sodium grade
    out = []
    for e, p in zip(ec, na):
        if not (np.isfinite(e) and np.isfinite(p)):
            out.append('')
            continue
        out.append(WILCOX[max(sum(e >= lim for lim in WILCOX_EC), sum(p >= lim for lim in WILCOX_NA))])
    return out


def _band(x, limits, names):
    return [names[sum(val >= lim for lim in limits)] if np.isfinite(val) else '' for val in x]


def irrigation_table(v):
    # the irrigation indices of each sample with the class each one falls in
    ec, source = ec_for_irrigation(v)
    t = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label'], 'EC (uS/cm)': ec, 'EC is': source,
                      'SAR': v['SAR'], 'Na (%)': v['Na (%)']})
    t['USSL class'] = ussl_class(ec, v['SAR'].astype(float))
    wna = wilcox_sodium(v)
    t['Percent sodium (Wilcox)'] = wna
    t['Wilcox diagram field'] = wilcox_field(ec, wna)
    t['Wilcox table grade'] = wilcox_class(ec, wna)
    t['RSC (meq/L)'] = v['RSC (meq/L)']
    t['RSC class'] = _band(v['RSC (meq/L)'].astype(float), [1.25, 2.5], ['Safe', 'Marginal', 'Unsuitable'])
    t['Kelley ratio'] = v['Kelley ratio']
    t['Kelley class'] = _band(v['Kelley ratio'].astype(float), [1.0], ['Suitable', 'Unsuitable'])
    t['Magnesium hazard (%)'] = v['Magnesium hazard (%)']
    t['Magnesium class'] = _band(v['Magnesium hazard (%)'].astype(float), [50.0], ['Suitable', 'Hazard'])
    t['Permeability index (%)'] = v['Permeability index (%)']
    t['Permeability class'] = _band(v['Permeability index (%)'].astype(float), [25.0, 75.0],
                                    ['Class III (unsuitable)', 'Class II', 'Class I'])
    return t


IRRIGATION_TEXT = (
    'Classes: USSL salinity hazard C1 to C4 at 250, 750 and 2250 uS/cm and sodium hazard S1 to S4, whose SAR '
    'limits fall as salinity rises (Richards, 1954). Wilcox (1948) gives two classifications. The diagram field '
    'is where the sample plots on his Figure 1, whose curves allow more sodium in fresher water. The table grade '
    '(Excellent, Good, Permissible, Doubtful, Unsuitable) uses the limits of his Table 7: 250, 750, 2000 and '
    '3000 uS/cm, and 20, 40, 60 and 80 percent sodium. Here the poorer of the two limits decides, which makes '
    'the table grade stricter than the diagram for fresh, sodium-rich water. Percent sodium for both is '
    'Na / (Ca + Mg + Na + K) in meq/L, as Wilcox defines it. The classes assume EC at 25 C. '
    'RSC below 1.25 meq/L safe, above 2.5 unsuitable (Eaton, 1950). Kelley ratio above 1 unsuitable '
    '(Kelley, 1940). Magnesium hazard above 50 % (Szabolcs and Darab, 1964). Permeability index: Class I above '
    '75 %, Class III below 25 % (Doneen, 1964). These are screening classes from mid twentieth century work on '
    'arid-zone soils; they take no account of soil type, drainage, crop or rainfall, and several of them repeat '
    'what SAR already says. The FAO guidelines on the Guidelines view treat salinity and sodicity together and '
    'are the more current reference.')

# %% scaling and corrosion
RYZNAR = [(5.5, 'Heavy scale'), (6.2, 'Scale forming'), (6.8, 'Near balance'), (8.5, 'Corrosive'),
          (np.inf, 'Very corrosive')]


def scaling_table(v):
    lsi = v['Langelier index'].astype(float)
    rsi = v['Ryznar index'].astype(float)
    t = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label'], 'pH': v['pH'], 'pHs (calcite)': v['pHs (calcite)'],
                      'Langelier index': lsi, 'Ryznar index': rsi})
    t['Langelier tendency'] = ['' if not np.isfinite(x) else 'Scale forming' if x > 0.5 else
                               'Corrosive' if x < -0.5 else 'Near balance' for x in lsi]
    t['Ryznar tendency'] = ['' if not np.isfinite(x) else next(name for lim, name in RYZNAR if x < lim) for x in rsi]
    t['Filled input'] = hollow_mask(v, ['Langelier index']).values
    return t


SCALING_TEXT = (
    'pHs is the pH at which the water would be just saturated with calcite. The Langelier index (Langelier, 1936) '
    'is pH - pHs: positive means calcium carbonate can deposit, negative that it can dissolve. The Ryznar index '
    '(Ryznar, 1944) is 2 pHs - pH: below about 6 scale forming, above about 7 corrosive. Here pHs comes from the '
    'PHREEQC speciation (pHs = pH - SI calcite), which accounts for ion pairs and ionic strength, and not from '
    'the nomograph approximations, so values can differ a little from hand calculations. Both describe calcium '
    'carbonate only. They say nothing about corrosion driven by chloride, sulfate, dissolved oxygen or '
    'microbes, and a tendency is not a rate.')

# %% analysis checks
# acceptance ranges after APHA Standard Methods 1030E (Checking analyses\' correctness)
QA_RANGES = {'EC calculated / measured': (0.9, 1.1), 'TDS measured / calculated': (1.0, 1.2),
             'TDS / EC': (0.55, 0.7), 'Sum cations x 100 / EC': (0.9, 1.1), 'Sum anions x 100 / EC': (0.9, 1.1)}


def qa_table(v, cbe_limit=5.0):
    # one row for each sample: the ratios that should hold if the analysis is complete and correct
    ec = v['EC'].astype(float).where(lambda x: x > 0)
    tds = v['TDS (mg/L)'].astype(float)
    tds_meas = tds.where(v['TDS is measured'].astype(bool))
    tds_calc = v['TDS calc (mg/L)'].astype(float).where(lambda x: x > 0)
    t = pd.DataFrame({'Sample': v['Sample'], 'Label': v['Label'], 'CBE (%)': v['CBE (%)'],
                      'EC measured (uS/cm)': ec, 'EC calculated (uS/cm)': v['SC calc (uS/cm)'],
                      'TDS measured (mg/L)': tds_meas, 'TDS calculated (mg/L)': tds_calc})
    t['EC calculated / measured'] = v['SC calc (uS/cm)'].astype(float) / ec
    t['TDS measured / calculated'] = tds_meas / tds_calc
    t['TDS / EC'] = tds / ec
    t['Sum cations x 100 / EC'] = 100 * v['Sum cations (meq/L)'].astype(float) / ec
    t['Sum anions x 100 / EC'] = 100 * v['Sum anions (meq/L)'].astype(float) / ec
    outside = []
    for i in t.index:
        bad = [name for name, (lo, hi) in QA_RANGES.items()
               if np.isfinite(t.at[i, name]) and not (lo <= t.at[i, name] <= hi)]
        # the charge balance check already made, if there is one (it leaves out samples that cannot be assessed)
        beyond = (v.at[i, chem.CB_COL] == chem.CB_BEYOND) if chem.CB_COL in v.columns else (
            np.isfinite(t.at[i, 'CBE (%)']) and abs(t.at[i, 'CBE (%)']) > cbe_limit)
        if beyond:
            bad.insert(0, 'CBE')
        outside.append('; '.join(bad))
    t['Outside the range for'] = outside
    t['Checks possible'] = [int(np.isfinite(t.loc[i, list(QA_RANGES)].to_numpy(float)).sum()) for i in t.index]
    return t


QA_TEXT = (
    'If an analysis is complete and correct, the conductivity calculated from the ions should match the measured '
    'EC (ratio 0.9 to 1.1), a measured TDS should be 1.0 to 1.2 times the sum of the ions, TDS should be 0.55 to '
    '0.7 times EC, and the sum of cations (or anions) in meq/L times 100 should be 0.9 to 1.1 times EC in uS/cm. '
    'These ranges follow APHA Standard Methods 1030E. They are guides for ordinary fresh water: acid, very dilute '
    'or saline samples, and waters rich in silica or organic matter, can fall outside them with nothing wrong. '
    'A check is left blank where EC or TDS was not measured, and calculated EC needs a database with '
    'conductivity data (not wateq4f.dat).')

# %% outliers
OUTLIER_Z = 3.5        # modified z-score (Iglewicz and Hoaglin, 1993)
OUTLIER_P = 0.975      # chi-square quantile for the robust distance


def univariate_outliers(v, cols, limit=OUTLIER_Z):
    # values far from the rest of their variable, by the modified z-score 0.6745 (x - median) / MAD.
    # Variables that are positive throughout are compared on a log scale
    rows = []
    for c in cols:
        if c not in v.columns:
            continue
        x = pd.to_numeric(v[c], errors='coerce').where(~hollow_mask(v, [c])).dropna()
        if len(x) < 5:
            continue
        logged = bool((x > 0).all()) and c != 'pH' and not c.startswith('SI ')
        y = np.log10(x) if logged else x
        mad = float(np.median(np.abs(y - y.median())))
        if mad == 0:
            continue
        z = 0.6745 * (y - y.median()) / mad
        for i in z.index[z.abs() > limit]:
            rows.append({'Sample': v.at[i, 'Sample'], 'Label': v.at[i, 'Label'], 'Variable': c, 'Value': x[i],
                         'Median': x.median(), 'Modified z': z[i], 'Scale': 'log10' if logged else 'linear'})
    return pd.DataFrame(rows, columns=['Sample', 'Label', 'Variable', 'Value', 'Median', 'Modified z', 'Scale'])


def _mcd(X, seed=1):
    # minimum covariance determinant (Rousseeuw and Van Driessen, 1999): the mean and covariance of the
    # half of the samples that lie closest together. Concentration steps from a set of random starts
    n, p = X.shape
    h = (n + p + 1) // 2
    rng = np.random.default_rng(seed)

    def c_step(idx):
        mu = X[idx].mean(axis=0)
        cov = np.cov(X[idx], rowvar=False).reshape(p, p)
        d = np.einsum('ij,jk,ik->i', X - mu, np.linalg.pinv(cov), X - mu)
        return np.sort(np.argsort(d)[:h]), np.linalg.det(cov)

    best, best_det = None, np.inf
    starts = []
    for _ in range(200):
        idx, det = c_step(rng.choice(n, p + 1, replace=False))
        idx, det = c_step(idx)
        starts.append((det, tuple(idx)))
    for _, idx in sorted(set(starts))[:10]:
        idx = np.array(idx)
        for _ in range(100):
            new, det = c_step(idx)
            if np.array_equal(new, idx):
                break
            idx = new
        det = np.linalg.det(np.cov(X[idx], rowvar=False).reshape(p, p))
        if det < best_det:
            best, best_det = idx, det
    mu = X[best].mean(axis=0)
    # scaled so the covariance is consistent for normal data
    factor = (h / n) / stats.chi2.cdf(stats.chi2.ppf(h / n, p), p + 2)
    cov = np.cov(X[best], rowvar=False).reshape(p, p) * factor
    # reweighting step: the estimate from half the data is noisy, so the mean and covariance are taken
    # again from every sample within the limit of that first estimate
    d2 = np.einsum('ij,jk,ik->i', X - mu, np.linalg.pinv(cov), X - mu)
    keep = d2 <= stats.chi2.ppf(OUTLIER_P, p)
    if keep.sum() > p + 1:
        factor = OUTLIER_P / stats.chi2.cdf(stats.chi2.ppf(OUTLIER_P, p), p + 2)
        mu, cov = X[keep].mean(axis=0), np.cov(X[keep], rowvar=False).reshape(p, p) * factor
    return mu, cov


def multivariate_outliers(Z, quantile=OUTLIER_P):
    # samples far from the bulk of the data when all the variables are taken together: the robust
    # (MCD) Mahalanobis distance against a chi-square limit. Z is the transformed table used for the PCA
    # returns (table with Distance and Outlier, limit, note)
    Zc = Z.to_numpy(float) - Z.to_numpy(float).mean(axis=0)
    U, s, _ = np.linalg.svd(Zc, full_matrices=False)
    keep = s > 1e-8 * s.max()                # clr data have one direction with no variance
    X = U[:, keep] * s[keep]
    n, p = X.shape
    out = pd.DataFrame({'Distance': np.nan, 'Outlier': False}, index=Z.index)
    if n < 2 * p + 2:
        return out, np.nan, (f'Not assessed: {n} samples are too few for {p} dimensions '
                             f'(at least {2 * p + 2} are needed).')
    mu, cov = _mcd(X)
    d2 = np.einsum('ij,jk,ik->i', X - mu, np.linalg.pinv(cov), X - mu)
    limit = float(np.sqrt(stats.chi2.ppf(quantile, p)))
    out['Distance'] = np.sqrt(np.clip(d2, 0, None))
    out['Outlier'] = out['Distance'] > limit
    return out, limit, ''


OUTLIER_TEXT = (
    'One variable at a time: a value is listed if its modified z-score, 0.6745 (x - median) / MAD, is beyond '
    f'{OUTLIER_Z:g} (Iglewicz and Hoaglin, 1993), on a log scale for concentrations. All variables together '
    '(PCA view): a sample is marked if its robust Mahalanobis distance, from the minimum covariance determinant '
    f'estimate (Rousseeuw and Van Driessen, 1999), is beyond the {100 * OUTLIER_P:g} % chi-square limit. An '
    'outlier is a sample that is unlike the others, not necessarily a wrong one: it may be an error, or the most '
    'interesting sample in the set. Where the data hold distinct groups, the robust method takes the largest '
    'coherent half as the bulk and can mark all of a smaller, different group; that is a statement about the '
    'groups, not about errors. Nothing is removed unless you choose to.')

# %% statistics with values below the LOR
CENSORED_COLS = ['Variable', 'n', 'Below LOR', 'Below LOR (%)', 'Method', 'Mean', 'SD', 'P25', 'Median', 'P75', 'Note']


def _ros(values, censored):
    # regression on order statistics (Helsel and Cohn, 1988). values: the measurement, or the LOR where censored
    # returns the data with the censored values replaced by estimates from a lognormal fit to the detected ones
    x = np.asarray(values, float)
    c = np.asarray(censored, bool)
    limits = np.unique(x[c])
    if (~c).any() and x[~c].min() < limits.min():
        limits = np.r_[0.0, limits]
    edges = np.r_[limits, np.inf]
    A = np.array([((~c) & (x >= edges[j]) & (x < edges[j + 1])).sum() for j in range(len(limits))])
    B = np.array([(c & (x <= edges[j])).sum() + ((~c) & (x < edges[j])).sum() for j in range(len(limits))])
    C = np.array([(c & (x == edges[j])).sum() for j in range(len(limits))])
    pe = np.zeros(len(limits) + 1)            # probability of exceeding each limit
    for j in range(len(limits) - 1, -1, -1):
        pe[j] = pe[j + 1] + (A[j] / (A[j] + B[j]) if (A[j] + B[j]) else 0.0) * (1 - pe[j + 1])
    pp = np.empty(len(x))
    for j in range(len(limits)):
        det = np.where((~c) & (x >= edges[j]) & (x < edges[j + 1]))[0]
        det = det[np.argsort(x[det], kind='stable')]
        pp[det] = (1 - pe[j]) + (pe[j] - pe[j + 1]) * np.arange(1, len(det) + 1) / (len(det) + 1)
        cen = np.where(c & (x == edges[j]))[0]
        pp[cen] = (1 - pe[j]) * np.arange(1, len(cen) + 1) / (len(cen) + 1)
    slope, intercept = np.polyfit(stats.norm.ppf(pp[~c]), np.log(x[~c]), 1)
    out = x.copy()
    out[c] = np.exp(intercept + slope * stats.norm.ppf(pp[c]))
    return out


def _kaplan_meier(values, censored):
    # Kaplan-Meier for left-censored data, by flipping the data so the method for right-censored data
    # applies. Returns (values, probability mass at each). If the lowest values are censored, their mass is
    # put at the lowest LOR (Efron), so the mean is biased high and is flagged
    x = np.asarray(values, float)
    c = np.asarray(censored, bool)
    top = x.max() + 1.0
    y = top - x
    order = np.argsort(y, kind='stable')
    y, event = y[order], ~c[order]
    times, mass, surv = [], [], 1.0
    for t in np.unique(y[event]):
        d = (event & (y == t)).sum()
        at_risk = (y >= t).sum()
        new = surv * (1 - d / at_risk)
        times.append(t)
        mass.append(surv - new)
        surv = new
    if surv > 1e-12:
        times.append(y.max())
        mass.append(surv)
    vals = top - np.array(times)
    order = np.argsort(vals)
    return vals[order], np.array(mass)[order]


def _weighted(vals, mass):
    mean = float(np.sum(vals * mass))
    cum = np.cumsum(mass)
    q = lambda p: float(vals[min(np.searchsorted(cum, p - 1e-12), len(vals) - 1)])
    return mean, float(np.sqrt(np.sum(mass * (vals - mean) ** 2))), q(0.25), q(0.5), q(0.75)


def censored_stats(df, below, cols=None):
    # summary statistics for the variables that have values below the LOR, three ways:
    #   substitution (the value used everywhere else in the app), ROS and Kaplan-Meier
    # df: the cleaned input table (process()['base']['df']); below: {(row, column): LOR}
    # values are in the unit they were entered in
    have = sorted({col for _, col in below}, key=lambda c: list(df.columns).index(c) if c in df.columns else 0)
    rows = []
    for col in (have if cols is None else [c for c in cols if c in have]):
        x = pd.to_numeric(df[col], errors='coerce')
        ok = x.notna()
        cen = pd.Series([(i, col) in below for i in df.index], index=df.index) & ok
        lor = pd.Series({i: below.get((i, col), np.nan) for i in df.index})
        n, n_c = int(ok.sum()), int(cen.sum())
        head = {'Variable': col, 'n': n, 'Below LOR': n_c, 'Below LOR (%)': 100 * n_c / n if n else np.nan}
        used = x[ok]
        rows.append(dict(head, Method='Substitution (as used in the app)', Mean=used.mean(),
                         SD=used.std(ddof=1) if n > 1 else np.nan, P25=used.quantile(0.25), Median=used.median(),
                         P75=used.quantile(0.75), Note=''))
        vals = np.where(cen[ok], lor[ok], x[ok]).astype(float)
        flags = cen[ok].to_numpy()
        n_det = int((~flags).sum())
        if n_det < 3 or n_c / n > 0.8 or len(np.unique(vals[~flags])) < 2 or (vals <= 0).any():
            note = 'Not estimated: needs at least 3 detected values, no more than 80 % below the LOR, and positive values.'
            rows.append(dict(head, Method='ROS', Note=note))
            rows.append(dict(head, Method='Kaplan-Meier', Note=note))
            continue
        est = pd.Series(_ros(vals, flags))
        rows.append(dict(head, Method='ROS', Mean=est.mean(), SD=est.std(ddof=1), P25=est.quantile(0.25),
                         Median=est.median(), P75=est.quantile(0.75), Note=''))
        km_vals, km_mass = _kaplan_meier(vals, flags)
        mean, sd, p25, med, p75 = _weighted(km_vals, km_mass)
        lowest_censored = bool(flags[np.argmin(vals)])
        note = ('The lowest values are below the LOR: the mean is biased high, and percentiles at or below the '
                'lowest LOR are that LOR.') if lowest_censored else ''
        rows.append(dict(head, Method='Kaplan-Meier', Mean=mean, SD=sd, P25=p25, Median=med, P75=p75, Note=note))
    return pd.DataFrame(rows, columns=CENSORED_COLS)


CENSORED_TEXT = (
    'Replacing values below the LOR with a fraction of the LOR is simple but distorts means and, above all, '
    'standard deviations when many values are below it (Helsel, 2012). Two alternatives are given for each '
    'variable that has such values. ROS (regression on order statistics; Helsel and Cohn, 1988) fits a lognormal '
    'distribution to the detected values and uses it to estimate those below the LOR, so it suits small '
    'datasets and several different LORs. Kaplan-Meier is non-parametric and assumes no distribution, but its '
    'mean is biased high when the lowest values are below the LOR. Where the three agree, substitution did no '
    'harm; where they differ, prefer ROS or Kaplan-Meier. Values are in the unit they were entered in (NO3 as N '
    'if that is how it was entered). The rest of the app, including the plots and PHREEQC, still uses '
    'substitution.')

# %% CCME water quality index
CCME_CLASSES = [(95, 'Excellent'), (80, 'Good'), (65, 'Fair'), (45, 'Marginal'), (0, 'Poor')]
CCME_COLS = ['Group', 'Samples', 'Variables', 'Tests', 'Failed variables', 'Failed tests', 'F1 (scope)',
             'F2 (frequency)', 'F3 (amplitude)', 'CCME WQI', 'Category', 'Meets CCME minimum']


def _ccme_tests(v, rows):
    # for each guideline value that is a single limit or a range: (name, assessed, failed, excursion)
    hardness = v['Hardness (mg/L as CaCO3)'].astype(float)
    tests = {}
    for g in rows:
        if g['kind'] not in ('max', 'range') or g['var'] not in v.columns:
            continue
        x = v[g['var']].astype(float)
        applies = x.notna()
        if 'hard' in g:
            applies &= hardness.notna() & (hardness >= g['hard'][0]) & (hardness < g['hard'][1])
        exc = pd.Series(0.0, index=v.index)
        if g['kind'] == 'max':
            if g['value'] <= 0:
                continue
            over = applies & (x > g['value'])
            exc[over] = x[over] / g['value'] - 1
        else:
            high = applies & (x > g['high'])
            low = applies & (x < g['low']) & (x > 0)
            exc[high] = x[high] / g['high'] - 1
            exc[low] = g['low'] / x[low] - 1
            over = high | (applies & (x < g['low']))
        # hardness bands of one parameter are one variable
        a, f, e = tests.get(g['var'], (False, False, 0.0))
        tests[g['var']] = (a | applies, f | over, e + exc)
    return tests


def ccme_wqi(v, rows, by_group=True):
    # CCME Water Quality Index 1.0 (CCME, 2001; 2017) for each Label, or for all samples when there is one
    # Label (or by_group=False). There is no pooled row alongside the groups: F1 counts a variable as failed
    # if it fails in any sample, so pooling the groups gives a lower score than any of the groups would
    # suggest, and the pooled value is not comparable with theirs.
    # rows: guideline values (from guidelines.SETS or guidelines.regional_rows)
    if by_group and v['Label'].nunique() > 1:
        groups = [(lab, v[v['Label'] == lab]) for lab in pd.unique(v['Label'])]
    else:
        groups = [(chem.ALL_SAMPLES, v)]
    out = []
    for name, g in groups:
        tests = {k: t for k, t in _ccme_tests(g, rows).items() if t[0].any()}
        n_var = len(tests)
        n_tests = int(sum(t[0].sum() for t in tests.values()))
        row = {'Group': name, 'Samples': len(g), 'Variables': n_var, 'Tests': n_tests}
        if n_tests:
            failed_var = sum(bool(t[1].any()) for t in tests.values())
            failed = int(sum(t[1].sum() for t in tests.values()))
            nse = float(sum(t[2].sum() for t in tests.values())) / n_tests
            f1, f2, f3 = 100 * failed_var / n_var, 100 * failed / n_tests, nse / (0.01 * nse + 0.01)
            wqi = 100 - np.sqrt(f1 ** 2 + f2 ** 2 + f3 ** 2) / 1.732
            row.update({'Failed variables': failed_var, 'Failed tests': failed, 'F1 (scope)': f1,
                        'F2 (frequency)': f2, 'F3 (amplitude)': f3, 'CCME WQI': wqi,
                        'Category': next(c for lim, c in CCME_CLASSES if wqi >= lim),
                        'Meets CCME minimum': n_var >= 4 and len(g) >= 4})
        out.append(row)
    return pd.DataFrame(out, columns=CCME_COLS)


CCME_TEXT = (
    'The CCME Water Quality Index (Canadian Council of Ministers of the Environment, 2001; user manual 2017) '
    'turns a set of guideline comparisons into one number from 0 (worst) to 100 (best). It combines scope (F1, '
    'the share of variables with at least one failure), frequency (F2, the share of individual tests that fail) '
    'and amplitude (F3, how far the failures are from the guideline): WQI = 100 - sqrt(F1^2 + F2^2 + F3^2) / '
    '1.732. Categories: Excellent 95 to 100, Good 80 to 94, Fair 65 to 79, Marginal 45 to 64, Poor below 45.\n\n'
    'It was designed for repeated sampling of one site, with at least four variables and four sampling visits '
    '(CCME recommends eight or more variables and no more than twenty). Here each group (Label) is treated as '
    'the site and its samples as the visits, which is the common way it is applied to groundwater surveys but '
    'is not what it was built for. With more than one group, no index is given for all samples pooled: F1 '
    'counts a variable as failed if it fails in any one sample, so pooling groups (and adding samples) lowers '
    'the score, and the pooled value is not comparable with the group values.\n\n'
    'Views on such indices are divided. In their favour: one number is easy to communicate and to compare. '
    'Against: the score depends on which variables and which guideline set are chosen, so two studies are '
    'rarely comparable; F1 penalises a variable equally for one failure or fifty; a serious exceedance of one '
    'toxicant can be diluted by many harmless passes; correlated variables are counted more than once; and '
    'values near a category boundary carry no uncertainty. CCME itself says the index supplements a '
    'parameter-by-parameter assessment and does not replace one. Only guideline values that are a single limit '
    'or a range are used; graded guidelines (FAO irrigation) are left out.')

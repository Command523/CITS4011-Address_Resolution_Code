"""Metric functions shared by 5_baselines.ipynb and 6_train_canine_reranker.ipynb.

Pure functions over arrays, with no file paths and no randomness except the
seeded fold split in cv_accuracy.

Names used below:
  score      the method's score for a query's top-1 candidate (higher = more alike)
  correct    True when that top-1 candidate renders to the true record's string
  matchable  True when the query has a true record (789 of the 950)
  accept     score >= threshold: the candidate is returned as the match
  reject     score <  threshold: the address is deemed unmatchable
  accuracy   (correct accepts + correct rejects) / all queries
"""
import numpy as np


def confusion(score, correct, matchable, t):
    """The five-way outcome count at threshold t, with coverage, precision,
    recall, specificity, F1 and accuracy."""
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    matchable = np.asarray(matchable, dtype=bool)
    a = score >= t
    ca = int((a & matchable & correct).sum())
    wa_wrong = int((a & matchable & ~correct).sum())
    wa_unm = int((a & ~matchable).sum())
    rm = int((~a & matchable).sum())
    cr = int((~a & ~matchable).sum())
    n = len(score)
    assert ca + wa_wrong + wa_unm + rm + cr == n, "confusion counts do not sum to n"
    prec = ca / max(ca + wa_wrong + wa_unm, 1)
    rec = ca / max(int(matchable.sum()), 1)
    spec = cr / max(int((~matchable).sum()), 1)
    return dict(threshold=float(t), n=n,
                correct_accept=ca, wrong_accept_wrong_record=wa_wrong,
                wrong_accept_unmatchable=wa_unm, rejected_matchable=rm,
                correct_reject=cr, accurate=ca + cr, accuracy=(ca + cr) / n,
                coverage=float(a.mean()), precision=prec, recall=rec,
                specificity=spec, f1=2 * prec * rec / max(prec + rec, 1e-9))


def best_threshold(score, correct, matchable):
    """The threshold that maximises accuracy over these rows.

    Candidates: every distinct score (accept everything at or above it) and
    'reject everything' (+inf). Ties go to the highest threshold, which
    accepts the fewest addresses. Returns (threshold, number of rows right at it)."""
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    matchable = np.asarray(matchable, dtype=bool)
    order = np.argsort(score, kind="stable")
    s = score[order]
    good = (matchable & correct)[order]        # a right answer, worth accepting
    unm = (~matchable)[order]                  # no answer, worth rejecting
    suf_good = np.cumsum(good[::-1])[::-1]     # suf_good[i] = good[i:].sum()
    pre_unm = np.concatenate([[0], np.cumsum(unm)])   # pre_unm[i] = unm[:i].sum()
    first = np.concatenate([[True], s[1:] != s[:-1]])  # first row of each distinct score
    idx = np.flatnonzero(first)
    cand_t = np.concatenate([s[idx], [np.inf]])
    cand_acc = np.concatenate([suf_good[idx] + pre_unm[idx], [int(unm.sum())]])
    best = int(cand_acc.max())
    j = np.flatnonzero(cand_acc == best)[-1]
    return float(cand_t[j]), best


def cv_accuracy(score, correct, matchable, strata, seed=4640, n_splits=5):
    """Five-fold cross-validated accuracy: the threshold is chosen on four
    fifths of the rows (best_threshold) and scored on the remaining fifth,
    once per fold, so every row is scored by a threshold that never saw it.
    strata: a label per row used to keep the folds balanced (population)."""
    from sklearn.model_selection import StratifiedKFold
    score = np.asarray(score, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    matchable = np.asarray(matchable, dtype=bool)
    strata = np.asarray(strata)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    total, thresholds, fold_acc, fold_n = 0, [], [], []
    for tr, te in skf.split(np.zeros(len(score)), strata):
        t, _ = best_threshold(score[tr], correct[tr], matchable[tr])
        c = confusion(score[te], correct[te], matchable[te], t)
        total += c["accurate"]; thresholds.append(t)
        fold_acc.append(c["accurate"]); fold_n.append(len(te))
    assert sum(fold_n) == len(score)
    return dict(accurate=int(total), accuracy=total / len(score),
                thresholds=thresholds, fold_accurate=fold_acc, fold_n=fold_n)


def auroc(pos, neg):
    """Threshold-free separation of two score sets (rank-based, ties averaged)."""
    import pandas as pd
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    x = np.concatenate([pos, neg]); r = pd.Series(x).rank().to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def recall_at(perquery, truth_str, uids, k):
    """Count of uids whose true string appears within the first k ranked
    candidates. perquery: DataFrame with uid, rank, cand. truth_str: uid -> str."""
    sub = perquery[perquery["rank"] <= k]
    hit = 0
    for u, g in sub.groupby("uid"):
        if u in truth_str and (g["cand"].to_numpy() == truth_str[u]).any():
            hit += 1
    return int(hit)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

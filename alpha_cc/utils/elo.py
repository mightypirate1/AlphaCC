from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_ELO_SCALE = 400.0 / np.log(10.0)


@dataclass
class EloFit:
    channels: list[int]
    ratings: dict[int, float]
    stderr: dict[int, float]
    white_advantage: float
    white_advantage_stderr: float
    n_games: int
    cov_ratings: np.ndarray  # (n, n), Elo² — full covariance of the rating block
    _idx: dict[int, int]

    def ranked(self) -> list[tuple[int, float, float]]:
        return sorted(
            ((c, self.ratings[c], self.stderr[c]) for c in self.channels),
            key=lambda x: -x[1],
        )

    def diff(self, a: int, b: int) -> tuple[float, float]:
        """Returns ``(r_a - r_b, stderr)`` in Elo, using the joint covariance."""
        i, j = self._idx[a], self._idx[b]
        var = self.cov_ratings[i, i] + self.cov_ratings[j, j] - 2.0 * self.cov_ratings[i, j]
        return self.ratings[a] - self.ratings[b], float(np.sqrt(max(var, 0.0)))


def fit_elo(
    channels: list[int],
    games: list[tuple[int, int, int]],
    *,
    prior_sigma: float = 400.0,
) -> EloFit:
    """Fit Bradley-Terry ratings plus a white-advantage term to tournament games.

    Each game is ``(white_channel, black_channel, winner)`` with ``winner``
    in ``{1: white, 2: black, 0: draw}``. Draws are split 0.5/0.5 — robust,
    pools information without a separate draw-propensity parameter (Rao-Kupper
    would be more principled but adds a constrained parameter for little gain
    unless the draw rate is very high).

    Returns ratings in Elo units, anchored so ``sum(r) = 0``. ``prior_sigma``
    is a weak Gaussian prior on ratings (in Elo) — keeps perfect records
    and disconnected sub-tournaments finite.
    """
    if len(channels) < 2:
        raise ValueError("need at least 2 channels")
    if not games:
        raise ValueError("no games to fit")

    idx = {c: i for i, c in enumerate(channels)}
    n = len(channels)

    score_of = {1: 1.0, 2: 0.0, 0: 0.5}
    iw = np.asarray([idx[w] for w, _, _ in games], dtype=np.int64)
    ib = np.asarray([idx[b] for _, b, _ in games], dtype=np.int64)
    s = np.asarray([score_of[r] for _, _, r in games], dtype=np.float64)

    prior_sigma_log = prior_sigma / _ELO_SCALE
    prior_prec = 1.0 / prior_sigma_log**2

    def neg_log_post(x: np.ndarray) -> float:
        r, w = x[:n], x[n]
        z = r[iw] - r[ib] + w
        nll = float((np.logaddexp(0.0, z) - s * z).sum())
        nll += 0.5 * float(r @ r) * prior_prec
        return nll

    def grad(x: np.ndarray) -> np.ndarray:
        r, w = x[:n], x[n]
        z = r[iw] - r[ib] + w
        err = 1.0 / (1.0 + np.exp(-z)) - s
        g = np.zeros_like(x)
        np.add.at(g, iw, err)
        np.add.at(g, ib, -err)
        g[n] = float(err.sum())
        g[:n] += r * prior_prec
        return g

    res = minimize(neg_log_post, np.zeros(n + 1), jac=grad, method="L-BFGS-B")
    params = res.x.copy()
    params[:n] -= params[:n].mean()  # exact gauge fix (prior already near-anchors)

    r, w = params[:n], params[n]
    z = r[iw] - r[ib] + w
    p = 1.0 / (1.0 + np.exp(-z))
    pq = p * (1.0 - p)

    # Hessian of -log posterior. Each game contributes pq[k] * v_k v_k^T where
    # v_k = e_{iw} - e_{ib} + e_n. Diagonal + all off-diagonal pairs below.
    H = np.zeros((n + 1, n + 1))
    np.add.at(H, (iw, iw), pq)
    np.add.at(H, (ib, ib), pq)
    H[n, n] += float(pq.sum())
    np.add.at(H, (iw, ib), -pq)
    np.add.at(H, (ib, iw), -pq)
    n_idx = np.full_like(iw, n)
    np.add.at(H, (iw, n_idx), pq)
    np.add.at(H, (n_idx, iw), pq)
    np.add.at(H, (ib, n_idx), -pq)
    np.add.at(H, (n_idx, ib), -pq)
    H[np.arange(n), np.arange(n)] += prior_prec

    cov = np.linalg.pinv(H) * _ELO_SCALE**2  # convert to Elo² units
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    ratings = {channels[i]: float(params[i] * _ELO_SCALE) for i in range(n)}
    stderr = {channels[i]: float(se[i]) for i in range(n)}

    return EloFit(
        channels=list(channels),
        ratings=ratings,
        stderr=stderr,
        white_advantage=float(params[n] * _ELO_SCALE),
        white_advantage_stderr=float(se[n]),
        n_games=len(games),
        cov_ratings=cov[:n, :n].copy(),
        _idx=dict(idx),
    )

"""Evaluate a smoothed delivery model against logistic regression."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path(__file__).parent / "ipl_chase_deliveries.csv"
PLOT = Path(__file__).parent / "reliability.svg"
OUTCOMES = (0, 1, 2, 3, 4, 6, "wicket")


def run_class(runs, wicket):
    if wicket:
        return "wicket"
    # Assign rare run totals to the closest listed total; ties go lower.
    return min((0, 1, 2, 3, 4, 6), key=lambda value: (abs(value - runs), value))


def run_bucket(value):
    value = int(value)
    if value < 40:
        return max(1, ((value - 1) // 2) * 2 + 1)
    return 40 + ((value - 40) // 5) * 5


def coarse_key(balls, wickets):
    return ((int(balls) + 5) // 6, int(wickets))


def logistic_fit(x, y):
    """Small L2-regularized logistic regression fit with Newton steps."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + [1.0] * x.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(design @ beta, -30, 30)))
        weight = np.maximum(p * (1 - p), 1e-8)
        grad = design.T @ (p - y) + penalty @ beta
        hessian = (design.T * weight) @ design + penalty
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return mean, scale, beta


def reliability_svg(labels, predictions, baseline, path):
    """Write a dependency-free reliability plot with ten probability bins."""
    width, height, pad = 680, 480, 55
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def points(probabilities):
        result = []
        edges = np.linspace(0, 1, 11)
        for index in range(10):
            mask = ((probabilities >= edges[index]) &
                    (probabilities <= edges[index + 1] if index == 9
                     else probabilities < edges[index + 1]))
            if mask.any():
                result.append((float(probabilities[mask].mean()),
                               float(labels[mask].mean())))
        return " ".join(
            f"{pad + x * plot_w:.1f},{height - pad - y * plot_h:.1f}"
            for x, y in result
        )

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{pad}" stroke="#aaa" stroke-dasharray="5,5"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="black"/>',
           f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="black"/>',
           f'<polyline points="{points(predictions)}" fill="none" stroke="#1769aa" stroke-width="3"/>',
           f'<polyline points="{points(baseline)}" fill="none" stroke="#e87500" stroke-width="3"/>']
    for value in range(0, 11, 2):
        x = pad + value / 10 * plot_w
        y = height - pad - value / 10 * plot_h
        svg += [f'<text x="{x}" y="{height-pad+22}" text-anchor="middle" font-size="12">{value/10:.1f}</text>',
                f'<text x="{pad-10}" y="{y+4}" text-anchor="end" font-size="12">{value/10:.1f}</text>']
    svg += [f'<text x="{width/2}" y="{height-10}" text-anchor="middle">Mean predicted win probability</text>',
            f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})">Observed win rate</text>',
            '<text x="390" y="35" fill="#1769aa">Smoothed DP</text>',
            '<text x="500" y="35" fill="#e87500">Logistic</text>', '</svg>']
    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    df = pd.read_csv(DATA)
    df["season"] = pd.to_numeric(df["season"])
    train = df[df.season <= 2023].copy()
    test = df[df.season >= 2024].copy()
    if train.empty or test.empty:
        raise SystemExit("Need training rows through 2023 and test rows from 2024 onward.")

    # Recover legal-ball use from the change in balls_left to the next delivery.
    next_left = train.groupby("match_id")["balls_left"].shift(-1)
    train["legal"] = (next_left.isna() | (next_left < train["balls_left"])).astype(int)
    train["runs_bucket"] = train.runs_needed_before_ball.map(run_bucket)
    test["runs_bucket"] = test.runs_needed_before_ball.map(run_bucket)
    train["outcome"] = [run_class(r, w) for r, w in
                         zip(train.runs_off_ball, train.wicket_on_ball)]

    # Exact-state and coarse-state counts support the requested 30-row backoff.
    exact, coarse = defaultdict(Counter), defaultdict(Counter)
    exact_legal, coarse_legal = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for row in train.itertuples():
        state = (int(row.balls_left), int(row.wickets_lost), int(row.runs_bucket))
        broad = coarse_key(row.balls_left, row.wickets_lost)
        exact[state][row.outcome] += 1
        coarse[broad][row.outcome] += 1
        exact_legal[(state, row.outcome)][0] += int(row.legal)
        exact_legal[(state, row.outcome)][1] += 1
        coarse_legal[(broad, row.outcome)][0] += int(row.legal)
        coarse_legal[(broad, row.outcome)][1] += 1

    def outcome_probs(balls, wickets, needed):
        state = (balls, wickets, run_bucket(needed))
        counts = exact[state] if sum(exact[state].values()) >= 30 else coarse[coarse_key(balls, wickets)]
        total = sum(counts.values()) + len(OUTCOMES)
        return {label: (counts[label] + 1) / total for label in OUTCOMES}

    def legal_prob(balls, wickets, needed, outcome):
        state = (balls, wickets, run_bucket(needed))
        key = (state, outcome)
        broad = (coarse_key(balls, wickets), outcome)
        legal, count = exact_legal[key]
        if count < 30:
            legal, count = coarse_legal[broad]
        return (legal + 1) / (count + 2)

    # Cache probability lookups: run buckets affect estimates, not DP states.
    outcome_cache, legal_cache = {}, {}

    def cached_outcome_probs(balls, wickets, needed):
        key = (balls, wickets, run_bucket(needed))
        if key not in outcome_cache:
            outcome_cache[key] = outcome_probs(balls, wickets, needed)
        return outcome_cache[key]

    def cached_legal_prob(balls, wickets, needed, outcome):
        key = (balls, wickets, run_bucket(needed), outcome)
        if key not in legal_cache:
            legal_cache[key] = legal_prob(balls, wickets, needed, outcome)
        return legal_cache[key]

    # Value iteration within each balls-left layer solves illegal-dot self loops.
    max_balls = int(max(train.balls_left.max(), test.balls_left.max()))
    max_runs = int(max(train.runs_needed_before_ball.max(),
                       test.runs_needed_before_ball.max()))
    values = {}
    for balls in range(max_balls + 1):
        for needed in range(1, max_runs + 1):
            for wickets in range(9, -1, -1):
                state = (balls, wickets, needed)
                if balls == 0 or wickets >= 10:
                    values[state] = 0.0
                    continue
                probs = cached_outcome_probs(balls, wickets, needed)
                total_value, self_loop = 0.0, 0.0
                for outcome, probability in probs.items():
                    next_need = needed if outcome == "wicket" else needed - outcome
                    next_wickets = wickets + int(outcome == "wicket")
                    legal = cached_legal_prob(balls, wickets, needed, outcome)
                    for consumed, chance in ((1, legal), (0, 1 - legal)):
                        mass = probability * chance
                        if next_need <= 0:
                            total_value += mass
                        elif consumed == 0 and next_need == needed and next_wickets == wickets:
                            self_loop += mass
                        else:
                            next_state = (balls - consumed, next_wickets, next_need)
                            total_value += mass * values.get(next_state, 0.0)
                values[state] = total_value / max(1 - self_loop, 1e-12)

    dp = np.array([values.get((int(r.balls_left), int(r.wickets_lost),
                               int(r.runs_needed_before_ball)), 0.0)
                   for r in test.itertuples()])
    train["required_rate"] = train.runs_needed_before_ball / train.balls_left.clip(lower=1)
    test["required_rate"] = test.runs_needed_before_ball / test.balls_left.clip(lower=1)
    features = ["balls_left", "wickets_lost", "runs_needed_before_ball", "required_rate"]
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    y_train = train.chasing_team_won.to_numpy(float)
    y_test = test.chasing_team_won.to_numpy(float)
    mean, scale, beta = logistic_fit(x_train, y_train)
    design = np.column_stack((np.ones(len(x_test)), (x_test - mean) / scale))
    logistic = 1 / (1 + np.exp(-np.clip(design @ beta, -30, 30)))

    print(f"P(win) at (60, 3, 1):   {values.get((60, 3, 1), 0.0):.6f}")
    print(f"P(win) at (6, 5, 100):  {values.get((6, 5, 100), 0.0):.6f}")
    print(f"P(win) at (120, 0, 170): {values.get((120, 0, 170), 0.0):.6f}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}; test seasons: {sorted(test.season.unique())}")
    print(f"Brier score, smoothed DP: {np.mean((dp - y_test) ** 2):.6f}")
    print(f"Brier score, logistic:    {np.mean((logistic - y_test) ** 2):.6f}")
    reliability_svg(y_test, dp, logistic, PLOT)
    print(f"Reliability plot: {PLOT}")


if __name__ == "__main__":
    main()

"""Create one row per legal-or-illegal delivery in each eligible IPL chase."""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "ipl_chase_deliveries.csv"

COLUMNS = ["match_id", "season", "over", "ball", "runs_needed_before_ball",
           "balls_left", "wickets_lost", "runs_off_ball", "wicket_on_ball",
           "chasing_team_won"]


def innings_total(innings):
    total = sum(d.get("runs", {}).get("total", 0)
                for o in innings.get("overs", []) for d in o.get("deliveries", []))
    pen = innings.get("penalty_runs", {})
    return total + pen.get("pre", 0) + pen.get("post", 0)


def is_legal(delivery):
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def dismissals(delivery):
    # Retired hurt does not count as a wicket lost.
    return [w for w in delivery.get("wickets", []) if w.get("kind") != "retired hurt"]


def season_year(season):
    return int(str(season)[:4])


def parse_match(path):
    with path.open(encoding="utf-8") as f:
        match = json.load(f)

    info = match.get("info", {})
    outcome = info.get("outcome", {})
    innings = match.get("innings", [])

    if outcome.get("result") == "no result":
        return [], "no_result"
    if any(i.get("super_over", False) for i in innings):
        return [], "super_over"
    if len(innings) < 2:
        return [], "fewer_than_2_innings"
    if outcome.get("winner") is None:
        return [], "no_winner_tie"

    first, chase = innings[0], innings[1]
    target = innings_total(first) + 1

    chase_target = chase.get("target", {})
    full_overs = info.get("overs", 20)
    try:
        reduced = ("overs" in chase_target
                   and float(chase_target["overs"]) != float(full_overs))
    except (TypeError, ValueError):
        reduced = True
    if (outcome.get("method") in {"D/L", "VJD"}
            or reduced
            or ("runs" in chase_target and chase_target["runs"] != target)):
        return [], "revised_target_or_reduced_overs"

    chasing_team_won = int(outcome["winner"] == chase.get("team"))
    total_legal = full_overs * info.get("balls_per_over", 6)

    rows, used, scored, wkts = [], 0, 0, 0
    for over_data in chase.get("overs", []):
        legal_in_over = 0
        for d in over_data.get("deliveries", []):
            runs = d.get("runs", {}).get("total", 0)
            out = dismissals(d)
            rows.append({
                "match_id": path.stem,
                "season": season_year(info.get("season")),
                "over": over_data.get("over", 0),
                "ball": legal_in_over + 1,
                "runs_needed_before_ball": target - scored,
                "balls_left": max(total_legal - used, 0),
                "wickets_lost": wkts,
                "runs_off_ball": runs,
                "wicket_on_ball": int(bool(out)),
                "chasing_team_won": chasing_team_won,
            })
            scored += runs
            wkts += len(out)
            if is_legal(d):
                used += 1
                legal_in_over += 1
    return rows, None


def main():
    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")
    all_rows, excluded, kept = [], {}, 0
    for path in sorted(DATA_DIR.glob("*.json")):
        rows, reason = parse_match(path)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            kept += 1
            all_rows.extend(rows)
    pd.DataFrame(all_rows, columns=COLUMNS).to_csv(OUTPUT_FILE, index=False)
    print(f"Matches kept: {kept}, delivery rows: {len(all_rows)}")
    print("Excluded:", excluded)


if __name__ == "__main__":
    main()

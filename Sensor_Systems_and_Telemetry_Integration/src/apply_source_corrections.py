"""Apply the documented corrections in config/source_corrections.json.

Only processed tables and derived exports are changed; the original NASA PSI
tables are never touched. Each cell is changed only if it still holds the
documented old value, so the script is safe to re-run. Every corrected value
gets a quality flag naming the evidence.

Usage (from the package root):
    python src/apply_source_corrections.py          # apply
    python src/apply_source_corrections.py --check  # report only, exit 1 if anything is still uncorrected
"""
from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "source_corrections.json"


def _num_eq(text, target):
    try:
        return abs(float(text) - target) < 1e-9
    except (TypeError, ValueError):
        return False


def _fmt(value):
    return repr(float(value))


def _add_flag(existing, flag):
    existing = "" if existing is None else str(existing)
    if flag in existing:
        return existing
    return flag if not existing.strip() else f"{existing}; {flag}"


def _lookup(corrections):
    table = {}
    for c in corrections:
        for rid in c["record_ids"]:
            table[rid] = c
    return table


def _correction_for(ids, table):
    for i in ids:
        if i in table:
            return table[i]
    return None


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def apply(check_only=False):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    table = _lookup(cfg["corrections"])
    report = {"changed": [], "already_correct": 0, "pending": []}

    def handle(where, ids, current, setter):
        c = _correction_for(ids, table)
        if c is None:
            return
        if _num_eq(current, c["new_value_mm"]):
            report["already_correct"] += 1
        elif _num_eq(current, c["old_value_mm"]):
            if check_only:
                report["pending"].append(f"{where}: {ids} = {current}")
            else:
                setter(c)
                report["changed"].append(f"{where}: {[i for i in ids if i in table][0]} {current} -> {c['new_value_mm']}")
        else:
            raise SystemExit(f"Unexpected value in {where} for {ids}: {current!r} (expected {c['old_value_mm']} or {c['new_value_mm']})")

    for t in cfg["targets"]:
        path = ROOT / t["file"]
        kind = t["kind"]

        if kind == "source_csv":
            fields, rows = _read_csv(path)
            if t["flag_column"] not in fields:
                fields = fields + [t["flag_column"]]
                for r in rows:
                    r.setdefault(t["flag_column"], "")
            for r in rows:
                def setter(c, r=r):
                    r[t["value_column"]] = _fmt(c["new_value_mm"])
                    r[t["flag_column"]] = _add_flag(r.get(t["flag_column"]), c["quality_flag"])
                handle(t["file"], [r.get(t["id_column"])], r.get(t["value_column"]), setter)
            if not check_only:
                _write_csv(path, fields, rows)

        elif kind in ("observations_json", "observations_csv"):
            if kind == "observations_json":
                rows = json.loads(path.read_text(encoding="utf-8"))
            else:
                fields, rows = _read_csv(path)
            for r in rows:
                if r["variable_id"] not in t["variable_ids"]:
                    continue
                def setter(c, r=r):
                    r["value"] = float(c["new_value_mm"]) if kind == "observations_json" else _fmt(c["new_value_mm"])
                    r["quality_flags"] = _add_flag(r.get("quality_flags"), c["quality_flag"])
                handle(t["file"], [r.get("record_id"), r.get("experiment_id")], r["value"], setter)
            if not check_only:
                if kind == "observations_json":
                    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                else:
                    _write_csv(path, fields, rows)

        elif kind == "packets_json":
            env = json.loads(path.read_text(encoding="utf-8"))
            for p in env["packets"]:
                ids = list(p.get("record_ids") or []) + [p.get("experiment_id")]
                for vid in t["variable_ids"]:
                    if vid not in p.get("readings", {}):
                        continue
                    def setter(c, p=p, vid=vid):
                        p["readings"][vid] = float(c["new_value_mm"])
                        p.setdefault("quality_flags", {})[vid] = c["quality_flag"]
                    handle(t["file"], ids, p["readings"][vid], setter)
            if not check_only:
                path.write_text(json.dumps(env, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

        elif kind == "packets_csv":
            fields, rows = _read_csv(path)
            for r in rows:
                ids = [x.strip(" '\"") for x in (r.get("record_ids") or "").strip("[]").split(",")] + [r.get("experiment_id")]
                for vid in t["variable_ids"]:
                    if vid not in r or r[vid] in ("", None):
                        continue
                    def setter(c, r=r, vid=vid):
                        r[vid] = _fmt(c["new_value_mm"])
                    handle(t["file"], ids, r[vid], setter)
            if not check_only:
                _write_csv(path, fields, rows)
        else:
            raise SystemExit(f"Unknown target kind: {kind}")

    return report


if __name__ == "__main__":
    check = "--check" in sys.argv
    rep = apply(check_only=check)
    print(json.dumps({"changed": len(rep["changed"]), "already_correct": rep["already_correct"],
                      "pending": len(rep["pending"])}, indent=2))
    for line in rep["changed"] + rep["pending"]:
        print("  " + line)
    if check and rep["pending"]:
        raise SystemExit(1)

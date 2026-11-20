"""Assemble the study's object set from the graspability tests (D16, D17): every object, from either library,
that has CoACD collision pieces, a resting pose stable at every shift, and >= 1 usable top-down GSNet grasp at
every yaw with the D435 depth model. Writes results/objects/final_set.csv.

    python scripts/final_set.py
"""
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SOURCES = [("gso", "objects"), ("ycb", "objects_ycb")]


def rows(path):
    return {r["name"]: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}


def main():
    out = []
    for src, d in SOURCES:
        g = rows(os.path.join(ROOT, d, "gtest_noise", "graspability_gsnet_30deg.csv"))
        coacd = rows(os.path.join(ROOT, d, "solid_survey_coacd.csv"))
        vhacd = rows(os.path.join(ROOT, d, "solid_survey.csv"))
        for name, r in sorted(g.items()):
            if r["passes"] != "True":
                continue
            c = coacd[name]
            out.append(dict(source=src, name=name, length_cm=f"{float(c['L']) * 100:.1f}",
                            usable_grasps_min=r["min_ok"], usable_grasps_median=r["median_ok"],
                            settle_mm=f"{float(r['max_settle_mm']):.2f}",
                            core_cm=f"{float(c['core_len']) * 100:.1f}",
                            core_density_8cm_g_cc=f"{float(c['core_density_8']) / 1000:.1f}",
                            shell_g_8cm=f"{float(c['shell_mass_8']) * 1000:.0f}",
                            bulge95_mm_coacd=f"{float(c['bulge95']) * 1000:.1f}",
                            bulge95_mm_vhacd=(f"{float(vhacd[name]['bulge95']) * 1000:.1f}" if name in vhacd else "")))
    path = os.path.join(ROOT, "objects", "final_set.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"  {r['source']} {r['name'][:46]:46s} L {r['length_cm']:>5} cm  grasps/yaw >= {r['usable_grasps_min']:>2}  "
              f"core {r['core_cm']:>4} cm @ {r['core_density_8cm_g_cc']:>4} g/cc  bulge95 {r['bulge95_mm_coacd']:>4} mm "
              f"(V-HACD {r['bulge95_mm_vhacd'] or '-':>4})")
    print(f"{len(out)} objects -> {path}")


if __name__ == "__main__":
    main()

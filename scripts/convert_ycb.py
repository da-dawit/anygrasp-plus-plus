"""Copy the YCB google_16k scans (metres) into the MuJoCo scanned-objects layout so the object scripts read both
sets the same way: <out>/<name>/model.obj, textured.mtl, texture_map.png, plus texture.png (what the scene builder
looks for). Collision pieces come from make_collision.py. YCB: Calli et al., IEEE RAM 2015, CC-BY 4.0.

    python scripts/convert_ycb.py --raw <dir with <name>/google_16k/> --out <models dir>
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done, skipped = 0, []
    for name in sorted(n for n in os.listdir(a.raw) if os.path.isdir(os.path.join(a.raw, n))):
        src = os.path.join(a.raw, name, "google_16k")
        need = ("textured.obj", "textured.mtl", "texture_map.png")
        if not all(os.path.exists(os.path.join(src, f)) for f in need):
            skipped.append(name)
            continue
        dst = os.path.join(a.out, name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, "textured.obj"), os.path.join(dst, "model.obj"))
        for f in ("textured.mtl", "texture_map.png"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        shutil.copy(os.path.join(src, "texture_map.png"), os.path.join(dst, "texture.png"))
        done += 1
    print(f"converted {done} objects; skipped (no textured google_16k scan): {', '.join(skipped) or 'none'}")


if __name__ == "__main__":
    main()

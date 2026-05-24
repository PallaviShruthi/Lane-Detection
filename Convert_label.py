"""
STEP 1 — Run this ONCE before training
Converts your segmentation labels → detection bounding boxes
Works on your existing C:/rf_data dataset
"""
import os

DATASET = "C:/rf_data"
SPLITS  = ["train", "valid", "test"]
total   = 0

for split in SPLITS:
    label_dir = os.path.join(DATASET, split, "labels")
    if not os.path.exists(label_dir):
        print(f"❌ Not found: {label_dir}")
        continue
    count = 0
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        fpath     = os.path.join(label_dir, fname)
        new_lines = []
        with open(fpath, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls    = parts[0]
            coords = list(map(float, parts[1:]))
            if len(coords) < 4:
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            cx = max(0, min(1, (min(xs) + max(xs)) / 2))
            cy = max(0, min(1, (min(ys) + max(ys)) / 2))
            bw = max(0, min(1,  max(xs) - min(xs)))
            bh = max(0, min(1,  max(ys) - min(ys)))
            new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            count += 1
        with open(fpath, "w") as f:
            f.writelines(new_lines)
    print(f"✅ {split} — {count} labels converted")
    total += count

print(f"\n✅ Done! {total} total labels converted")
print("▶  Now run: python train.py")

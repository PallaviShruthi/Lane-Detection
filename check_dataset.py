import os
import yaml

# ─────────────────────────────────────
# Checks your Roboflow dataset is correct
# before training. Same dataset: lane_train_v6-1
# ─────────────────────────────────────

dataset_path = "lane_train_v6-1"

print(f"🔍 Checking dataset at: {dataset_path}/\n")

# Check image + label folders
for folder in ["train", "valid", "test"]:
    images_path = os.path.join(dataset_path, folder, "images")
    labels_path = os.path.join(dataset_path, folder, "labels")

    if os.path.exists(images_path):
        img_count = len(os.listdir(images_path))
        print(f"✅ {folder}/images — {img_count} images found")
    else:
        print(f"❌ {folder}/images — NOT FOUND")

    if os.path.exists(labels_path):
        lbl_count = len(os.listdir(labels_path))
        print(f"✅ {folder}/labels — {lbl_count} labels found")
    else:
        print(f"❌ {folder}/labels — NOT FOUND")

    print()

# Check data.yaml
yaml_path = os.path.join(dataset_path, "data.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    print(f"📌 Classes : {data['names']}")
    print(f"📌 Num classes: {data['nc']}")
    print(f"📌 Train path: {data.get('train', 'not set')}")
    print(f"📌 Val path  : {data.get('val', 'not set')}")

    # ── IMPORTANT: YOLOv8 detection needs nc to match ──
    if data['nc'] == 1 and data['names'] == ['lane']:
        print("\n✅ data.yaml is correct for YOLOv8 detection!")
    else:
        print("\n⚠️  Check your class names — expected: ['lane'] with nc=1")
else:
    print("❌ data.yaml not found")
    print("   Make sure your dataset folder is named exactly: lane_train_v6-1")
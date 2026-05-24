
import cv2, os, glob
VIDEOS_FOLDER = r"C:\Users\DELL\Downloads\my_videos"
OUTPUT_FOLDER = "my_road_frames"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
videos = []
for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
    videos += glob.glob(os.path.join(VIDEOS_FOLDER, ext))
    videos += glob.glob(os.path.join(VIDEOS_FOLDER, "**", ext), recursive=True)
videos = list(set([v for v in videos if not v.endswith('.adding')]))
print(f"Found {len(videos)} videos")
for v in videos: print(f"  {v}")
total_saved = 0
for video_path in videos:
    name = os.path.splitext(os.path.basename(video_path))[0].replace('(','').replace(')','').replace(' ','_')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): continue
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    print(f"Processing: {name}")
    for i in range(0, total, 20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{OUTPUT_FOLDER}/{name}_{saved:04d}.jpg", frame)
            saved += 1
    cap.release()
    total_saved += saved
    print(f"  Saved {saved} frames")
print(f"Done! Total {total_saved} frames")
#"@ | Out-File -FilePath extract_frames.py -Encoding utf8
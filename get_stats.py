import os, glob, shutil, cv2

# Copy 10 videos to normal
shop_vids = glob.glob('D:\\shoplifting\\data\\raw\\shoplifting\\*.mp4')
for i, src in enumerate(shop_vids[:10]):
    shutil.copy(src, os.path.join('D:\\shoplifting\\data\\raw\\normal', f'dummy_normal_{i}.mp4'))

# Get stats
def get_stats(folder):
    vids = glob.glob(os.path.join(folder, '*.mp4'))
    if not vids: return 0, 0, (0,0)
    durations = []
    width, height = 0, 0
    # Process max 50 videos to save time on stats
    for v in vids[:50]:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened(): continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0: durations.append(frames/fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    avg_dur = sum(durations)/len(durations) if durations else 0
    return len(vids), avg_dur, (width, height)

n_shop, d_shop, res_shop = get_stats('D:\\shoplifting\\data\\raw\\shoplifting')
n_norm, d_norm, res_norm = get_stats('D:\\shoplifting\\data\\raw\\normal')

summary = f"""# Dataset Summary

## Shoplifting Class
- **Number of clips**: {n_shop}
- **Average duration (sampled)**: {d_shop:.2f} seconds
- **Resolution**: {res_shop[0]}x{res_shop[1]}

## Normal Class (Dummy Data for Pipeline Testing)
- **Number of clips**: {n_norm}
- **Average duration (sampled)**: {d_norm:.2f} seconds
- **Resolution**: {res_norm[0]}x{res_norm[1]}
"""

with open('D:\\shoplifting\\report\\dataset_summary.md', 'w') as f:
    f.write(summary)

print('Stats generated and saved to report/dataset_summary.md')

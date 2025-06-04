import numpy as np
import pandas as pd
# 验证10秒片段
sample = np.load("./professional_processeddd/10s_segments/04015_10s_20250507214207.npy")
labels = pd.read_csv("./professional_processeddd/10s_segments/labels.csv")
print(f"Segment length: {len(sample)/250:.1f}s")
print(f"Label: {labels.iloc[0].label}")

# 统计各时长AF比例
for duration in [10, 30]:
    df = pd.read_csv(f"./professional_processeddd/{duration}s_segments/labels.csv")
    print(f"{duration}s AF rate: {df.label.mean():.2%}")
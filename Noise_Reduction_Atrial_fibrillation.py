import os
import numpy as np
import pywt  # 小波变换
import matplotlib.pyplot as plt

def wavelet_denoise(signal, wavelet='db4', level=5, mode='soft'):  # Daubechies 4小波基函数、小波分解的层数5、软阈值处理模式
    """小波阈值去噪"""
    # 小波分解：得到各级系数（近似系数 + 细节系数）
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 估计噪声标准差：使用最高层细节系数的绝对中位差(MAD)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    # 计算通用阈值：基于Donoho-Johnstone理论
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    # 对各层系数应用阈值处理
    coeffs = [pywt.threshold(c, uthresh, mode=mode) for c in coeffs]
    # 小波重构：将处理后的系数重构为信号
    return pywt.waverec(coeffs, wavelet)

# 加载降采样后的数据
data = np.load("./processed_ecg/04015_processed.npz")
signal = data['signal']  # 获取ECG信号

# 执行去噪
clean_signal = wavelet_denoise(signal)

# 可视化对比
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(signal, label="Original signal")
plt.title("Downsampled signal")
plt.subplot(2, 1, 2)
plt.plot(clean_signal, label="Noise-reduced signal", color='orange')
plt.title("The signal after wavelet denoising")
plt.tight_layout()    # tight_layout()会自动计算合适的布局参数
plt.show()

def segment_signal(signal, labels, fs=250, window_sec=10): # window_sec=10: 窗口时间，默认每段切割成10秒
    '''信号片段分割————10s'''
    # 计算每个窗口包含多少个数据点
    # 例如：10秒 × 250点/秒 = 2500个点
    window_size = window_sec * fs
    # 计算信号总长度能分成几个完整窗口
    # 例如：信号有7500个点 → 7500//2500=3个完整段
    num_segments = len(signal) // window_size   # 计算完整分段数
    # 创建两个空列表，用于存放切割后的结果
    segments = []         # 存放信号片段
    segment_labels = []   # 存放每个片段对应的标签

    # 开始循环处理每一个分段，循环次数等于总段数
    # 例如：如果总共有3段，i会依次取0,1,2
    for i in range(num_segments):
        # 计算当前段的起始位置 = 当前段号 × 每段长度
        # 第0段起始位置：0 × 2500 = 0
        # 第1段起始位置：1 × 2500 = 2500
        # 第2段起始位置：1 × 2500 = 5000
        # 第3段起始位置：1 × 2500 = 7500
        start = i * window_size
        end = start + window_size
        # 截取信号片段
        seg = signal[start:end]
        # 计算当前段的标签（判断是否为房颤）：
        # 1. 取出当前段对应的所有标签（labels[start:end]）
        # 2. 计算这些标签的平均值（假设标签是0或1，1表示房颤）
        # 3. 如果平均值超过0.5（即超过50%的时间是房颤），标记为1，否则标记为0
        label = 1 if np.mean(labels[start:end]) > 0.5 else 0
        segments.append(seg)     # 将当前信号段添加到结果列表
        segment_labels.append(label)  # 将当前标签添加到结果列表

    return np.array(segments), np.array(segment_labels)


# 加载标签数据
labels = data['labels']

# 切割为10秒片段（得到形状为[N, 2500]的数组）
segments_10s, labels_10s = segment_signal(clean_signal, labels, window_sec=10)
print(f"生成 {len(segments_10s)} 个片段 | 房颤片段数: {sum(labels_10s)}")


def normalize_segments(segments):
    """信号标准化（z-score归一化）"""
    normalized = []   # 创建一个空列表，用于存放处理后的信号片段
    for seg in segments:
        # 1. 计算当前片段的平均值（相当于找这个片段的"中间值"）
        # 2. 计算标准差（衡量数据波动大小的指标，类似"音量波动程度"）
        # 3. 标准化公式：(当前值 - 平均值) / 标准差 → 将数据调整到以0为中心，波动范围为1的尺度
        # 加1e-8（0.00000001）是为了防止标准差为0时出现除以0的错误（比如所有数据完全一样时）
        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)  # 添加极小值防止除零
        normalized.append(seg)
    return np.array(normalized)

# 应用归一化
normalized_segments = normalize_segments(segments_10s)
# 保存处理结果
output_path = "D:/SRDP/吕浩毅/代码实现/pythonProject/final_data/04015_processed_10s.npz"
# np.savez 是numpy库的保存函数，可以同时保存多个数据
np.savez(output_path,
         signals=normalized_segments,
         labels=labels_10s,
         fs=250,
         window_sec=10)
print(f"数据已保存至 {output_path}")
# 设置批量处理的文件夹路径（类似准备两个大箱子）
input_dir = "./processed_ecg"  # 输入文件夹：存放已经预处理好的心电图文件
output_dir = "./final_data"    # 输出文件夹：准备存放最终处理结果的空箱子
# 创建输出文件夹（如果文件夹不存在就新建，存在也不报错）
# 类比：在电脑上右键新建文件夹，如果已经有这个文件夹就什么也不做
# exist_ok=True 是关键参数，保证不会因为文件夹已存在而报错
os.makedirs(output_dir, exist_ok=True)


for filename in os.listdir(input_dir):
    if filename.endswith(".npz"):
        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)
        # 降噪 → 分割 → 归一化
        # 步骤1：用之前写好的函数降噪
        clean_signal = wavelet_denoise(data['signal'])
        # 步骤2：切割成10秒一段的信号
        segments, labels = segment_signal(clean_signal, data['labels'], window_sec=10)
        # 步骤3：标准化处理
        normalized = normalize_segments(segments)
        # 保存处理结果
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_10s.npz")
        np.savez(output_path, signals=normalized, labels=labels)


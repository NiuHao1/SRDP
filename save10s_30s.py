import os
import wfdb
import numpy as np
import pywt
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def wavelet_denoise(signal, wavelet='db6', level=5, threshold_scale=0.8):
    """工业级小波降噪（保留瞬态特征）"""
    coeffs = pywt.wavedec(signal, wavelet, mode='sym', level=level)

    # 改进的自适应噪声估计
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresholds = [sigma * np.sqrt(2 * np.log(len(signal))) * (1 + 0.2 * i)
                  for i in range(1, len(coeffs))]

    # 动态阈值处理
    denoised_coeffs = [coeffs[0]]  # 保留近似系数
    for i in range(1, len(coeffs)):
        threshold = thresholds[i - 1] * threshold_scale
        denoised = pywt.threshold(coeffs[i], threshold, mode='soft')
        denoised_coeffs.append(denoised)

    return pywt.waverec(denoised_coeffs, wavelet, mode='sym')


def create_segment_folder(output_dir, duration):
    """创建分段存储目录结构"""
    segment_dir = os.path.join(output_dir, f"{duration}s_segments")
    os.makedirs(segment_dir, exist_ok=True)

    # 初始化CSV标签文件
    csv_path = os.path.join(segment_dir, "labels.csv")
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=['filename', 'label', 'record', 'original_fs']).to_csv(csv_path, index=False)

    return csv_path


def save_segment(segment, label, record_name, original_fs, duration, csv_path):
    """保存单个分段数据及标签"""
    # 生成唯一文件名
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    filename = f"{record_name}_{duration}s_{timestamp}.npy"

    # 保存numpy文件
    np.save(os.path.join(os.path.dirname(csv_path), filename), segment)

    # 更新CSV文件
    new_row = pd.DataFrame([{
        'filename': filename,
        'label': int(label),
        'record': record_name,
        'original_fs': original_fs
    }])
    new_row.to_csv(csv_path, mode='a', header=False, index=False)


def process_ecg_records(data_dir, target_fs=250,
                        segment_durations=[10, 30],
                        output_dir=None,
                        plot_full_record=True,
                        plot_segments=True):
    """多阶段信号处理流水线（支持多时长）"""
    # 初始化数据结构
    processed_data = {'metadata': []}

    # 获取记录列表（兼容复杂文件名）
    hea_files = [f.split('.')[0].split('_')[0] for f in os.listdir(data_dir)
                 if f.lower().endswith('.hea')]
    unique_records = sorted(list(set(hea_files)))

    # 主处理循环
    for record_name in tqdm(unique_records, desc="Processing Records", unit='rec'):
        try:
            # === 阶段1：数据读取 ===
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            ann = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')
            original_fs = record.fs
            ecg_signal = record.p_signal[:, 0].astype(np.float32)

            # === 阶段2：信号预处理 ===
            # 抗混叠滤波
            cutoff = 0.9 * (target_fs / 2)
            b, a = butter(5, cutoff / (original_fs / 2), 'low')
            filtered = filtfilt(b, a, ecg_signal)

            # 小波降噪
            denoised = wavelet_denoise(filtered)

            # 降采样
            new_length = int(len(denoised) * target_fs / original_fs)
            resampled = resample(denoised, new_length)

            # === 阶段3：多时长分段处理 ===
            for duration in segment_durations:
                samples_per_segment = duration * target_fs
                num_segments = len(resampled) // samples_per_segment

                # 初始化存储路径
                csv_path = create_segment_folder(output_dir, duration)

                # 获取房颤标注时间区间
                af_intervals = []
                for i, note in enumerate(ann.aux_note):
                    if '(AF' in note:
                        start = ann.sample[i]
                        end = ann.sample[i + 1] if i + 1 < len(ann.sample) else len(ecg_signal)
                        af_intervals.append((start, end))

                # 分段处理
                for seg_idx in range(num_segments):
                    start = seg_idx * samples_per_segment
                    end = start + samples_per_segment
                    segment = resampled[start:end]

                    # 零填充处理
                    if len(segment) < samples_per_segment:
                        pad_width = samples_per_segment - len(segment)
                        segment = np.pad(segment, (0, pad_width),
                                         mode='constant', constant_values=0)

                    # 转换到原始时间轴
                    original_start = int(start * original_fs / target_fs)
                    original_end = int(end * original_fs / target_fs)

                    # 判断房颤标签
                    is_af = False
                    for af_start, af_end in af_intervals:
                        if (original_start < af_end) and (original_end > af_start):
                            is_af = True
                            break

                    # 保存分段数据
                    save_segment(segment, is_af, record_name, original_fs, duration, csv_path)

            # === 阶段4：专业级可视化 ===
            if output_dir:
                # [保持原有可视化代码不变]
                pass

            # 存储元数据
            processed_data['metadata'].append({
                'record': record_name,
                'original_fs': original_fs,
                'processed_length': len(resampled),
                'segment_durations': segment_durations
            })

        except Exception as e:
            print(f"\n⚠️ Error processing {record_name}: {str(e)}")
            continue

    # 保存元数据
    if output_dir:
        metadata_path = os.path.join(output_dir, "metadata.csv")
        pd.DataFrame(processed_data['metadata']).to_csv(metadata_path, index=False)

    return processed_data


if __name__ == "__main__":
    # 配置参数
    data_folder = "D:/SRDP/吕浩毅/代码实现/pythonProject/Atrial fibrillation"
    output_folder = "./statistics"

    # 执行处理流程
    ecg_data = process_ecg_records(
        data_dir=data_folder,
        target_fs=250,
        segment_durations=[10, 30],  # 同时处理10秒和30秒片段
        output_dir=output_folder,
        plot_full_record=True,
        plot_segments=True
    )

    # 生成处理报告
    print("\n" + "=" * 40)
    print("ECG Processing Summary".center(40))
    print("=" * 40)
    print(f"Total Records Processed: {len(ecg_data['metadata'])}")
    print(f"Segment Durations: {ecg_data['metadata'][0]['segment_durations']}")
    print(f"Output Directory: {os.path.abspath(output_folder)}")
    print("=" * 40)
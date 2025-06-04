import os
import wfdb
import numpy as np
import pywt
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


def process_ecg_records(data_dir, target_fs=250,
                        segment_duration=10,
                        output_dir=None,
                        plot_full_record=True,
                        plot_segments=True):
    """多阶段信号处理流水线"""
    # 初始化数据结构
    processed_data = {
        'segments': [],
        'labels': [],
        'metadata': [],
        'quality_metrics': []
    }

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
            # 步骤2.1 抗混叠滤波
            cutoff = 0.9 * (target_fs / 2)
            b, a = butter(5, cutoff / (original_fs / 2), 'low')
            filtered = filtfilt(b, a, ecg_signal)

            # 步骤2.2 小波降噪
            denoised = wavelet_denoise(filtered)

            # 步骤2.3 降采样
            new_length = int(len(denoised) * target_fs / original_fs)
            resampled = resample(denoised, new_length)

            # === 阶段3：分段处理 ===
            samples_per_segment = segment_duration * target_fs
            num_segments = len(resampled) // samples_per_segment

            for seg_idx in range(num_segments):
                start = seg_idx * samples_per_segment
                end = start + samples_per_segment
                segment = resampled[start:end]

                # 边缘处理（零填充）
                if len(segment) < samples_per_segment:
                    pad_width = samples_per_segment - len(segment)
                    segment = np.pad(segment, (0, pad_width),
                                     mode='constant', constant_values=0)

                # 标签生成（精确时间映射）
                original_start = int(start * original_fs / target_fs)
                original_end = int(end * original_fs / target_fs)
                af_mask = np.zeros_like(ann.sample, dtype=bool)
                for i, note in enumerate(ann.aux_note):
                    if '(AF' in note:
                        af_mask[i] = True
                is_af = np.any(af_mask & (ann.sample >= original_start) &
                               (ann.sample < original_end))

                # 存储数据
                processed_data['segments'].append(segment)
                processed_data['labels'].append(int(is_af))
                processed_data['metadata'].append({
                    'record': record_name,
                    'segment': seg_idx,
                    'original_fs': original_fs
                })

            # === 阶段4：专业级可视化 ===
            if output_dir:
                plot_dir = os.path.join(output_dir, "advanced_plots")
                os.makedirs(plot_dir, exist_ok=True)

                # 图1：全记录信号处理流水线
                if plot_full_record:
                    plt.figure(figsize=(18, 12))

                    # 原始信号
                    plt.subplot(3, 1, 1)
                    t_orig = np.arange(len(ecg_signal)) / original_fs
                    plt.plot(t_orig, ecg_signal, color='#4B4B8A', alpha=0.7)
                    plt.title(f'Signal Processing - {record_name}\nOriginal Signal ({original_fs}Hz)')
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.xlim(0, 30)

                    # 降噪后信号
                    plt.subplot(3, 1, 2)
                    plt.plot(t_orig, denoised, color='#2E8B57')
                    plt.title('After Wavelet Denoising')
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.xlim(0, 30)

                    # 最终分段（示例显示前3段）
                    plt.subplot(3, 1, 3)
                    t_proc = np.arange(len(resampled)) / target_fs
                    plt.plot(t_proc, resampled, color='#B22222')
                    for seg in range(3):
                        start = seg * samples_per_segment
                        end = start + samples_per_segment
                        plt.axvspan(t_proc[start], t_proc[end],
                                    color='orange', alpha=0.2)
                    plt.title(f'Resampled to {target_fs}Hz with Segmentation')
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.xlim(0, 30)

                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{record_name}_pipeline.png"),
                                dpi=300, bbox_inches='tight')
                    plt.close()

                # 图2：分段信号细节
                if plot_segments and num_segments > 0:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
                    axs = axs.flatten()

                    for plot_idx in range(min(6, num_segments)):
                        seg_idx = plot_idx
                        start = seg_idx * samples_per_segment
                        end = start + samples_per_segment

                        # 获取数据
                        segment = resampled[start:end]
                        label = processed_data['labels'][seg_idx]

                        # 绘制时域信号
                        t_seg = np.linspace(0, segment_duration, len(segment))
                        axs[plot_idx].plot(t_seg, segment,
                                           color='#1F77B4',
                                           linewidth=1.5)

                        # 标注房颤状态
                        axs[plot_idx].add_patch(Rectangle((0, np.min(segment)),
                                                          segment_duration,
                                                          np.ptp(segment),
                                                          alpha=0.2,
                                                          color='red' if label else 'green'))
                        axs[plot_idx].set_title(f'Segment {seg_idx + 1} - {"AF" if label else "Normal"}')
                        axs[plot_idx].grid(True, alpha=0.3)
                        axs[plot_idx].set_xlim(0, segment_duration)

                    plt.suptitle(f'Segmentation Examples - {record_name}', y=0.98)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, f"{record_name}_segments.png"),
                                dpi=300, bbox_inches='tight')
                    plt.close()

        except Exception as e:
            print(f"\n⚠️ Error processing {record_name}: {str(e)}")
            continue

    # 转换为numpy数组
    processed_data['segments'] = np.array(processed_data['segments'])
    processed_data['labels'] = np.array(processed_data['labels'])

    return processed_data


if __name__ == "__main__":
    # 配置参数（根据论文设置）
    data_folder = "D:/SRDP/吕浩毅/代码实现/pythonProject/Atrial fibrillation"
    output_folder = "./professional_processed"

    # 执行处理
    ecg_data = process_ecg_records(
        data_dir=data_folder,
        target_fs=250,
        segment_duration=10,  # 10秒片段
        output_dir=output_folder,
        plot_full_record=True,
        plot_segments=True
    )

    # 打印专业报告
    print("\n" + "=" * 30)
    print("PROFESSIONAL PROCESSING REPORT".center(60))
    print("=" * 30)
    print(f"Total Segments: {len(ecg_data['segments'])}")
    print(f"AF Prevalence: {np.mean(ecg_data['labels']) * 100:.1f}%")
    print(f"Segment Duration: {ecg_data['metadata'][0]['original_fs']}Hz → 10s segments")
    print(f"Output Directory: {os.path.abspath(output_folder)}")
    print("=" * 30)
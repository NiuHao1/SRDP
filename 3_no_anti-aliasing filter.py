import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm


def process_ecg_records(data_dir, target_fs=250, output_dir=None, show_plot=False):
    """带专业可视化的ECG处理函数"""
    # 改进的文件名处理
    hea_files = [f.split('.')[0] for f in os.listdir(data_dir) if f.lower().endswith('.hea')]
    unique_records = sorted(list(set(hea_files)))

    processed_data = {
        'record_names': [],
        'signals': [],
        'labels': [],
        'original_fs': 360,
        'target_fs': target_fs
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, "visualization")
        os.makedirs(plot_dir, exist_ok=True)

    for record_name in tqdm(unique_records, desc="Processing ECG Records"):
        try:
            record_path = os.path.join(data_dir, record_name)

            # 读取数据
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')

            # 信号处理流程
            original_fs = record.fs
            ecg_signal = record.p_signal[:, 0]  # 使用第一导联

            # 抗混叠滤波
            cutoff = 0.9 * target_fs / 2
            nyq = 0.5 * original_fs
            normal_cutoff = cutoff / nyq
            b, a = butter(5, normal_cutoff, btype='low')
            filtered = filtfilt(b, a, ecg_signal)

            # 降采样
            new_length = int(len(filtered) * target_fs / original_fs)
            resampled = resample(filtered, new_length)

            # 标签转换
            labels = np.zeros_like(resampled, dtype=int)
            for i in range(len(ann.sample)):
                if '(AF' in ann.aux_note[i]:
                    start = int(ann.sample[i] * target_fs / original_fs)
                    end = int(ann.sample[i + 1] * target_fs / original_fs) if i + 1 < len(ann.sample) else len(labels)
                    labels[start:end] = 1

            # 存储数据
            processed_data['record_names'].append(record_name)
            processed_data['signals'].append(resampled)
            processed_data['labels'].append(labels)

            # 可视化部分
            if output_dir:
                plt.figure(figsize=(15, 6))
                time = np.arange(len(resampled)) / target_fs

                # 绘制完整信号
                plt.plot(time, resampled,
                         linewidth=0.8,
                         color='#1f77b4',
                         label=f'ECG {target_fs}Hz')

                # 标记房颤区域
                af_starts = np.where(np.diff(labels) == 1)[0] + 1
                af_ends = np.where(np.diff(labels) == -1)[0]

                if len(af_starts) > 0:
                    if len(af_ends) == 0 or af_ends[-1] < af_starts[-1]:
                        af_ends = np.append(af_ends, len(labels))

                    for s, e in zip(af_starts, af_ends):
                        plt.axvspan(time[s], time[e],
                                    color='red', alpha=0.3,
                                    label='AF Episode' if s == af_starts[0] else "")

                # 图像美化
                plt.title(f'ECG Record {record_name}\n[{original_fs}Hz → {target_fs}Hz]')
                plt.xlabel('Time (seconds)', fontsize=12)
                plt.ylabel('Amplitude (mV)', fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.xlim(0, 20)  # 显示前20秒
                plt.legend(loc='upper right')

                # 保存图像
                plt.savefig(os.path.join(plot_dir, f"{record_name}_ecg.png"),
                            dpi=300,
                            bbox_inches='tight')
                plt.close()

            if show_plot:
                plt.show()

        except Exception as e:
            print(f"\nError processing {record_name}: {str(e)}")
            continue

    return processed_data


if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "D:/SRDP/吕浩毅/代码实现/pythonProject/Atrial fibrillation"
    OUTPUT_DIR = "./processed_ecg_未抗混叠滤波0"
    TARGET_FS = 250

    # 执行处理
    ecg_data = process_ecg_records(
        data_dir=DATA_PATH,
        target_fs=TARGET_FS,
        output_dir=OUTPUT_DIR,
        show_plot=False
    )

    # 输出统计信息
    print(f"\n处理完成报告:")
    print(f"成功处理记录数: {len(ecg_data['record_names'])}")
    print(f"采样率转换: {ecg_data['original_fs']}Hz → {ecg_data['target_fs']}Hz")
    print(f"平均信号长度: {np.mean([len(s) for s in ecg_data['signals']]):.1f} 点")
    print(f"房颤占比: {np.mean(np.concatenate(ecg_data['labels'])) * 100:.2f}%")

    # 示例输出
    sample_signal = ecg_data['signals'][0]
    print(f"\n示例记录 ({ecg_data['record_names'][0]}):")
    print(f"总时长: {len(sample_signal) / TARGET_FS:.1f} 秒")
    print(f"信号范围: {np.min(sample_signal):.2f} ~ {np.max(sample_signal):.2f} mV")
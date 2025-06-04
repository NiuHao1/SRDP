import os
import wfdb
import numpy as np
from scipy.signal import resample, butter, filtfilt  # 新增滤波器相关函数
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_ecg_records(data_dir, target_fs=250, output_dir=None, plot_samples=5000):
    # 获取所有记录基础名（兼容带后缀的文件如04043_hr.hea）
    hea_files = [f.split('.')[0].split('_')[0] for f in os.listdir(data_dir) if f.lower().endswith('.hea')]
    unique_records = list(set(hea_files))

    processed_data = {
        'record_names': [],
        'signals': [],
        'labels': [],
        'original_fs': 360
    }

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for record_name in tqdm(unique_records, desc="Processing Records"):
        try:
            record_path = os.path.join(data_dir, record_name)
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')

            # ================== 新增抗混叠滤波部分 ==================
            # 1. 滤波器参数设计
            original_fs = record.fs
            ecg_signal = record.p_signal[:, 0]

            # 计算目标Nyquist频率（保留90%带宽）
            nyquist_ratio = 0.9
            cutoff_freq = nyquist_ratio * (target_fs / 2)

            # 设计5阶巴特沃斯低通滤波器
            norm_cutoff = cutoff_freq / (original_fs / 2)
            b, a = butter(N=5, Wn=norm_cutoff, btype='low')

            # 应用零相位滤波（双向滤波消除延迟）
            filtered_signal = filtfilt(b, a, ecg_signal)
            # =====================================================

            # 2. 执行降采样（使用滤波后信号）
            num_samples = int(len(filtered_signal) * target_fs / original_fs)
            resampled_signal = resample(filtered_signal, num_samples)

            # 3. 生成标签（与原代码相同）
            af_labels = np.zeros_like(resampled_signal, dtype=int)
            for i in range(len(ann.sample)):
                if '(AF' in ann.aux_note[i]:
                    start = int(ann.sample[i] * target_fs / original_fs)
                    end = int(ann.sample[i + 1] * target_fs / original_fs) if i + 1 < len(ann.sample) else len(
                        af_labels)
                    af_labels[start:end] = 1

            # 保存结果
            processed_data['record_names'].append(record_name)
            processed_data['signals'].append(resampled_signal)
            processed_data['labels'].append(af_labels)

            if output_dir:
                np.savez(
                    os.path.join(output_dir, f"{record_name}.npz"),
                    signal=resampled_signal,
                    labels=af_labels,
                    fs=target_fs
                )
            if output_dir:
                # 创建可视化子目录
                plot_dir = os.path.join(output_dir, "plots")
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                # 创建带生理意义的坐标轴
                time_axis = np.arange(len(resampled_signal)) / target_fs  # 转换为秒

                plt.figure(figsize=(15, 6))

                # 绘制完整信号背景
                plt.plot(time_axis, resampled_signal,
                         color='#1f77b4',
                         linewidth=0.8,
                         label=f'ECG ({target_fs}Hz)')

                # 高亮房颤区域
                af_regions = np.where(af_labels == 1)[0]
                if af_regions.size > 0:
                    start_idx = af_regions[0]
                    last_idx = start_idx
                    for idx in af_regions[1:]:
                        if idx != last_idx + 1:
                            plt.axvspan(time_axis[start_idx], time_axis[last_idx],
                                        color='red', alpha=0.3)
                            start_idx = idx
                        last_idx = idx
                    plt.axvspan(time_axis[start_idx], time_axis[last_idx],
                                color='red', alpha=0.3, label='AF Episode')

                # 美观化设置
                plt.xlabel('Time (seconds)', fontsize=12)
                plt.ylabel('Amplitude (mV)', fontsize=12)
                plt.title(f'Processed ECG - Record {record_name}\n'
                          f'Original FS: {original_fs}Hz → Target FS: {target_fs}Hz',
                          fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.xlim(0, plot_samples / target_fs)  # 显示前N秒
                plt.legend(loc='upper right')

                # 保存高分辨率图像
                plt.savefig(os.path.join(plot_dir, f"{record_name}_ecg.png"),
                            dpi=300,
                            bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Error processing {record_name}: {str(e)}")
            continue

    return processed_data


if __name__ == "__main__":
    data_folder = "D:/SRDP/吕浩毅/代码实现/pythonProject/Atrial fibrillation"
    output_folder = "./processed_ecg_抗混叠滤波"
    target_fs = 250

    ecg_data = process_ecg_records(
        data_dir=data_folder,
        target_fs=target_fs,
        output_dir=output_folder,
        plot_samples = 5000
    )


    print(f"\n成功处理 {len(ecg_data['record_names'])} 条记录")
    print(f"示例信号长度：{len(ecg_data['signals'][0])}点")
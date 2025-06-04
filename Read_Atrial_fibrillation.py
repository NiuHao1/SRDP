import os
import wfdb
import numpy as np
from scipy.signal import resample
from tqdm import tqdm  # 进度条工具


def process_ecg_records(data_dir, target_fs=250, output_dir=None):
    # 遍历数据目录，筛选所有以.hea结尾的文件，并去除扩展名（如：100.hea -> 100）
    hea_files = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.hea')]
    unique_records = list(set(hea_files))   # 使用集合去重，避免重复处理同一记录

    processed_data = {
        'record_names': [],   # 记录名列表（如：['100', '101']）
        'signals': [],        # 降采样后的ECG信号列表
        'labels': [],         # 房颤标签列表（0表示正常，1表示房颤）
        'original_fs': 360    # MIT-BIH默认采样率
    }

    # 如果指定了输出目录，则创建该目录（如果在文件系统中不存在--避免重复）
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 如果用os.mkdir只能创建单层目录

    # 使用tqdm包装循环，显示处理进度条
    for record_name in tqdm(unique_records, desc="Processing Records"):
        try:
            # 1. 构建完整记录路径并读取ECG记录和注释文件
            # 示例：记录名称为'100'，则文件为100.hea, 100.dat等
            record_path = os.path.join(data_dir, record_name)

            # 使用wfdb读取记录数据（自动解析.hea和.dat文件）
            record = wfdb.rdrecord(record_path)
            # 读取对应的注释文件（'atr'为注释类型，包含房颤等标签信息）
            ann = wfdb.rdann(record_path, 'atr')

            # 2. 提取原始信号和元数据。
            original_fs = record.fs   # 获取原始采样率（MIT-BIH通常为360Hz）
            # 提取第一导联的信号（假设为MLII导联，常用于房颤检测）
            ecg_signal = record.p_signal[:, 0]  # p_signal是(n_samples, n_channels)的数组

            # 3. 执行降采样：将信号从原始采样率调整到目标采样率
            # 计算降采样后的样本数，保持信号时长不变
            # 公式：新样本数 = 原样本数 * (目标采样率 / 原采样率)
            num_samples = int(len(ecg_signal) * target_fs / original_fs)
            # 使用scipy的resample函数进行重采样
            resampled_signal = resample(ecg_signal, num_samples)

            # 4. 生成房颤标签数组（与降采样后的信号长度一致）
            # 初始化全零标签数组，数据类型为整数
            af_labels = np.zeros_like(resampled_signal, dtype=int)
            # 遍历所有注释点，检测房颤事件
            for i in range(len(ann.sample)):
                # 检查注释中是否包含房颤标记（如'(AFIB'表示房颤）
                if '(AF' in ann.aux_note[i]:
                    # 计算当前注释点在降采样后的对应位置
                    start = int(ann.sample[i] * target_fs / original_fs)
                    # 计算结束位置：如果是最后一个注释，则到信号末尾；否则为下一个注释的起始位置
                    end = int(ann.sample[i + 1] * target_fs / original_fs) if i + 1 < len(ann.sample) else len(
                        af_labels)
                    # 将[start, end)区间内的标签设为1（房颤）
                    af_labels[start:end] = 1

            # 将处理后的数据存入结果字典
            processed_data['record_names'].append(record_name)
            processed_data['signals'].append(resampled_signal)
            processed_data['labels'].append(af_labels)

            # 5. 保存处理后的数据到NPZ文件
            if output_dir:
                # 使用np.savez保存为压缩文件，包含信号、标签和采样率
                np.savez(
                    os.path.join(output_dir, f"{record_name}_processed.npz"),
                    signal=resampled_signal,
                    labels=af_labels,
                    fs=target_fs     # 保存目标采样率以便后续使用
                )

        except Exception as e:
            # 捕获并打印错误信息，跳过当前记录，继续处理后续数据
            print(f"\nError processing {record_name}: {str(e)}")
            continue

    return processed_data


if __name__ == "__main__":
    # 设置数据路径和输出目录
    data_folder = "D:/SRDP/吕浩毅/代码实现/pythonProject/Atrial fibrillation"
    output_folder = "./processed_ecg"    # 处理后的数据保存目录
    target_fs = 250  # 目标采样率设置为250Hz

    # 调用处理函数
    ecg_data = process_ecg_records(
        data_dir=data_folder,
        target_fs=target_fs,
        output_dir=output_folder
    )

    # 打印处理结果摘要
    print(f"\n成功处理 {len(ecg_data['record_names'])} 条记录")
    print(f"采样率：{ecg_data['original_fs']}Hz -> {target_fs}Hz")
    print(f"示例信号长度：{len(ecg_data['signals'][0])} 点")







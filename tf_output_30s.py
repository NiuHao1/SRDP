import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm


def generate_medical_tfr(segment, fs=250, wavelet='cmor1.5-1.0', n_scales=128):
    """生成医学级时频表示图"""
    max_freq = 40.0
    min_freq = 0.5
    scales = pywt.central_frequency(wavelet) * fs / (np.linspace(max_freq, min_freq, n_scales) * 2)
    coeffs, freqs = pywt.cwt(segment, scales, wavelet, sampling_period=1 / fs)
    power = np.log1p(np.abs(coeffs) ** 2)
    power = (power - np.min(power)) / (np.max(power) - np.min(power))
    return power, freqs


def save_tfr_as_image(tfr, save_path, dpi=150):
    """保存时频图"""
    plt.figure(figsize=(6, 4), dpi=dpi)
    plt.imshow(tfr, cmap='jet', aspect='auto',
               extent=[0, len(tfr[0]) / 250, 40, 0],
               vmax=np.percentile(tfr, 99), vmin=np.percentile(tfr, 1))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_10s_patients(input_dir, output_dir):
    """安全处理多患者数据（不修改任何文件名）"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成元数据
    metadata = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            patient_id = filename.split("_")[0]
            metadata.append({
                "filename": filename,
                "patient_id": patient_id,
                "segment_duration": 10,
                "original_fs": 250,
                "label": 0
            })

    # 按患者处理
    df = pd.DataFrame(metadata)
    for patient_id, group in tqdm(df.groupby("patient_id"), desc="Processing"):
        # 创建患者专属目录
        patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        # 处理每个文件
        for _, row in group.iterrows():
            try:
                seg = np.load(os.path.join(input_dir, row['filename']))
                tfr, _ = generate_medical_tfr(seg)
                output_name = f"{os.path.splitext(row['filename'])[0]}_TFR.png"
                save_tfr_as_image(tfr, os.path.join(patient_dir, output_name))
            except Exception as e:
                print(f"Error processing {row['filename']}: {str(e)}")


# 使用示例
input_folder = r"D:\SRDP\吕浩毅\代码实现\pythonProject\tf_images\30s_segments"
output_folder = r"D:\SRDP\吕浩毅\代码实现\pythonProject\tf_images\30s_output"
process_10s_patients(input_folder, output_folder)
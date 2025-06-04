import os
import numpy as np
from vmdpy import VMD  # 变分模态分解
from scipy.signal import hilbert  # Hilbert变换
import matplotlib.pyplot as plt

def generate_tf_image(signal, k=5, alpha=2000):
    """生成时频图（VMD + Hilbert谱）"""
    # 第一步：使用VMD（变分模态分解）将信号分解成K个子成分
    u_hat, omega_hat, _ = VMD(signal,
                              alpha=alpha,  # 带宽参数（控制每个子成分的频率范围）
                              tau=0.,       # 时间步长（0表示自动计算）
                              K=k,          # 分解层数（分成K个子信号）
                              DC=0,         # 不包含直流分量（去除恒定偏移）
                              init=1,       # 初始化方式（1表示均匀分布）
                              tol=1e-7)     # 收敛精度（数值越小分解越精确）

    # 创建一个全零矩阵，用于存放时频图数据
    # 形状为[K, 信号长度]，比如k=5、信号3000点 → 5行3000列的网格
    tf_matrix = np.zeros((k, len(signal)))

    # 对每个分解出的子成分进行处理
    for i in range(k):
        # Hilbert变换：将实信号转为复信号，用于提取瞬时特征
        # 类比：把普通照片转为3D立体图，能看清明暗和深度
        analytic_signal = hilbert(u_hat[i])
        # 计算瞬时幅值（信号强度的波动）
        amplitude = np.abs(analytic_signal)
        # 计算瞬时相位（信号角度的变化）
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        # 计算瞬时频率（相位变化的导数 × 采样率/2π）
        # 250是采样率（假设每秒250个点），将弧度转为实际频率值
        instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * 250  # 计算瞬时频率
        instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])  # 末尾补一个值保持长度一致
        # 能量谱 = 幅值 × 频率
        # 类似计算不同时间点、不同频率成分的能量强度
        tf_matrix[i] = amplitude * instantaneous_freq

    return tf_matrix


# 生成并保存时频图 -------------------------------------------------
def process_and_save_tf(input_dir, output_dir, k=5):
    """批量处理时频图生成"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith("_10s.npz"):
            data = np.load(os.path.join(input_dir, filename))

            all_tf_images = []
            for i in range(data['signals'].shape[0]):
                # 生成时频图
                tf_image = generate_tf_image(data['signals'][i], k=k)

                # 标准化到0-1范围
                tf_image = (tf_image - tf_image.min()) / (tf_image.max() - tf_image.min() + 1e-8)
                all_tf_images.append(tf_image)

                # 可视化示例（保存前3个样本）
                if i < 3:
                    plt.figure(figsize=(10, 4))
                    plt.imshow(tf_image, aspect='auto', # 自动调整宽高比
                               cmap='jet',              # 颜色映射
                               extent=[0, 10, 0, k],    # X轴0-10秒，Y轴0-k个模态（类似坐标刻度）
                               vmin=0, vmax=1)          # 颜色范围锁定为0-1
                    plt.colorbar(label='Normalized Energy')
                    plt.title(f"TF Image - {filename} Segment {i + 1}")
                    plt.xlabel("Time (s)")
                    plt.ylabel("Mode Index")
                    plt.savefig(os.path.join(output_dir, f"{filename[:-4]}_seg{i}_tf.png"))
                    plt.close()

            # 保存为numpy数组
            output_path = os.path.join(output_dir, f"tf_{filename}")
            np.savez(output_path,
                     tf_images=np.array(all_tf_images),
                     labels=data['labels'])
            print(f"Saved: {output_path}")


# 执行处理 ------------------------------------------------------
if __name__ == "__main__":
    # 参数设置
    INPUT_DIR = "D:/SRDP/吕浩毅/代码实现/pythonProject/final_data"  # 预处理后的数据目录
    OUTPUT_DIR = "D:/SRDP/吕浩毅/代码实现/pythonProject/tf_images"  # 时频图输出目录
    K_MODES = 5  # VMD分解模态数
    # 执行处理
    process_and_save_tf(INPUT_DIR, OUTPUT_DIR, k=K_MODES)
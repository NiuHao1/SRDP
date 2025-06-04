import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_af_statistics_report(data_dir):
    """生成符合医学论文要求的统计报告"""
    # 初始化统计数据结构（设置默认值）
    stats = {
        'signal_stats': {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        },
        'segment_stats': {}
    }

    # ================== 全局信号统计 ==================
    all_signals = []

    # 遍历所有时长类型
    for duration in [10, 30]:
        duration_dir = os.path.join(data_dir, f"{duration}s_segments")
        csv_path = os.path.join(duration_dir, "labels.csv")

        if not os.path.exists(csv_path):
            print(f"⚠️ 警告：未找到{duration}s的标签文件 {csv_path}")
            continue

        try:
            # 读取标签数据
            df = pd.read_csv(csv_path)

            # 统计当前时长的片段分布
            stats['segment_stats'][duration] = {
                'AF': len(df[df['label'] == 1]),
                'Non-AF': len(df[df['label'] == 0]),
                'Total': len(df)
            }

            # 抽样读取信号数据用于统计（添加文件存在性检查）
            sample_files = df.sample(min(100, len(df)))['filename']
            valid_signals = []
            for f in sample_files:
                file_path = os.path.join(duration_dir, f)
                if os.path.exists(file_path):
                    signal = np.load(file_path, allow_pickle=True).item()['signal']
                    valid_signals.append(signal)
                else:
                    print(f"⚠️ 警告：文件 {file_path} 不存在")

            if valid_signals:
                all_signals.extend(valid_signals)

        except Exception as e:
            print(f"处理{duration}s数据时出错: {str(e)}")
            continue

    # 计算全局统计量（添加空值保护）
    if all_signals:
        try:
            combined_signals = np.concatenate(all_signals)
            stats['signal_stats'] = {
                'mean': np.mean(combined_signals),
                'std': np.std(combined_signals),
                'min': np.min(combined_signals),
                'max': np.max(combined_signals)
            }
        except Exception as e:
            print(f"计算统计量时出错: {str(e)}")

    # ================== 专业可视化 ==================
    plt.figure(figsize=(15, 6))

    # 子图1：信号统计（添加有效性检查）
    try:
        plt.subplot(1, 2, 1)
        stats_labels = ['Mean', 'STD', 'Min', 'Max']
        stats_values = [
            stats['signal_stats']['mean'],
            stats['signal_stats']['std'],
            stats['signal_stats']['min'],
            stats['signal_stats']['max']
        ]
        # 将None转换为0
        stats_values = [v if v is not None else 0 for v in stats_values]
        plt.bar(stats_labels, stats_values, color=['#4B96E9', '#88C999', '#FF6B6B', '#FFD700'])
        plt.title('Global Signal Statistics')
        plt.ylabel('Amplitude (mV)')
        plt.grid(axis='y', alpha=0.3)
    except Exception as e:
        print(f"绘制信号统计图时出错: {str(e)}")

    # 子图2：分段分布（添加空数据检查）
    try:
        plt.subplot(1, 2, 2)
        if stats['segment_stats']:
            durations = list(stats['segment_stats'].keys())
            af_counts = [v['AF'] for v in stats['segment_stats'].values()]
            non_af_counts = [v['Non-AF'] for v in stats['segment_stats'].values()]

            bar_width = 0.25
            x = np.arange(len(durations))

            plt.bar(x - bar_width / 2, af_counts, bar_width, label='AF', color='#FF6B6B')
            plt.bar(x + bar_width / 2, non_af_counts, bar_width, label='Non-AF', color='#4ECDC4')

            plt.xticks(x, [f"{d}s" for d in durations])
            plt.title('Segment Distribution by Duration')
            plt.xlabel('Duration')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, '无有效分段数据', ha='center', va='center')
    except Exception as e:
        print(f"绘制分段分布图时出错: {str(e)}")

    plt.tight_layout()

    return stats, plt


def generate_latex_table(stats):
    """生成LaTeX格式的统计表格"""
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{ECG Signal Statistics and AF Distribution}
\\label{tab:stats}
\\begin{tabular}{lrrr}
\\toprule
 & 10s Segments & 30s Segments & Total \\\\
\\midrule
"""

    # 信号统计部分
    latex_code += f"Mean (mV) & \\multicolumn{{2}}{{c}}{{{stats['signal_stats']['mean']:.4f}}} & - \\\\\n"
    latex_code += f"STD (mV) & \\multicolumn{{2}}{{c}}{{{stats['signal_stats']['std']:.4f}}} & - \\\\\n"
    latex_code += f"Min (mV) & \\multicolumn{{2}}{{c}}{{{stats['signal_stats']['min']:.2f}}} & - \\\\\n"
    latex_code += f"Max (mV) & \\multicolumn{{2}}{{c}}{{{stats['signal_stats']['max']:.2f}}} & - \\\\\n"
    latex_code += "\\midrule\n"

    # 分段统计部分
    total_af = sum(v['AF'] for v in stats['segment_stats'].values())
    total_non_af = sum(v['Non-AF'] for v in stats['segment_stats'].values())

    for duration in [10, 30]:
        if duration in stats['segment_stats']:
            d = stats['segment_stats'][duration]
            af_percent = d['AF'] / d['Total'] * 100
            latex_code += f"AF ({duration}s) & {d['AF']} & ({af_percent:.1f}\\%) & \\\\\n"
            non_af_percent = d['Non-AF'] / d['Total'] * 100
            latex_code += f"Non-AF ({duration}s) & {d['Non-AF']} & ({non_af_percent:.1f}\\%) & \\\\\n"

    latex_code += f"\\midrule\nTotal AF & \\multicolumn{{2}}{{c}}{{{total_af}}} & \\\\\n"
    latex_code += f"Total Non-AF & \\multicolumn{{2}}{{c}}{{{total_non_af}}} & \\\\\n"
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}"""

    return latex_code


if __name__ == "__main__":
    # 配置路径
    data_folder = "./statistics"

    # 生成统计报告
    stats_report, visualization = generate_af_statistics_report(data_folder)

    # 打印统计结果
    print("\n=== ECG Signal Statistics ===")
    print(f"Mean ± STD (mV): {stats_report['signal_stats']['mean']:.4f} ± {stats_report['signal_stats']['std']:.4f}")
    print(f"Range: [{stats_report['signal_stats']['min']:.2f}, {stats_report['signal_stats']['max']:.2f}]")

    print("\n=== AF Segment Distribution ===")
    for duration, data in stats_report['segment_stats'].items():
        af_percent = data['AF'] / data['Total'] * 100
        print(f"{duration}s Segments:")
        print(f"  AF: {data['AF']} ({af_percent:.1f}%)")
        print(f"  Non-AF: {data['Non-AF']} ({100 - af_percent:.1f}%)\n")

    # 生成LaTeX表格
    print("\n=== LaTeX Table ===")
    print(generate_latex_table(stats_report))

    # 保存可视化结果
    visualization.savefig(os.path.join(data_folder, 'statistics_report.png'), dpi=300)
    print(f"\n可视化报告已保存至：{os.path.join(data_folder, 'statistics_report.png')}")
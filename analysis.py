import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("1. 生成模拟数据...")
t = np.linspace(0, 10, 200)
data = {'time': t}
for i in range(6):
    target = np.sin(2 * np.pi * (0.5 + 0.1 * i) * t)
    actual = target + 0.02 * np.random.randn(200)
    data[f'joint_{i}'] = actual
    data[f'target_joint_{i}'] = target

os.makedirs('data', exist_ok=True)
pd.DataFrame(data).to_csv('data/joint_data.csv', index=False)
print("   数据已保存至 data/joint_data.csv")

print("2. 读取数据并计算误差...")
df = pd.read_csv('data/joint_data.csv')
time = df['time'].values
actual_pos = df[[f'joint_{i}' for i in range(6)]].values
target_pos = df[[f'target_joint_{i}' for i in range(6)]].values

print("=== 实际位置最大值 ===")
print(np.max(actual_pos, axis=0))
print("=== 实际位置平均值 ===")
print(np.mean(actual_pos, axis=0))
print("=== 平均绝对误差 (MAE) ===")
print(np.mean(np.abs(target_pos - actual_pos), axis=0))

print("3. 绘制目标与实际曲线...")
os.makedirs('output', exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(time, target_pos[:, 0], 'r--', label='Target')
axes[0].plot(time, actual_pos[:, 0], 'b-', label='Actual')
axes[0].set_ylabel('Joint 1 Pos')
axes[0].legend()
axes[1].plot(time, target_pos[:, 1], 'r--', label='Target')
axes[1].plot(time, actual_pos[:, 1], 'b-', label='Actual')
axes[1].set_ylabel('Joint 2 Pos')
axes[1].legend()
axes[1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig('output/test_plot.png')
print("   图表已保存至 output/test_plot.png")
print("程序运行结束！")

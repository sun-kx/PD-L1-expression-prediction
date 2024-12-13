import os.path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# 读取Excel文件
file_paths = r'E:\skx\DP-L1\HistoBistro\result\AttentionMIL/outputs_1_2_val_33_good.csv'
pd_l1_models = 'AttentionMIL_NF+HY'

# 为每个文件绘制ROC曲线

data = pd.read_csv(file_paths)

# # 提取 ground_truth 和 logits 列
ground_truth = data['ground_truth'].values
prediction = data['prediction'].values

logits = data['logits'].apply(lambda x: float(x.split('(')[-1].split(',')[0])).values  # 将字符串形式的tensor转换为浮点数

# # 计算预测概率（sigmoid函数）
probs = torch.sigmoid(torch.tensor(logits)).numpy()

# 将 probs 作为新的一列添加到原始数据中
data['probs'] = probs

# # 将结果保存为新的表格
new_file_path = os.path.dirname(file_paths)

csv_file_path = os.path.join(new_file_path, 'AttentionMIL_NF+HY.csv')
new_img = os.path.join(new_file_path, 'AttentionMIL_NF+HY.png')
data.to_csv(csv_file_path, index=False)

# 计算Sensitivity-Specificity曲线
fpr, tpr, _ = roc_curve(ground_truth, probs)
roc_auc = auc(fpr, tpr)
specificity = 1 - fpr
print(roc_auc)
ids = range(len(data))  # 用作x轴的索引
target_reg_values = probs
colors = data['ground_truth'].map({0: 'green', 1: 'red'})  # 根据correct列选择颜色

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(ids, target_reg_values, c=colors)
# 将 neg 和 pos 的标签分别添加到图例中
plt.scatter([], [], color='red', label='PD-L1+_prediction')  # 红点表示 neg
plt.scatter([], [], color='green', label='PD-L1-_prediction')  # 绿点表示 pos

# 添加横线
plt.axhline(y=0.5, color='blue', linestyle='--', label='Probs threshold')

# 图表设置
plt.xlabel('Patient Index')
plt.ylabel('PD-L1 Probs')
plt.title('AttentionMIL_NF+HY')
plt.legend()

plt.savefig(new_img, dpi=300, bbox_inches='tight')  # Save with 300 dpi for higher quality
# 显示图表
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
import torch
# 读取csv文件
name = 'TCGA-TNBC'
model = 'TransMIL'
data = pd.read_csv(rf'E:\skx\DP-L1\HistoBistro\result\{model}/{model}_{name}.csv')

# 提取数据
ids = range(len(data))  # 用作x轴的索引

# logits = data['TARGET_REG'].apply(lambda x: float(x.split('(')[-1].split(',')[0])).values  # 将字符串形式的tensor转换为浮点数
logits = data['TARGET_REG']
# logits = data['probs']  # y轴的值

colors = data['prediction'].map({0: 'orange', 1: 'blue'})  # 根据correct列选择颜色
# probs = torch.sigmoid(torch.tensor(logits)).numpy()

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(ids, logits, c=colors)
# 将 neg 和 pos 的标签分别添加到图例中

plt.scatter([], [], color='blue', label='PD-L1+_prediction')  # 红点表示 neg
plt.scatter([], [], color='orange', label='PD-L1-_prediction')  # 绿点表示 pos

# 添加横线
plt.axhline(y=1.9355, color='purple', linestyle='--', label='Threshold')

# 图表设置
plt.xlabel('Patient Index')
plt.ylabel('PD-L1 RNA FPKM')
plt.title(f'TCGA-TNBC')
plt.legend()

plt.savefig(rf'E:\skx\DP-L1\HistoBistro\result\{model}/{model}_{name}.png', dpi=300, bbox_inches='tight')  # Save with 300 dpi for higher quality
# 显示图表
plt.show()

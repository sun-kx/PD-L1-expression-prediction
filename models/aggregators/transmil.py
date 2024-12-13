import numpy as np
import torch
import torch.nn as nn

from models.aggregators import BaseAggregator
from models.aggregators.model_utils import PPEG, NystromTransformerLayer


class TransMIL(BaseAggregator):
    def __init__(self, num_classes, input_dim=1024, pos_enc='PPEG', cls='class', **kwargs):
        super(BaseAggregator, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self.pos_enc = pos_enc
        print(f'Using {self.pos_enc} positional encoding')
        self._fc1 = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = NystromTransformerLayer(dim=512)
        self.layer2 = NystromTransformerLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)
        # 新增回归头
        self._reg_head = nn.Linear(512, 1)  # 输出一个标量值
        self.cls = cls
    def forward(self, x, coords=None):


        h = x  #[B, n, 1024]

        h = self._fc1(h)  #[B, n, 512]

        #----> padding
        H = h.shape[1]
        if self.pos_enc == 'PPEG':
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))  # find smallest square larger than n
            add_length = _H * _W - H  # add N - n, first entries of feature vector added at the end to fill up until square number
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  #[B, N, 512]
        elif self.pos_enc == 'PPEG_padded':  # 仅适用于 batch size 为 1 的情况
            if h.shape[1] > 1:  # 确保有多个 patch
                # 获取 x, y 坐标和 patch 大小
                x_coords = coords[:, :, 1]
                y_coords = coords[:, :, 2]
                patch_width = coords[:, :, 3].unique().item()  # 假设所有 patch 宽度相同
                patch_height = coords[:, :, 4].unique().item()  # 假设所有 patch 高度相同
                patch_size = torch.tensor([patch_width, patch_height], device=coords.device)

                # 计算坐标范围
                min_x, max_x = x_coords.min(), x_coords.max()
                min_y, max_y = y_coords.min(), y_coords.max()

                # 动态调整网格大小，计算网格尺寸
                _H = ((max_y - min_y) // patch_height).int() + 1
                _W = ((max_x - min_x) // patch_width).int() + 1

                # 创建 base_grid
                base_grid = torch.zeros((h.shape[0], _H, _W, h.shape[-1]), device=h.device)

                # 动态计算 grid_indices（在网格中的位置）
                grid_indices = torch.stack([(y_coords - min_y) // patch_height,
                                            (x_coords - min_x) // patch_width], dim=-1).long()

                # 检查 patch 的数量是否适合放入网格
                if grid_indices.shape[1] <= _H * _W:
                    # # 将每个 patch 的特征放置到 base_grid 的相应位置
                    # for i in range(h.shape[1]):
                    #     base_grid[:, grid_indices[0, i, 0], grid_indices[0, i, 1]] = h[:, i, :]
                    # 直接使用高级索引将 patch 特征放入 base_grid 的相应位置
                    base_grid[:, grid_indices[:, :, 0], grid_indices[:, :, 1]] = h
                else:
                    # 若 patch 数量超过网格容量，可能需要截断或其他操作
                    print(f"Warning: Too many patches ({grid_indices.shape[1]}) for the grid size ({_H}, {_W}).")

                # 将 base_grid 转换为与输入形状一致的形状
                h = base_grid.view(h.shape[0], -1, h.shape[-1])

            else:
                _H, _W = 1, 1

        #----> cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #----> first translayer
        h = self.layer1(h)  #[B, N, 512]

        #----> ppeg
        h = self.pos_layer(h, _H, _W)  #[B, N, 512]

        #----> second translayer
        h = self.layer2(h)  #[B, N, 512]

        #----> cls_token
        h = self.norm(h)[:, 0]

        #----> predict
        logits = self._fc2(h)  #[B, n_classes]

        if self.cls == 'class+reg' or self.cls == 'reg':
            # ----> 回归头预测
            reg_out = self._reg_head(h)  # [B, 1]
            return logits, reg_out
        else:
            return logits


import torch
import torch.nn as nn
from einops import repeat

from models.aggregators.aggregator import BaseAggregator
from models.aggregators.model_utils import Attention, FeedForward, PreNorm


class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class Transformer(BaseAggregator):
    def __init__(
        self,
        *,
        num_classes,
        input_dim=2048,
        dim=512,
        depth=2,
        heads=8,
        mlp_dim=512,
        pool='mean',
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        pos_enc=True,
        cls='class',
    ):
        super(BaseAggregator, self).__init__()
        assert pool in {
            'cls', 'mean'
        }, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.projection = nn.Sequential(nn.Linear(input_dim, heads*dim_head, bias=True), nn.ReLU())
        self.mlp_head_class = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, num_classes))
        self.mlp_head_reg = nn.Sequential(nn.LayerNorm(mlp_dim), nn.Linear(mlp_dim, 1))
        self.transformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.cls = cls
        self.pos_enc = pos_enc
        self.shortcut = nn.Sequential(nn.ReLU())

    def forward(self, x, coords=None, register_hook=False):
        b, _, _ = x.shape

        x = self.projection(x)

        # if self.pos_enc:
        #     x = x + self.pos_enc(coords)
        # if self.pos_enc:
        #     x = torch.cat((x, coords), dim=-1)
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        out = self.dropout(x)
        out = self.transformer(out, register_hook=register_hook)

        out = out + self.shortcut(x)

        out = out.mean(dim=1) if self.pool == 'mean' else out[:, 0]
        if self.cls == 'class+reg' or self.cls == 'reg':
            class_out = self.mlp_head_class(self.norm(out))
            reg_out = self.mlp_head_reg(self.norm(out))
            return class_out, reg_out
        else:
            class_out = self.mlp_head_class(self.norm(out))
            return class_out




# transformer = Transformer(num_classes=2)
# transformer(torch.rand(1, 1, 2048))


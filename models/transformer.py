from torch import nn



class MultiHeadAttentionBlock(nn.Module):
  def __init__(self,
               embeding_dim : int = 768,
               num_heads : int = 12,
               attn_dropout : float = 0.0):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)

    self.multihead_attention = nn.MultiheadAttention(embed_dim=embeding_dim,
                                                     num_heads=num_heads,
                                                     dropout=attn_dropout,
                                                     batch_first=True)

  def forward(self,x):
    x = self.layer_norm(x)

    attn_output, _ = self.multihead_attention(query=x,key=x,value=x,need_weights=False)

    return attn_output


class MLPBlock(nn.Module):
  def __init__(self,
               embeding_dim : int = 768,
               mlp_size : int = 3072,
               dropout : float = 0.1
               ):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)

    self.MLP = nn.Sequential(
        nn.Linear(in_features=embeding_dim,out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,out_features=embeding_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self,x):
    x = self.layer_norm(x)

    return self.MLP(x)


class TransformerEncoderBlock(nn.Module):
  def __init__(self,
               embeding_dim : int = 768,
               num_heads : int = 12,
               attn_dropout : float = 0.0,
               mlp_size : int = 3072,
               dropout : float = 0.1):

    super().__init__()

    self.multihead_attention_block = MultiHeadAttentionBlock(embeding_dim=embeding_dim,
                                                             num_heads=num_heads,
                                                             attn_dropout=attn_dropout)

    self.mlp_block = MLPBlock(embeding_dim=embeding_dim,
                              mlp_size=mlp_size,
                              dropout=dropout)


  def forward(self,x):
    x = self.multihead_attention_block(x) + x #residual connection
    x = self.mlp_block(x) + x

    return x
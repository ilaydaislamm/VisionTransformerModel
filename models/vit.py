from models.patching import PatchingModule,PositionAndClassEmbeding
from models.transformer import TransformerEncoderBlock
from torch import nn
class VitModel(nn.Module):
  def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 number_of_transformer_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000):

    super().__init__()

    self.patch_embeding = PatchingModule(in_channels=in_channels,embeding_size=embedding_dim,patching_size=patch_size)

    self.classtoken_and_position = PositionAndClassEmbeding(img_size=img_size,patching_size=patch_size,embeding_size=embedding_dim)

    self.transformer_encoder = nn.Sequential( *[TransformerEncoderBlock(embeding_dim=embedding_dim,
                                                       num_heads=num_heads,
                                                       attn_dropout=attn_dropout,
                                                       mlp_size=mlp_size,
                                                       dropout=mlp_dropout) for _ in range(number_of_transformer_layers)]
                                              )

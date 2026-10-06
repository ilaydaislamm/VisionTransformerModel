from torch import nn
import torch

class PatchingModule(nn.Module):
  def __init__(self,
               in_channels : int = 3,
               embeding_size : int = 768,
               patching_size : int = 16):
    super().__init__()

    self.patching = nn.Conv2d(in_channels=in_channels,
                              out_channels=embeding_size,
                              kernel_size=patching_size,
                              stride=patching_size)

    self.flatten = nn.Flatten(start_dim=2,end_dim=3)


  def forward(self,x):
    return self.flatten(self.patching(x)).permute(0,2,1)

class PositionAndClassEmbeding(nn.Module):
    def __init__(self,
                img_size: int = 224,
                patching_size: int = 16,
                embeding_size: int = 768):
        super().__init__()

        self.number_of_patches = (img_size // patching_size) ** 2

        self.class_token = nn.Parameter(torch.randn((1, 1, embeding_size)), requires_grad=True)

        self.position_embeding = nn.Parameter(torch.randn((1, self.number_of_patches + 1, embeding_size)))

    def forward(self, x):
        current_batch_size = x.shape[0]

        class_token_expanded = self.class_token.expand(current_batch_size, -1, -1)

        return torch.cat((class_token_expanded, x), dim=1) + self.position_embeding


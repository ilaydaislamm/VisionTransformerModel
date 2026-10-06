import torch
from openpyxl.drawing.geometry import Path2D
from torch import nn

from models.vit import VitModel
from utils.data_setup import create_dataloaders
from pathlib import Path
import torchvision.transforms as transforms
from utils.engine import train
from utils.predictions import transform_predict
data_path = Path(r"C:\Users\omen\PycharmProjects\VisionTransformerModel\Caltech101")

def to_rgb(image):
  return image.convert("RGB")

IMG_SIZE = 224
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.RandomResizedCrop(size = IMG_SIZE,scale = (0.6,1)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.4],std = [0.229, 0.224, 0.225])

])

test_transform = transforms.Compose([
    transforms.Lambda(to_rgb),
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.4],std = [0.229, 0.224, 0.225])
])



train_data_loader,test_data_loader,class_names = create_dataloaders(
                                    train_dir=data_path,
                                    train_transformer=train_transform,
                                    test_transformer=test_transform,
                                    batch_size=BATCH_SIZE,
                                    train_and_test_seperate=False
                                    )



vit_model = VitModel(img_size = IMG_SIZE,
                 in_channels = 3,
                 patch_size = 16,
                 number_of_transformer_layers = 12,
                 embedding_dim = 768,
                 mlp_size = 3072,
                 num_heads = 12,
                 attn_dropout = 0,
                 mlp_dropout = 0.1,
                 embedding_dropout = 0.1,
                 num_classes = len(class_names)
                )

optimizer = torch.optim.Adam(vit_model.parameters(),lr = 3e-3)

loss_func = nn.CrossEntropyLoss()


results = train(model=vit_model,
                train_dataloader=train_data_loader,
                test_dataloader=test_data_loader,
                optimizer=optimizer,
                loss_fn=loss_func,
                epochs=10,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )


image_predict = Path(r"C:\Users\omen\PycharmProjects\VisionTransformerModel\image.jpg")

prediction = transform_predict(model=vit_model,
                               image_path=image_predict,
                               class_names=class_names,
                               image_size=IMG_SIZE,
                               transform=test_transform)
import torch.utils
import torchvision
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for x, label in dataloader:
    print('x shape:', x.shape)
    print('label shape:', label.shape)
    break

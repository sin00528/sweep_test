import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import timm
import torchmetrics

class MyLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        num_classes = 10

        self.model = timm.create_model(config.model_name, pretrained=True)

        # Check if the model has a `classifier` or `fc` attribute
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise AttributeError(f"{config.model_name} model does not have 'classifier' or 'fc' attribute.")

        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        # self.val_roc_auc = torchmetrics.ROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("train_f1_score", f1_score, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1(y_hat, y)

        self.log("validation_loss", loss, on_step=False, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log("validation_f1_score", f1_score, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


# Modify the LightningDataModule for MNIST
class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = 31

    def prepare_data(self):
        # Download the MNIST dataset
        datasets.MNIST(root="./data", train=True, download=True)
        datasets.MNIST(root="./data", train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for EfficientNet
            transforms.Grayscale(num_output_channels=3),  # Convert MNIST to RGB
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Use torchvision to load MNIST dataset
        self.mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        self.mnist_val = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

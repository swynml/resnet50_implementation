import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn as nn
import numpy as np
import sys

class Block(pl.LightningModule):
    def __init__(self, channels, start_expansion, end_expansion, stride):
        super(Block, self).__init__()
        self.start_expansion = start_expansion
        self.end_expansion = end_expansion
        self.stride = stride
        self.residual_rescale = None

        self.conv_1 = nn.Conv2d(in_channels=channels*self.start_expansion, 
                                out_channels=channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(channels)
        self.conv_2 = nn.Conv2d(in_channels=channels, 
                                out_channels=channels, 
                                kernel_size=3, 
                                stride=self.stride, 
                                padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(channels)
        self.conv_3 = nn.Conv2d(in_channels=channels, 
                                out_channels=self.end_expansion*channels, 
                                kernel_size=1, 
                                stride=1, 
                                padding=0)
        self.batch_norm_3 = nn.BatchNorm2d(self.end_expansion*channels)

        if self.start_expansion != self.end_expansion:
            self.residual_rescale = nn.Sequential(
                    nn.Conv2d(channels*self.start_expansion, 
                            channels*self.end_expansion, 
                            kernel_size=1, 
                            stride=self.stride),
                    nn.BatchNorm2d(channels*self.end_expansion)
                )

        self.relu = nn.ReLU()

    def forward(self, x):
        input_x = x.clone()
        x = self.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.relu(self.batch_norm_2(self.conv_2(x)))
        x = self.batch_norm_3(self.conv_3(x))

        if self.residual_rescale is not None:
            input_x = self.residual_rescale(input_x)
        
        # skip connection
        x = self.relu(x + input_x)

        return x

class MyResNet50(pl.LightningModule):
    def __init__(self, lr):
        super(MyResNet50, self).__init__()
        self.in_channels = 64
        self.expansion = 4
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.input_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.blocks_layers_1 = self.make_initial_blocks_layer(sublayers_num=3, bl_channels=64, stride=1)
        self.blocks_layers_2 = self.make_blocks_layer(sublayers_num=4, bl_channels=128, stride=2)
        self.blocks_layers_3 = self.make_blocks_layer(sublayers_num=6, bl_channels=256, stride=2)
        self.blocks_layers_4 = self.make_blocks_layer(sublayers_num=3, bl_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*self.expansion, 10)
    
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.input_conv(x)))
        x = self.max_pool(x)
        x = self.blocks_layers_1(x)
        x = self.blocks_layers_2(x)
        x = self.blocks_layers_3(x)
        x = self.blocks_layers_4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_initial_blocks_layer(self, sublayers_num, bl_channels, stride):
        blocks_layers = []
        for i in range(sublayers_num):
            if i == 0:
                blocks_layers.append(Block(channels=bl_channels, start_expansion=1, end_expansion=self.expansion, stride=stride))
            else:
                blocks_layers.append(Block(channels=bl_channels, start_expansion=self.expansion, end_expansion=self.expansion, stride=stride))

        return nn.Sequential(*blocks_layers)

    def make_blocks_layer(self, sublayers_num, bl_channels, stride):
        blocks_layers = []
        for i in range(sublayers_num):
            if i == 0:
                blocks_layers.append(Block(channels=bl_channels, start_expansion=2, end_expansion=self.expansion, stride=stride))
            else:
                blocks_layers.append(Block(channels=bl_channels, start_expansion=self.expansion, end_expansion=self.expansion, stride=1))          
                

        return nn.Sequential(*blocks_layers)

    def training_step(self, batch, batch_nb):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x_val, y_val = batch
        preds_val = self(x_val)
        loss_val = self.loss_fn(preds_val, y_val)

        accuracy = Accuracy().to(self.device)
        acc = accuracy(preds_val, y_val)
        self.log('accuracy', acc, on_epoch=True)
        self.log("validation_loss", loss_val, on_epoch=True)

        return loss_val
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5),
            # "interval": "epoch",
            "monitor": "validation_loss",
            # "frequency": 5
        },
    }

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        # Given a batch of examples return a batch of predicted classes.
        preds_val = self(torch.Tensor(input_array))
        
        return preds_val.cpu().detach().numpy()

    def certainty(self, input_array: np.ndarray) -> np.ndarray:
        # Given a batch of examples return a batch of certainty levels.
        predictions = self.predict(input_array)
        epsilon = sys.float_info.min

        predictive_entropies = []
        for pred in predictions:
            pe = -np.sum( np.mean(pred, axis=0) * np.log(np.mean(pred, axis=0) + epsilon),
                    axis=-1)
            predictive_entropies.append(pe)

        return predictive_entropies

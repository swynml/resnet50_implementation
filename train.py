import torchvision
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
from ResNet50Model import MyResNet50
from RN50_jay import RN50
from pytorch_lightning.callbacks import ModelCheckpoint

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Training_resnet50")


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=6, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100*batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = MyResNet50(lr=0.1)
# model = RN50(lr=0.1)
# print(model)

# checkpoint = torch.load('saved_models/v2/resnet50_model-epoch=99-train_loss=1.31-val_loss=0.00-accuracy=0.50.ckpt')
# model.load_state_dict(checkpoint["state_dict"])

checkpoint_callback = ModelCheckpoint(
    monitor='accuracy',
    mode = 'max',
    save_top_k=5,
    dirpath='saved_models/',
    filename='resnet50_model-{epoch:02d}-{train_loss:.2f}-{validation_loss:.2f}-{accuracy:.2f}'
)

trainer = pl.Trainer(max_epochs=200, accelerator='gpu', devices=1, callbacks=[checkpoint_callback])

mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
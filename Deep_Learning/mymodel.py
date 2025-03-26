import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import timm

class MyModel(L.LightningModule):
    def __init__(self,
                 lr: float = 0.001,
                 weight_decay: float = 1e-4,
                 num_class: int = 3,
                 dropout_rate = 0.3,
                 class_counts= [531, 423, 702],
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        frequencies = np.array(class_counts) / np.array(class_counts).sum()
        inverse_frequencies = 1 / frequencies
        weight = inverse_frequencies / inverse_frequencies.max()
        class_weights = torch.FloatTensor(weight)
        print(class_weights)

        self.model = timm.create_model(
            'resnet51q.ra2_in1k',
            pretrained=True,
            pretrained_cfg_overlay=dict(file='../Timm/resnet51q.safetensors'),
            in_chans=1,
            num_classes=self.hparams.num_class)

        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.hparams.dropout_rate),
            nn.Linear(in_features, self.hparams.num_class))

        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        self.acc_macro = Accuracy(task="multiclass", num_classes=self.hparams.num_class, average='macro')
        self.acc_micro = Accuracy(task="multiclass", num_classes=self.hparams.num_class, average='micro')

        self.f1score = F1Score(task='multiclass', num_classes=self.hparams.num_class, average='weighted')

        self.train_labels = []
        self.train_preds = []

        self.validation_step_outputs_preds = []
        self.validation_step_outputs_targets = []

    def forward(self, x):
        return self.model(x)

    def get_input(self, batch):
        return batch['preMRI'], batch['class']

    def training_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc_macro = self.acc_macro(preds, y)
        acc_micro = self.acc_micro(preds, y)

        f1score = self.f1score(preds, y)

        self.log('train/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('train/acc_macro', acc_macro, logger=True, batch_size=x.shape[0])
        self.log('train/acc_micro', acc_micro, logger=True, batch_size=x.shape[0])
        self.log('train/f1score', f1score, logger=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc_macro = self.acc_macro(preds, y)
        acc_micro = self.acc_micro(preds, y)

        f1score = self.f1score(preds, y)

        self.log('val/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('val/acc_macro', acc_macro, logger=True, batch_size=x.shape[0])
        self.log('val/acc_micro', acc_micro, logger=True, batch_size=x.shape[0])
        self.log('val/f1score', f1score, logger=True, batch_size=x.shape[0])

        self.validation_step_outputs_preds.extend(preds.cpu().numpy())
        self.validation_step_outputs_targets.extend(y.cpu().numpy())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_input(batch)

        y_hat = self(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc_macro = self.acc_macro(preds, y)
        acc_micro = self.acc_micro(preds, y)

        f1score = self.f1score(preds, y)

        self.log('test/loss', loss, logger=True, batch_size=x.shape[0])
        self.log('test/acc_macro', acc_macro, logger=True, batch_size=x.shape[0])
        self.log('test/acc_micro', acc_micro, logger=True, batch_size=x.shape[0])
        self.log('test/f1score', f1score, logger=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
            }
        }
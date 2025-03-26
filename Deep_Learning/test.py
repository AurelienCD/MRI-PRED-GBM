import lightning as L
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger
#from data.MRIDataModule_Alexandre import MRIDataModule
from data.MRIDataModule_MAJ09122024 import MRIDataModule
from classifieur.mymodel import MyModel
import datetime
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
from lightning.pytorch.strategies import DDPStrategy

class ConfusionMatrixCallback(Callback):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def on_validation_epoch_end(self, trainer, pl_module):
        # Récupération des prédictions et des cibles depuis le modèle
        preds = pl_module.validation_step_outputs_preds
        targets = pl_module.validation_step_outputs_targets

        # Calcul de la matrice de confusion
        cm = confusion_matrix(targets, preds, labels=np.arange(self.num_classes))

        # Visualisation de la matrice de confusion
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.num_classes))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix - Epoch {trainer.current_epoch}")
        plt.close(fig)

        # Enregistrement dans TensorBoard
        trainer.logger.experiment.add_figure(f"Confusion Matrix", fig,
                                             global_step=trainer.current_epoch)

        # Réinitialiser les listes pour la prochaine époque
        pl_module.validation_step_outputs_preds = []
        pl_module.validation_step_outputs_targets = []
def main():

    # TODO :  système arguments
    # TODO : chargement automatique de configuration de modèle via fichier yaml

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    postfixname = 'classifier'

    logdir = './logs/2024-12-23_150642_classifier'  # valeur par défaut None
    #logdir = None

    lr = 0.00024793835138482544
    weight_decay = 3.5637293220378274e-05
    dropout_rate = 0.1835202564030761

    batch_size = 30
    n_epoch = 300
    patience = 50

    ckptdir = os.path.join(logdir, 'checkpoints')

    confusion_matrix_callback = ConfusionMatrixCallback(num_classes=3)

    model = MyModel(lr=lr, weight_decay=weight_decay, dropout_rate=dropout_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc_macro",
        dirpath=os.path.join(logdir, 'checkpoints'),
        filename="best_model",
        save_top_k=3,
        mode="max",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/acc_macro",
        mode="max",
        patience=patience,
        verbose=True
    )

    tensorboard_callback = TensorBoardLogger(name="tensorboard", save_dir=logdir, default_hp_metric="val_acc_macro")

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, confusion_matrix_callback],
        logger=tensorboard_callback,
        max_epochs=n_epoch,
        accelerator='gpu',
        devices='1',
        log_every_n_steps=5  # Par defaut log_every_n_steps = 50
    )

    db_path = '../data'
    task = 'classification'
    manifest = f'MRI_dataset_{task}.csv'
    patient_split = f'patient_splits'

    data = MRIDataModule(
        dataset_path=db_path,
        manifest_filename=manifest,
        batch_size=batch_size,
        task=task,
        crop_size=None,
        train_val_test_shuffle=(True, False, False),
        train_val_test_split=(0.6, 0.2, 0.2),
        weighted_sample=False,
        seed=23,
        verbose=True,
        normalization='max',
        num_workers=None,
        patient_splits_file=patient_split)

    evaluation = {}
    for ckpt in os.listdir(ckptdir):
        evaluation[ckpt] = trainer.test(model, data, ckpt_path=os.path.join(ckptdir, ckpt))[0]
    pd.DataFrame(evaluation).transpose().to_csv(os.path.join(logdir, 'evaluation.csv'))


if __name__ == '__main__':
    main()

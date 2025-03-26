import lightning as L
import sys
import os
import optuna
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from data.MRIDataModule_MAJ09122024 import MRIDataModule
from classifieur.mymodel import MyModel
import datetime
import pandas as pd

def main():

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    postfixname = 'classifier'

    logdir = f'././logs/{now}_{postfixname}'
    os.makedirs(logdir, exist_ok=True)

    db_name = f'ds_{now}.sqlite3'
    db_path = f'sqlite:///{db_name}'

    n_epoch = 300
    n_trials = 50
    batch_size = 30

    min_lr = 1e-7
    max_lr = 1e-2

    min_weight_decay = 1e-6
    max_weight_decay = 1e-2

    min_dropout = 0.1
    max_dropout = 0.5

    patience = 50

    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', min_lr, max_lr, log=True)
        weight_decay = trial.suggest_float('weight_decay', min_weight_decay, max_weight_decay, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', min_dropout, max_dropout, log=True)

        print(f'\ntrial : {trial}')
        print(f'lr : {lr}')
        print(f'weight_decay : {weight_decay})')
        print(f'dropout_rate : {dropout_rate})\n')

        model = MyModel(lr=lr, weight_decay=weight_decay, dropout_rate=dropout_rate)

        checkpoint_callback = ModelCheckpoint(
            monitor="val/acc_macro",
            dirpath=os.path.join(logdir, 'checkpoints'),
            filename=f"best_model_trial",
            save_top_k=3,
            mode="max",
        )

        early_stopping_callback = EarlyStopping(
            monitor="val/acc_macro",
            mode="max",
            patience=patience,
            verbose=True
        )

        trainer = L.Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback],
            max_epochs=n_epoch,
            accelerator='gpu',
            devices='1',
            log_every_n_steps=5
        )

        db_path = '../data'
        task = 'classification'
        manifest = f'MRI_dataset_{task}.csv'

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
            num_workers=None)

        data.setup('fit')
        split_file = data.save_patient_splits()

        trainer.fit(model, data)
        val_acc = trainer.callback_metrics['val/acc_macro'].item()
        return val_acc


    study = optuna.create_study(
            direction="maximize",
            storage=db_path,
            study_name=f"mri_classification_study_{now}",
            load_if_exists=False
        )

    study.optimize(objective, n_trials=n_trials)

    best_hp = pd.DataFrame([{
            'lr': study.best_params['lr'],
            'weight_decay': study.best_params['weight_decay'],
            'dropout_rate': study.best_params['dropout_rate']
        }])
    best_hp.to_csv(os.path.join(logdir, 'best_hp.csv'))

    print("Optimisation terminée")
    print(f"Base de données créée : {db_name}")
    print("Best hyperparameters:", study.best_params)

if __name__ == '__main__':
    main()



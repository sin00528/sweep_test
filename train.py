import os
import random
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# My model
from module import MyLightningModule, MyDataModule

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_model():
    wandb.init(project="sweep")
    config = wandb.config
    wandb_logger = WandbLogger()
    data = MyDataModule(config)
    module = MyLightningModule(config)

    wandb_logger.watch(module.model)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=10,
                         default_root_dir="./lightning-example",
                         logger=wandb_logger)

    trainer.fit(module, data)
    

if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'model_name': {
                'values': ['resnet34', 'efficientnet_b0']
            },
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            'learning_rate': {
                'distribution': 'uniform', 'min': 1e-5, 'max': 1e-1
            },
        }
    }
    seed_everything(42)
    torch.set_float32_matmul_precision('high')
    sweep_id = wandb.sweep(sweep_config, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=10)
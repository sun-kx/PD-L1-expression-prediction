import gc
import os
import wandb
import argparse
import warnings
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import time
import numpy as np

import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from classifier import ClassifierLightning
from data import MILDataset, get_multi_cohort_df
from options import Options
from utils import save_results

"""
train and validate a model with nested k-fold cross validation without evaluating on the test set.

k-fold cross validation (k=5)
[--|--|--|**|##]
[--|--|**|##|--]
[--|**|##|--|--]
[**|##|--|--|--]
[##|--|--|--|**]
where 
-- train
** val
## test
"""



# filter out UserWarnings from the torchmetrics package
warnings.filterwarnings("ignore", category=UserWarning)


def main(cfg):
    cfg.seed = torch.randint(0, 1000, (1, )).item() if cfg.seed is None else cfg.seed
    pl.seed_everything(cfg.seed, workers=True)

    # --------------------------------------------------------
    # set up paths
    # --------------------------------------------------------

    base_path = Path(cfg.save_dir)
    cfg.logging_name = f'{cfg.name}_{"-".join(cfg.cohorts)}' if cfg.name != 'debug' else 'debug'
    base_path = base_path / cfg.logging_name
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = base_path / 'models'
    result_path = base_path / 'results'
    result_path.mkdir(parents=True, exist_ok=True)

    norm_val = 'raw' if cfg.norm in ['histaugan', 'efficient_histaugan'] else cfg.norm

    # --------------------------------------------------------
    # load data
    # --------------------------------------------------------

    print('\n--- load dataset ---')
    data, clini_info = get_multi_cohort_df(
        cfg.data_config,
        cfg.cohorts, cfg.target,
        cfg.label_dict,
        norm=cfg.norm,
        feats=cfg.feats,
        clini_info=cfg.clini_info
    )
    cfg.clini_info = clini_info
    cfg.input_dim += len(cfg.clini_info.keys())

    train_cohorts = f'{", ".join(cfg.cohorts)}'
    val_cohorts = [train_cohorts]
    results_validation = {t: [] for t in val_cohorts}

    # --------------------------------------------------------
    # k-fold cross validation
    # --------------------------------------------------------

    # load fold directory from data_config
    with open(cfg.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        # fold_path = Path(data_config[train_cohorts]['folds']) / f"{cfg.folds}folds"
        fold_path = Path(data_config[train_cohorts]['folds']) / f"folds"
        fold_path.mkdir(parents=True, exist_ok=True)

    # # split data stratified by the labels
    # skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    # patient_df = data.groupby('PATIENT').first().reset_index()
    # target_stratisfy = cfg.target if type(cfg.target) is str else cfg.target[0]
    # splits = skf.split(patient_df, patient_df[target_stratisfy])
    # splits = list(splits)
    for k in range(cfg.folds):
        # read split from csv-file if exists already else save split to csv
        if Path(fold_path / f'fold{k}_train.csv').exists():

            train_patient = pd.read_csv(fold_path / f'fold{k}_train.csv')
            val_patient = pd.read_csv(fold_path / f'fold{k}_val.csv')
            # 使用 .isin() 方法查找 PATIENT 列中是否包含 train_patient 中的值

            train_idxs = data[data['PATIENT'].isin(train_patient['PATIENT'])].index
            val_idxs = data[data['PATIENT'].isin(val_patient['PATIENT'])].index
            # train_idxs = pd.read_csv(fold_path / f'fold{k}_train.csv', index_col='Unnamed: 0').index
            # val_idxs = pd.read_csv(fold_path / f'fold{k}_val.csv', index_col='Unnamed: 0').index

        else:
            train_idxs, val_idxs = train_test_split(
                splits[k][0],
                stratify=patient_df.iloc[splits[k][0]][target_stratisfy],
                random_state=cfg.seed
            )
            patient_df['PATIENT'][train_idxs].to_csv(fold_path / f'fold{k}_train.csv')
            patient_df['PATIENT'][val_idxs].to_csv(fold_path / f'fold{k}_val.csv')
            patient_df['PATIENT'][splits[k][1]].to_csv(fold_path / f'fold{k}_test.csv')

        train_dataset = MILDataset(
            data,
            train_idxs, [cfg.target],
            num_tiles=cfg.num_tiles,
            pad_tiles=cfg.pad_tiles,
            norm=cfg.norm,
            clini_info=cfg.clini_info,
            cls=cfg.cls,)
        print(f'num training samples in fold {k}: {len(train_dataset)}')




        # class weighting for binary classification
        if cfg.task == 'binary':
            # 计算类别 0 和类别 1 的样本数量
            num_pos = sum([train_dataset[i][2] for i in range(len(train_dataset))])  # 类别 1 的样本数量
            num_neg = len(train_dataset) - num_pos  # 类别 0 的样本数量
            if cfg.pos_weight is None:
                cfg.pos_weight = torch.Tensor(num_neg / num_pos)
            else:
                cfg.pos_weight = torch.tensor(cfg.pos_weight)
            # if cfg.class_weight is None:
            #     cfg.class_weight = torch.Tensor([[1.0, (len(train_dataset) - num_pos) / num_pos]])
            # else:
            #     cfg.class_weight = torch.tensor(cfg.class_weight)

            # 为类别 0 和类别 1 设置权重
            # # 假设你想将类别 0 的权重调低 5 倍
            # weight_0 = 1.0 / 5  # 类别 0 的权重调低 5 倍
            # weight_1 = 1.0  # 类别 1 的权重保持不变
            #
            # # 根据类别数量计算权重，类别数量少的权重大
            # weight_0 = 1.0 / num_neg  # 类别 0 的权重
            # weight_1 = 1.0 / num_pos  # 类别 1 的权重
            #
            # weight_0 = num_pos / num_neg  # 类别 0 的权重
            # weight_1 = 1.0    # 类别 1 的权重
            #
            # # 创建每个样本的权重
            # samples_weight = np.array(
            #     [weight_0 if train_dataset[i][2] == 0 else weight_1 for i in range(len(train_dataset))])
            # # 转换为 Tensor
            # samples_weight = torch.from_numpy(samples_weight).double()
            # # 创建 WeightedRandomSampler
            # sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            # # 创建 train_dataloader 时使用 sampler
            # train_dataloader = DataLoader(
            #     dataset=train_dataset,
            #     batch_size=cfg.bs,
            #     sampler=sampler,  # 使用 sampler 替代 shuffle
            #     num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
            #     pin_memory=True)

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.bs,
            shuffle=True,
            num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
            pin_memory=True)


        # validation dataset
        val_dataset = MILDataset(
            data, val_idxs, [cfg.target], norm=norm_val, clini_info=cfg.clini_info, cls=cfg.cls,)
        print(f'num validation samples in fold {k}: {len(val_dataset)}')

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
            pin_memory=True)

        # if len(train_dataset) < cfg.val_check_interval:
        #     cfg.val_check_interval = len(train_dataset)

        if cfg.lr_scheduler == 'OneCycleLR':
            cfg.lr_scheduler_config['total_steps'] = cfg.num_epochs * len(train_dataloader)

        # --------------------------------------------------------
        # model
        # --------------------------------------------------------
        torch.set_float32_matmul_precision('high')

        model = ClassifierLightning(cfg)
        # --------------------------------------------------------
        # logging
        # --------------------------------------------------------

        logger = WandbLogger(
            project=cfg.project,
            name=f'{cfg.logging_name}_fold{k}',
            save_dir=cfg.save_dir,
            reinit=True,
            settings=wandb.Settings(start_method='fork'),
            mode='online'
        )

        csv_logger = CSVLogger(
            save_dir=result_path,
            name=f'fold{k}',
        )

        # --------------------------------------------------------
        # model saving
        # --------------------------------------------------------

        checkpoint_callback = ModelCheckpoint(
            monitor=cfg.stop_criterion, #if cfg.stop_criterion == 'auroc' else 'loss/val',
            dirpath=model_path,
            filename=f'best_model_{cfg.logging_name}_fold{k}',
            save_top_k=-1,
            mode='min' if cfg.stop_criterion == 'loss/val' else 'max',
        )

        # --------------------------------------------------------
        # set up trainer
        # --------------------------------------------------------

        trainer = pl.Trainer(
            logger=[logger, csv_logger],
            # logger=[csv_logger],
            accelerator='auto',
            devices=1,
            callbacks=[checkpoint_callback],
            max_epochs=cfg.num_epochs,
            # val_check_interval=cfg.val_check_interval,
            # check_val_every_n_epoch=None,
            check_val_every_n_epoch=cfg.check_val_every_n_epoch,

            enable_model_summary=False,
        )


        # --------------------------------------------------------
        # training
        # --------------------------------------------------------

        if Path(model_path / f'best_model_{cfg.logging_name}_fold{k}.pth').exists():
            pass
        else:
            results_val = trainer.fit(
                model,
                train_dataloader,
                val_dataloader,
            )
            logger.log_table('results/val', results_val)


        # --------------------------------------------------------
        # validating
        # --------------------------------------------------------

        print("Evaluating: ", val_cohorts)
        results_val_final = trainer.validate(
            model,
            val_dataloader,
            ckpt_path='best',
        )
        results_validation[val_cohorts[0]].append(results_val_final[0])
        # 清理缓存
        # 清理缓存
        del val_dataloader, train_dataloader, model, trainer, logger, csv_logger, checkpoint_callback
        # del val_dataloader, train_dataloader, model, trainer, csv_logger, checkpoint_callback
        torch.cuda.empty_cache()
        gc.collect()
        wandb.finish()  # required for new wandb run in next fold


    # save results to csv file
    save_results(cfg, results_validation, base_path, train_cohorts, val_cohorts, mode="val")


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'  # 选择使用编号为1的GPU设备
    os.environ["WANDB_API_KEY"] = 'c7d87dcf7ee5d7770d6d1c30c217b23342e71ec1'
    # os.environ["WANDB_MODE"] = "offline"
    parser = Options()
    args = parser.parse()

    # Load the configuration from the YAML file
    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration with the values from the argument parser
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name != 'config_file':
            config[arg_name] = getattr(args, arg_name)

    print('\n--- load options ---')
    for name, value in sorted(config.items()):
        print(f'{name}: {str(value)}')

    config = argparse.Namespace(**config)
    main(config)

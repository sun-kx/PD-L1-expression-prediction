import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from matplotlib import pyplot as plt

from utils import get_loss, get_model, get_optimizer, get_scheduler, LabelSmoothingLoss, FocalLoss


class ClassifierLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_model(
            self.config.model,
            num_classes=self.config.num_classes,
            input_dim=config.input_dim,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            pool=config.pool,
            dim_head=config.dim_head,
            dropout=config.dropout,
            cls=config.cls,
            # **self.config.model_config,
        )
        if config.criterion == "LabelSmoothingLoss":
            self.criterion = LabelSmoothingLoss()
        elif config.criterion == "FocalLoss":
            self.criterion = FocalLoss(pos_weight=config.pos_weight)
            # self.criterion = FocalLoss()
        else:
            if config.task == "binary":
                print(config.pos_weight)
                self.criterion = get_loss(config.criterion,  pos_weight=config.pos_weight, )
                # self.criterion = get_loss(config.criterion)
            else:
                self.criterion = get_loss(config.criterion)
        #self.criterion = get_loss(config.criterion) if config.task == "binary" else get_loss(config.criterion)
        # if config.criterion == "binary"  BCEWithLogitsLossBCEWithLogitsLoss, CrossEntropyLoss:
        #     self.criterion = get_loss(config.criterion)
        self.reg_criterion = get_loss(config.reg_criterion) if config.task == "binary" else get_loss(config.reg_criterion)
        self.save_hyperparameters()

        self.lr = config.lr
        self.wd = config.wd

        self.acc_train = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=config.task, num_classes=config.num_classes)

        self.auroc_val = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.auroc_test = torchmetrics.AUROC(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.f1_val = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.f1_test = torchmetrics.F1Score(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.precision_val = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.precision_test = torchmetrics.Precision(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.recall_val = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.recall_test = torchmetrics.Recall(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.specificity_val = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )
        self.specificity_test = torchmetrics.Specificity(
            task=config.task,
            num_classes=config.num_classes,
        )

        self.cm_val = torchmetrics.ConfusionMatrix(task=config.task, num_classes=config.num_classes)
        self.cm_test = torchmetrics.ConfusionMatrix(
            task=config.task, num_classes=config.num_classes
        )

    def forward(self, x, *args):
        logits = self.model(x, *args)
        return logits

    def configure_optimizers(self, ):
        optimizer = get_optimizer(
            name=self.config.optimizer,
            model=self.model,
            lr=self.lr,
            wd=self.wd,
        )
        if self.config.lr_scheduler:
            scheduler = get_scheduler(
                self.config.lr_scheduler,
                optimizer,
                **self.config.lr_scheduler_config,
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        if self.config.cls == 'class':
            x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
            logits = self.forward(x, coords)
            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)
                self.acc_train(cls_preds, y)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            loss, probs, preds = cls_loss, cls_probs, cls_preds

        elif self.config.cls == 'reg' or self.config.cls == 'class+reg':
            x, coords, y, y_reg, _, _ = batch  # x = features, coords, y = labels, tiles, patient
            logits, reg_logits = self.forward(x, coords)

            # 回归头的损失 (假设你使用MSELoss或类似的损失函数)
            reg_loss = self.reg_criterion(reg_logits, y_reg.float())
            # 对回归结果进行平移，使得 42 对应 0.5 的概率
            reg_probs = torch.sigmoid(reg_logits - 42)
            # 使用 reg_probs 进行分类
            reg_preds = torch.where(reg_probs >= 0.5,
                                    torch.tensor(1, device=self.device),
                                    torch.tensor(0, device=self.device))

            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)

                # 综合分类头和回归头的预测进行类别决定
                # 你可以使用分类头和回归头预测的加权和，或者其他方式结合
                final_probs = (cls_probs + reg_probs) // 2  # 简单平均，两者共同决定类别
                final_preds = torch.round(final_probs)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            # 计算总损失 (加权的分类损失和回归损失)
            total_loss = 1.0 * cls_loss + 1.0 * reg_loss  # 可以加权，如果某一头的损失更加重要

            if self.config.cls == 'reg':
                loss, probs, preds = reg_loss, reg_probs, reg_preds
            else:
                if self.config.task == "binary":
                    loss, probs, preds = total_loss, final_probs, final_preds
                else:
                    loss, probs, preds = total_loss, cls_probs, cls_preds

        self.acc_train(preds, y)
        self.log("acc/train", self.acc_train, prog_bar=True)
        self.log("loss/train", loss, prog_bar=False)

        return loss


    def validation_step(self, batch, batch_idx):
        if self.config.cls == 'class':
            x, coords, y, _, _ = batch  # x = features, coords, y = labels, tiles, patient
            logits = self.forward(x, coords)
            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)
                self.acc_train(cls_preds, y)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            loss, probs, preds = cls_loss, cls_probs, cls_preds

        elif self.config.cls == 'reg' or self.config.cls == 'class+reg':
            x, coords, y, y_reg, _, _ = batch  # x = features, coords, y = labels, tiles, patient
            logits, reg_logits = self.forward(x, coords)

            # 回归头的损失 (假设你使用MSELoss或类似的损失函数)
            reg_loss = self.reg_criterion(reg_logits, y_reg.float())
            # 对回归结果进行平移，使得 42 对应 0.5 的概率
            reg_probs = torch.sigmoid(reg_logits - 42)
            # 使用 reg_probs 进行分类
            reg_preds = torch.where(reg_probs >= 0.5,
                                    torch.tensor(1, device=self.device),
                                    torch.tensor(0, device=self.device))

            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)

                # 综合分类头和回归头的预测进行类别决定
                # 你可以使用分类头和回归头预测的加权和，或者其他方式结合
                final_probs = (cls_probs + reg_probs) // 2  # 简单平均，两者共同决定类别
                final_preds = torch.round(final_probs)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            # 计算总损失 (加权的分类损失和回归损失)
            total_loss = 1.0 * cls_loss + 1.0 * reg_loss  # 可以加权，如果某一头的损失更加重要

            if self.config.cls == 'reg':
                loss, probs, preds = reg_loss, reg_probs, reg_preds
            else:
                if self.config.task == "binary":
                    loss, probs, preds = total_loss, final_probs, final_preds
                else:
                    loss, probs, preds = total_loss, cls_probs, cls_preds

        self.acc_val(preds, y)
        self.auroc_val(probs, y.squeeze(1))
        self.f1_val(preds, y)
        self.precision_val(preds, y)
        self.recall_val(preds, y)
        self.specificity_val(preds, y)

        self.cm_val(preds, y)

        self.log("loss/val", loss, prog_bar=True)
        self.log("acc/val", self.acc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/val", self.auroc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/val", self.f1_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision/val", self.precision_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log("recall/val", self.recall_val, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity/val", self.specificity_val, prog_bar=False, on_step=False, on_epoch=True
        )

    def on_validation_epoch_end(self):
        if self.global_step != 0:
            cm = self.cm_val.compute()
            print(cm)
            # normalise the confusion matrix
            norm = cm.sum(axis=1, keepdims=True)
            normalized_cm = cm / norm
            # log to wandb
            plt.clf()
            cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
            wandb.log({"confusion_matrix/val": wandb.Image(cm)})

        self.cm_val.reset()

    def on_test_epoch_start(self) -> None:
        # save test outputs in dataframe per test dataset
        column_names = ['patient', 'ground_truth', 'predictions', 'logits', 'correct']
        self.outputs = pd.DataFrame(columns=column_names)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.config.cls == 'class':
            x, coords, y, _, patient = batch  # x = features, coords, y = labels, tiles, patient
            logits = self.forward(x, coords)
            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)
                self.acc_train(cls_preds, y)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            loss, probs, preds = cls_loss, cls_probs, cls_preds

        elif self.config.cls == 'reg' or self.config.cls == 'class+reg':
            x, coords, y, y_reg, _, patient = batch  # x = features, coords, y = labels, tiles, patient
            logits, reg_logits = self.forward(x, coords)

            # 回归头的损失 (假设你使用MSELoss或类似的损失函数)
            reg_loss = self.reg_criterion(reg_logits, y_reg.float())
            # 对回归结果进行平移，使得 42 对应 0.5 的概率
            reg_probs = torch.sigmoid(reg_logits - 42)
            # 使用 reg_probs 进行分类
            reg_preds = torch.where(reg_probs >= 0.5,
                                    torch.tensor(1, device=self.device),
                                    torch.tensor(0, device=self.device))

            if self.config.task == "binary":
                y = y.unsqueeze(1)
                cls_loss = self.criterion(logits, y.float())
                cls_probs = torch.sigmoid(logits)
                cls_preds = torch.round(cls_probs)

                # 综合分类头和回归头的预测进行类别决定
                # 你可以使用分类头和回归头预测的加权和，或者其他方式结合
                final_probs = (cls_probs + reg_probs) // 2  # 简单平均，两者共同决定类别
                final_preds = torch.round(final_probs)
            else:
                # 多分类任务
                cls_loss = self.criterion(logits, y)
                cls_probs = torch.softmax(logits, dim=1)  # 获取 softmax 概率
                cls_preds = torch.argmax(cls_probs, dim=1, keepdim=True)  # 使用 argmax 获取预测类别
                y = y.unsqueeze(1)
            # 计算总损失 (加权的分类损失和回归损失)
            total_loss = 1.0 * cls_loss + 1.0 * reg_loss  # 可以加权，如果某一头的损失更加重要

            if self.config.cls == 'reg':
                loss, probs, preds = reg_loss, reg_probs, reg_preds
            else:
                if self.config.task == "binary":
                    loss, probs, preds = total_loss, final_probs, final_preds
                else:
                    loss, probs, preds = total_loss, cls_probs, cls_preds

        self.acc_test(preds, y)
        self.auroc_test(probs, y.squeeze(1))
        self.f1_test(preds, y)
        self.precision_test(preds, y)
        self.recall_test(preds, y)
        self.specificity_test(preds, y)
        self.cm_test(preds, y)
        self.log("loss/test", loss, prog_bar=False)
        self.log("acc/test", self.acc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("auroc/test", self.auroc_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log("f1/test", self.f1_test, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "precision/test", self.precision_test, prog_bar=False, on_step=False, on_epoch=True
        )
        self.log("recall/test", self.recall_test, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "specificity/test", self.specificity_test, prog_bar=False, on_step=False, on_epoch=True
        )

        outputs = pd.DataFrame(
            data=[
                [patient[0],
                 y.item(),
                 preds.item(),
                 logits.squeeze(), (y == preds).int().item()]
            ],
            columns=['patient', 'ground_truth', 'prediction', 'logits', 'correct']
        )
        self.outputs = pd.concat([self.outputs, outputs], ignore_index=True)

    def on_test_epoch_end(self):
        # if self.global_step != 0:
        cm = self.cm_test.compute()
        print(cm)
        # normalise the confusion matrix
        norm = cm.sum(axis=1, keepdims=True)
        normalized_cm = cm / norm

        # log to logger
        plt.clf()
        cm = sns.heatmap(normalized_cm.cpu(), annot=cm.cpu(), cmap='rocket_r', vmin=0, vmax=1)
        # wandb.log({"confusion_matrix/test": wandb.Image(cm)})

        self.cm_test.reset()


import torch
import torch.nn as nn
import typing
import torchvision
import pytorch_lightning as pl

from torch import Tensor
from model.vig import PyramidViG
from typing import Tuple, List
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)

class PyramidViGLT(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 heads: int,
                 n_classes: int,
                 input_resolution: Tuple[int, int],
                 reduce_factor: int,
                 pyramid_reduction: int = 2,
                 act: str = 'relu',
                 k: int = 4,
                 overlapped_patch_emb: bool = True,
                 relative_positional_embedding: bool = True,
                 grapher_layer: str = 'default',
                 dataset: str = None,
                 batch_size: int = -1,
                 **kwargs) -> None:
        super(PyramidViGLT, self).__init__()

        self.model = PyramidViG(in_channels,
                                out_channels,
                                heads,
                                n_classes,
                                input_resolution,
                                reduce_factor,
                                pyramid_reduction,
                                act,
                                k,
                                overlapped_patch_emb,
                                relative_positional_embedding,
                                grapher_layer,
                                **kwargs)
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.dataset = dataset
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        return

    # Runs the prediction + evaluation step for training/validation/testing.
    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch["image"]
        targets = batch["label"]
        if self.in_channels == 3:
            indices_rgb = torch.tensor([3, 2, 1]).to("cuda")
            inputs = torch.index_select(input=inputs, dim=1, index=indices_rgb)
        logits = self.forward(inputs)
        logits = logits.squeeze(-1).squeeze(-1)
        loss = self.loss_fn(logits, targets.float())
        return {"loss": loss, "targets": targets, "logits": logits}

    def _generic_epoch_end(self, step_outputs):
        all_targets = []
        all_preds = []
        all_loss = []
        for outputs in step_outputs:
            logits = outputs["logits"]
            targets = outputs["targets"]
            preds = torch.sigmoid(logits) > 0.5
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.type(targets.dtype).cpu().numpy())

            loss = outputs["loss"]
            all_loss.append(loss.cpu().detach().numpy())

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true=all_targets,
            y_pred=all_preds,
            average="micro",
            zero_division=0,
        )
        avg_loss = sum(all_loss) / len(all_loss)
        report = classification_report(
            y_true=all_targets,
            y_pred=all_preds,
            zero_division=0,
        )

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "report": report,
            "loss": avg_loss,
        }
        return metrics

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        print('\n')
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/train",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.training_step_outputs.append(outputs)
        return outputs

    def on_train_epoch_end(self):
        metrics = self._generic_epoch_end(self.training_step_outputs)
        self.log_metrics(metrics, split="train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/val",
            outputs["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            metrics = self._generic_epoch_end(self.validation_step_outputs)
            self.val_metrics = metrics  # cache for use in callback
            self.log_metrics(metrics, split="val")
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Runs a predictionval_ step for testing, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        metrics = self._generic_epoch_end(self.test_step_outputs)
        self.test_metrics = metrics
        self.log_metrics(metrics, split="test")
        self.test_step_outputs.clear()

    def log_metrics(self, metrics: typing.Dict, split: str):
        assert split in ["train", "val", "test"]

        if split in ["train", "val", "test"]:
            self.log(f"precision/{split}", metrics["precision"], on_epoch=True)
            self.log(f"recall/{split}", metrics["recall"], on_epoch=True)
            self.log(f"f1_score/{split}", metrics["f1_score"], on_epoch=True)



    

class Resnet101LT(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 num_classes : int,
                 batch_size: int = -1,
                 **kwargs) -> None:
        super(Resnet101LT, self).__init__()
        self.model = torchvision.models.resnet101(weights=None, num_classes = num_classes,**kwargs)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.batch_size = batch_size
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        return

    # Runs the prediction + evaluation step for training/validation/testing.
    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch["image"]
        targets = batch["label"]
        logits = self.forward(inputs)
        logits = logits.squeeze(-1).squeeze(-1)
        loss = self.loss_fn(logits, targets.float())
        return {"loss": loss, "targets": targets, "logits": logits}

    def _generic_epoch_end(self, step_outputs):
        all_targets = []
        all_preds = []
        all_loss = []
        for outputs in step_outputs:
            logits = outputs["logits"]
            targets = outputs["targets"]
            preds = torch.sigmoid(logits) > 0.5
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.type(targets.dtype).cpu().numpy())

            loss = outputs["loss"]
            all_loss.append(loss.cpu().detach().numpy())

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true=all_targets,
            y_pred=all_preds,
            average="micro",
            zero_division=0,
        )
        avg_loss = sum(all_loss) / len(all_loss)
        report = classification_report(
            y_true=all_targets,
            y_pred=all_preds,
            zero_division=0,
        )

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "report": report,
            "loss": avg_loss,
        }
        return metrics

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        print('\n')
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/train",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.training_step_outputs.append(outputs)
        return outputs

    def on_train_epoch_end(self):
        metrics = self._generic_epoch_end(self.training_step_outputs)
        self.log_metrics(metrics, split="train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/val",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            metrics = self._generic_epoch_end(self.validation_step_outputs)
            self.val_metrics = metrics  # cache for use in callback
            self.log_metrics(metrics, split="val")
            self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Runs a predictionval_ step for testing, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        metrics = self._generic_epoch_end(self.test_step_outputs)
        self.test_metrics = metrics
        self.log_metrics(metrics, split="test")
        self.test_step_outputs.clear()

    def log_metrics(self, metrics: typing.Dict, split: str):
        assert split in ["train", "val", "test"]


        if split in ["train", "val", "test"]:
            self.log(f"precision/{split}", metrics["precision"], on_epoch=True)
            self.log(f"recall/{split}", metrics["recall"], on_epoch=True)
            self.log(f"f1_score/{split}", metrics["f1_score"], on_epoch=True)
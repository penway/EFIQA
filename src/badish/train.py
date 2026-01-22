import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from badish import dataset, model, loss  # use to register
from badish.utils.utils import set_seeds, log_val_grid
from badish.utils.registry import (
    ModelRegistry,
    DatasetRegistry,
    OptimRegistry,
    LossRegistry,
) 


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, writer, cfg):
    model.train()
    device = cfg.trainer.device
    running_loss = 0.0
    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        outputs = model(X)

        # if mismatch squeeze
        if outputs.ndim != y.ndim:
            outputs = outputs.squeeze()
            y = y.squeeze()
        # if output res is larger than y res, it is okay, we use adaptive avg pool
        if outputs.shape[-2:] != y.shape[-2:]:
            # outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, y.shape[-2:])
            # scale up the y to match outputs, with nearest
            y = F.interpolate(y, size=outputs.shape[-2:], mode="nearest")

        loss = criterion(outputs, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step = epoch * len(dataloader) + idx
        if global_step % cfg.trainer.log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
    epoch_loss = running_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)


def train_one_epoch_pu(model, loader_P, loader_U, optimizer, 
                       criterion, epoch, writer, cfg):
    """ Positive-Unlabeled learning """
    model.train()
    steps = min(len(loader_P), len(loader_U))
    iter_P, iter_U = iter(loader_P), iter(loader_U)
    running_loss = 0.0
    device = cfg.trainer.device

    for idx in range(steps):
        Xp = next(iter_P)
        Xu = next(iter_U)
        Xp, Xu = Xp.to(device), Xu.to(device)

        logits_p = model(Xp).squeeze()
        logits_u = model(Xu).squeeze()

        loss, stats = criterion(logits_p, logits_u)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        running_loss += loss.item()
        global_step = epoch * steps + idx

        if global_step % cfg.trainer.log_interval == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            with torch.no_grad():
                prob_p = torch.sigmoid(logits_p).mean().item()
                prob_u = torch.sigmoid(logits_u).mean().item()
            writer.add_scalar("Train/Prob_Pos", prob_p, global_step)
            writer.add_scalar("Train/Prob_Unl", prob_u, global_step)

            for k, v in stats.items():
                val = float(v.detach().item() if hasattr(v, "detach") else float(v))
                writer.add_scalar(f"Train/{k}", val, global_step)

    epoch_loss = running_loss / steps
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)
    logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


@torch.inference_mode()
def validate(model, dataset, epoch, writer, cfg):
    model.eval()
    # random select cfg.trainer.val_samples samples from dataset
    indices = np.random.choice(len(dataset), cfg.valdata.val_sample_num, replace=False)

    Is, Xs, Ms = [], [], []
    for i in indices:
        I, X, M = dataset[i]
        X = torch.tensor(X, dtype=torch.float32)
        M = torch.tensor(M, dtype=torch.uint8)
        Is.append(I)
        Xs.append(X)
        Ms.append(M)

    Xs = torch.stack(Xs, 0).to(cfg.trainer.device)
    Ms = torch.stack(Ms, 0).to(cfg.trainer.device)

    fig = model.visualize(Is, Xs, Ms, img_cols=cfg.valdata.val_img_cols)
    writer.add_figure("Val/Vis", fig, global_step=epoch, close=True)
    plt.close(fig)


@torch.inference_mode()
def test_seg(model, dataloader, epoch, writer, cfg):
    model.eval()
    outputs = []
    labels = []

    for image, label, feat in dataloader:
        image = image.to(cfg.trainer.device)
        label = label.to(cfg.trainer.device)
        feat = feat.to(cfg.trainer.device)

        # if model has method '_image_apply', use it
        if hasattr(model, "_image_apply"):
            logits = model._image_apply(feat)
        else:
            logits = model(feat)
        probs = torch.sigmoid(logits)

        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs.squeeze(1)

        outputs.append(probs.cpu())
        labels.append(label.cpu())

    outputs = torch.cat(outputs, 0)
    labels = torch.cat(labels, 0)

    label_thr= cfg.testdata.label_threshold
    bin_labels = (labels >= label_thr)

    # then find the best threshold to binarize outputs
    best_dice = 0.0
    best_out_t = 0.0
    for t in np.arange(0.0, 1.0, 0.05):
        preds = (outputs >= t)

        tp = (preds & bin_labels).sum().item()
        fp = (preds & ~bin_labels).sum().item()
        fn = (~preds & bin_labels).sum().item()
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        if dice > best_dice:
            best_dice = dice
            best_out_t = float(t)

    # provide acc, pre, rec and f1 at best threshold
    preds = (outputs >= best_out_t).long()
    tp = (preds & bin_labels).sum().item()
    fp = (preds & ~bin_labels).sum().item()
    fn = (~preds & bin_labels).sum().item()
    tn = (~preds & ~bin_labels).sum().item()

    N = tp + fp + fn + tn

    acc = (tp + tn) / (N + 1e-8)
    pre = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1  = (2 * pre * rec) / (pre + rec + 1e-8)

    logging.info(
        f"Test Epoch {epoch+1}" 
        f"Best Dice: {best_dice:.4f} at threshold {best_out_t:.2f}"
    )
    writer.add_scalar("Test/Best_Dice", best_dice, epoch)
    writer.add_scalar("Test/Accuracy", acc, epoch)
    writer.add_scalar("Test/Precision", pre, epoch)
    writer.add_scalar("Test/Recall", rec, epoch)
    writer.add_scalar("Test/F1_Score", f1, epoch)
    writer.add_scalar("Test/Best_Threshold", best_out_t, epoch)


@torch.inference_mode()
def test_control(model, thres_loader, control_loader, epoch, writer, cfg):
    """
    Test on the control set to see how model is activated by disease.
    Lower the better.
    Thres_loader: dataloader for a normal dataset to find best threshold
    then the threshold is applied to control_loader to see how many images are false positive
    the control set is full negative.
    thres_loader: feat, label
    control_loader: feat (no label, all negative)
    this is a classification type evaluation, the score is the mean of output map.
    """
    model.eval()
    # find best threshold on thres_loader
    scores = []
    labels = []
    best_thres = 0.0
    best_f1 = -1
    for feats, label in thres_loader:
        feats = feats.to(cfg.trainer.device)
        label = label.to(cfg.trainer.device)
        # invert the label: 0 means good quality (negative), 1 means bad quality (positive)
        label = 1 - label
        
        if hasattr(model, "_image_apply"):
            logits = model._image_apply(feats)
        else:
            logits = model(feats)
        probs = torch.sigmoid(logits)
        score = probs.view(probs.size(0), -1).mean(dim=1)
        scores.extend(score.cpu().tolist())
        labels.extend(label.view(-1).cpu().tolist())

    for t in np.arange(0.0, 1.0, 0.01):
        tp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= t and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s < t and l == 1)

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = (2 * prec * rec) / (prec + rec + 1e-8)

        if f1 >= best_f1:
            best_f1 = f1
            best_thres = t

    logging.info(
        f"Control Test Epoch {epoch+1}, Best Threshold on thres set: {best_thres:.2f}, Best F1: {best_f1:.4f}"
    )
    writer.add_scalar("Control/Best_Threshold", best_thres, epoch)
    writer.add_scalar("Control/Best_F1", best_f1, epoch)

    # apply best threshold to control_loader
    false_positives = 0
    total = 0
    for feats in control_loader:
        feats = feats.to(cfg.trainer.device)

        if hasattr(model, "_image_apply"):
            logits = model._image_apply(feats)
        else:
            logits = model(feats)
        probs = torch.sigmoid(logits)
        score = probs.view(probs.size(0), -1).mean(dim=1)
        false_positives += (score >= best_thres).sum().item()
        total += score.numel()

    false_positive_rate = false_positives / (total + 1e-8)
    logging.info(
        f"Control Test Epoch {epoch+1}, threshold: {best_thres:.2f}, "
        f"False Positive Rate: {false_positive_rate:.4f}"
    )
    writer.add_scalar("Control/False_Positive_Rate", false_positive_rate, epoch)


@hydra.main(config_path="config", version_base=None)
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, log_dir / "config.yaml")

    logging.info(f"Training started with cfg: {cfg}")

    writer = SummaryWriter(log_dir=log_dir)
    set_seeds(cfg.trainer.seed)

    valset    = DatasetRegistry.get(cfg.valdata.key, **cfg.valdata.params)
    model     = ModelRegistry.get(cfg.model.key, **cfg.model.params).to(cfg.trainer.device)
    optimizer = OptimRegistry.get(cfg.optim.key, model.parameters(), **cfg.optim.params)
    loss      = LossRegistry.get(cfg.loss.key, **cfg.loss.params)

    testset   = None
    if cfg.get("testdata") is not None:
        testset = DatasetRegistry.get(cfg.testdata.key, **cfg.testdata.params)
        testloader = DataLoader(
            testset,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )

    if cfg.get("controldata") is not None:
        controldata_thres = DatasetRegistry.get(
            cfg.controldata.thres_key, **cfg.controldata.thres_params)
        controldata_control = DatasetRegistry.get(
            cfg.controldata.control_key, **cfg.controldata.control_params)
        controldata_thres_loader = DataLoader(
            controldata_thres,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )
        controldata_control_loader = DataLoader(
            controldata_control,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )

    if cfg.loss.key.startswith("nnpu"):
        trainset_P = DatasetRegistry.get(
            cfg.traindata.key, split="P", **cfg.traindata.params)
        trainset_U = DatasetRegistry.get(
            cfg.traindata.key, split="U", **cfg.traindata.params)
        train_loader_P = DataLoader(
            trainset_P,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )
        train_loader_U = DataLoader(
            trainset_U,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )
    else:
        trainset  = DatasetRegistry.get(cfg.traindata.key, **cfg.traindata.params)
        train_loader = DataLoader(
            trainset,
            batch_size=cfg.trainer.batch_size,
            shuffle=True,
            num_workers=cfg.trainer.num_workers,
            pin_memory=True,
        )
    for epoch in range(cfg.trainer.epochs):
        if cfg.loss.key.startswith("nnpu"):
            train_one_epoch_pu(model, train_loader_P, train_loader_U, 
                               optimizer, loss, epoch, writer, cfg)
        else:
            train_one_epoch(model, train_loader, optimizer, loss, epoch, writer, cfg)
        validate(model, valset, epoch, writer, cfg)
        test_seg(model, testloader, epoch, writer, cfg) if testset is not None else None
        test_control(model, controldata_thres_loader, 
                     controldata_control_loader, epoch, writer, cfg) if cfg.get("controldata") is not None else None
        if (epoch + 1) % cfg.trainer.ckpt_interval == 0:
            torch.save(model.state_dict(), log_dir / f"model_epoch{epoch+1}.pth")

    writer.close()
    logging.info("Training completed.")


if __name__ == "__main__":
    main()

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from vuad import model, augment, loss, dataset
from vuad.utils.utils import set_seeds, on_epoch_start_transform
from vuad.utils.registry import ModelRegistry, LossRegistry, DegradeRegistry, OptimRegistry



def train_one_epoch(model, dataloader, degrade, criterion, optimizer, device, epoch, writer, log_interval):
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    running_loss = 0.0
    for idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        masked_images = degrade(images)
        outputs = model(masked_images)
        loss = criterion(outputs, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": loss.item()})
        writer.add_scalar("Train/Loss", loss.item(), epoch * len(dataloader) + idx)
        # if idx % log_interval == 0: 
        # use writer global index for better tracking
        global_step = epoch * len(dataloader) + idx
        if global_step % log_interval == 0:
            writer.add_images("Train/Masked_Images", masked_images, epoch * len(dataloader) + idx)
            writer.add_images("Train/Outputs", outputs, epoch * len(dataloader) + idx)
            writer.add_images("Train/Original_Images", images, epoch * len(dataloader) + idx)
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    writer.add_scalar("Train/Epoch_Loss", epoch_loss, epoch)


@hydra.main(config_path="./config", version_base=None)
def main(cfg: DictConfig):
    log_dir = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(cfg, log_dir / "config.yaml")

    logging.info(f"Training started with cfg: {cfg}")

    writer = SummaryWriter(log_dir=log_dir)

    set_seeds(cfg.trainer.seed)

    fov_enabled = OmegaConf.select(cfg, "data.fov_crop.enabled", default=False)
    if fov_enabled:
        fov_transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=360, fill=0),
            augment.random_fov.RandomFOV(**cfg.data.fov_crop.params)
        ]
    else:
        fov_transform = []
    transform = transforms.Compose(
        fov_transform + [
        transforms.Resize((cfg.data.resize, cfg.data.resize)),
        transforms.ToTensor(),
    ])

    degrade = DegradeRegistry.get( 
        cfg.degrade.key,
        **cfg.degrade.params,
    )

    dataset = ImageFolder(cfg.data.path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers
    )

    model = ModelRegistry.get(
        cfg.model.key, **cfg.model.params 
        ).to(cfg.trainer.device)

    optimizer = OptimRegistry.get(
        cfg.optim.key,
        model.parameters(),
        **cfg.optim.params
    )

    criterion = LossRegistry.get(cfg.loss.key, **cfg.loss.params)

    for epoch in range(cfg.trainer.epochs):
        if hasattr(degrade, "on_epoch_start"):
            degrade.on_epoch_start(epoch)
            logging.info(f"Epoch {epoch+1}, Degrade ratio range: \
                         ({degrade.min_ratio:.2f}, {degrade.max_ratio:.2f})")
        on_epoch_start_transform(transform, epoch)
        
        train_one_epoch(
            model,
            dataloader,
            degrade,
            criterion,
            optimizer,
            cfg.trainer.device,
            epoch,
            writer,
            cfg.trainer.log_interval
        )
        if (epoch + 1) % cfg.trainer.ckpt_interval == 0: 
            torch.save(model.state_dict(), log_dir / f"model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()

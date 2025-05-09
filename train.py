# train.py

import torch
import torch.nn.functional as F
import time
import wandb
import os
from torchmetrics.classification import MulticlassJaccardIndex
from fvcore.nn import FlopCountAnalysis

CHECKPOINT_DIR = "checkpoints"

def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes, epoch=None):
    model.train()
    total_loss = 0.0
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    start_time = time.time()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        preds, _, _ = model(images)

        if preds.shape[-2:] != labels.shape[-2:]:
            preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

        loss = criterion(preds, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        miou_metric.update(preds.argmax(1), labels)

    avg_loss = total_loss / len(dataloader)
    miou = miou_metric.compute().item()
    latency = (time.time() - start_time) / len(dataloader)

    if epoch == 0:
        sample_input = torch.randn(1, 3, 512, 1024).to(device)
        flops = FlopCountAnalysis(model, sample_input)
        total_flops = flops.total() / 1e9
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    else:
        total_flops = None
        params = None

    wandb.log({
        "loss": avg_loss,
        "mIoU": miou,
        "latency_per_batch": latency,
        **({"FLOPs (GFLOPs)": total_flops} if total_flops is not None else {}),
        **({"Parameters (M)": params} if params is not None else {})
    }, step=epoch)

    return {
        "loss": avg_loss,
        "mIoU": miou,
        "latency": latency,
        "FLOPs": total_flops,
        "parameters": params
    }

def train(model, dataloader, optimizer, criterion, device, num_classes, config, model_name, project="cityscapes-segmentation"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")

    start_epoch = 0

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    wandb.init(project=project, name=model_name, config=config, resume="allow")
    num_epochs = config["num_epochs"]
    all_metrics = []

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epoch=epoch
        )
        all_metrics.append(metrics)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()
    return all_metrics

def test(model, dataloader, device, num_classes, model_name, project="cityscapes-segmentation"):

    checkpoint_path = os.path.join("checkpoints", f"{model_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for model '{model_name}' at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path} for evaluation...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)

    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            preds, _, _ = model(images)
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

            miou_metric.update(preds.argmax(1), labels)

    total_latency = time.time() - start_time
    avg_latency = total_latency / len(dataloader)
    final_miou = miou_metric.compute().item()

    wandb.init(project=project, name=f"test_{model_name}", reinit=True)
    wandb.log({
        "Test mIoU": final_miou,
        "Test latency_per_batch": avg_latency
    })
    wandb.finish()

    print(f"[Test] mIoU: {final_miou:.4f}, Avg Latency: {avg_latency:.4f}s")

    return {
        "mIoU": final_miou,
        "latency": avg_latency
    }

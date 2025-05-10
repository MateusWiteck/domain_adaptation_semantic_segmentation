import torch
import torch.nn.functional as F
import time
import wandb
import os
from torchmetrics.classification import MulticlassJaccardIndex
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm

PATH_STORE_RESULTS = "/content/drive/MyDrive/MLDL_PROJECT/results/"
CHECKPOINT_DIR = PATH_STORE_RESULTS + "checkpoints/"

def train_one_epoch(model, dataloader, optimizer, criterion, num_classes, device='cuda', epoch=None):
    model.train()
    total_loss = 0.0
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    start_time = time.time()

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):

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

    return {
        "loss": avg_loss,
        "mIoU": miou,
        "latency": latency,
        "FLOPs": total_flops,
        "parameters": params
    }

def evaluate_model(model, dataloader, num_classes, device='cuda'):
    model.eval()
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)

            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

            loss = criterion(preds, labels.long())
            total_loss += loss.item()
            miou_metric.update(preds.argmax(1), labels)

    avg_loss = total_loss / len(dataloader)
    miou = miou_metric.compute().item()
    return {"val_loss": avg_loss, "val_mIoU": miou}

def train_model(model, dataloader, val_dataloader, optimizer, criterion, num_classes, num_epochs, model_name, device='cuda'):

    if not os.path.exists(PATH_STORE_RESULTS):
        raise FileNotFoundError(f"Path {PATH_STORE_RESULTS} does not exist. Please mount the drive.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pt")

    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f"{current_time}_{model_name}_start_epoch_{start_epoch}"
    wandb.init(project=model_name, name=run_name, resume="allow")

    all_metrics = []

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_classes=num_classes,
            epoch=epoch,
            device=device
        )
        val_metrics = evaluate_model(model, val_dataloader, num_classes, device=device)
        all_metrics.append({**train_metrics, **val_metrics})

        wandb.log({
            "train_loss": train_metrics["loss"],
            "train_mIoU": train_metrics["mIoU"],
            "train_latency_per_batch": train_metrics["latency"],
            "val_loss": val_metrics["val_loss"],
            "val_mIoU": val_metrics["val_mIoU"],
            **({"FLOPs (GFLOPs)": train_metrics["FLOPs"]} if train_metrics["FLOPs"] is not None else {}),
            **({"Parameters (M)": train_metrics["parameters"]} if train_metrics["parameters"] is not None else {})
        }, step=epoch)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    wandb.finish()
    return all_metrics



# TODO: Latency per image
# Evolution of the performace over the validation set over epochs of training
def test_model(model, dataloader, num_classes, model_name, device='cuda'):
    model = model.to(device)
    model.eval()

    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)

    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images) # TODO: Explain in the report why the model do not return a tuple when testing
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = F.interpolate(preds, size=labels.shape[-2:], mode='bilinear', align_corners=True)

            miou_metric.update(preds.argmax(1), labels)

    total_latency = time.time() - start_time
    avg_latency = total_latency / len(dataloader)
    final_miou = miou_metric.compute().item()


    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f"{current_time}_test_{model_name}"
    wandb.init(project=f"test_{model_name}", name=run_name, reinit=True)
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

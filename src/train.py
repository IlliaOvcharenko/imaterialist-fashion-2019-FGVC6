import torch
import torchvision
import time

import numpy as np

from tqdm.notebook import tqdm
from warmup_scheduler import GradualWarmupScheduler

def make_scheduler_step(scheduler,  metrics):
    main_scheduler = scheduler.after_scheduler \
        if isinstance(scheduler, GradualWarmupScheduler) else scheduler
    
    if isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metrics=metrics)
    else:
        scheduler.step()

def generate_visualization(model, device, dataloader):
    model.eval()
    
    _, origins, _ = next(iter(dataloader))
    origins = origins.to(device)
    with torch.no_grad():
        outs = model(origins)
        outs = outs.softmax(1)
        outs = outs.cpu()
        
    return np.hstack(np.hstack(outs))
        
        
        
def train(model, device, criterion, optimizer, dataloader, accumulation_steps, pbar_desc="train phase"):
    model.train()
    train_loss = 0.0
    
    for i, (names, origins, masks) in enumerate(tqdm(dataloader, desc=pbar_desc)):
        
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)
                
        outs = model(origins)
        loss = criterion(outs, masks)
        
        train_loss += loss.item() * num
        
        loss = loss / accumulation_steps
        loss.backward() 
        
        if (i+1) % accumulation_steps == 0:             
            optimizer.step()                            
            optimizer.zero_grad()
            
        
        
    train_loss = train_loss / len(dataloader.sampler)
    return {
        "train_loss": train_loss,
    }

    
def validation(model, device, criterion, metrics, dataloader, pbar_desc="validation phase"):
    model.eval()
    val_loss = 0.0
    val_metrics = {k: 0.0 for k, v in metrics.items()}
    
    for names, origins, masks in tqdm(dataloader, desc=pbar_desc):
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outs = model(origins)
        
            val_loss += criterion(outs, masks).item() * num
            val_metrics = {k: v + metrics[k](masks, outs).item() * num for k, v in val_metrics.items()}
        

    val_loss = val_loss / len(dataloader.sampler)
    val_metrics = {k: v / len(dataloader.sampler) for k, v in val_metrics.items()}
    
    return {
        "val_loss": val_loss,
        **val_metrics,
    }


def fit(
    model,
    device,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    metrics={},
    metrics_monitor="val_loss",
    metrics_lower_is_better=True,
    metrics_initial_best_val=None,
    writer=None,
    writer_add_visualizations=False,
    model_folder=None,
    model_name=None,
    remove_previous_ckpt=True,
    epochs=200,
    initial_epoch=0,
    accumulation_steps=1,
):
    
    if metrics_initial_best_val is None:
        metrics_initial_best_val = np.inf if metrics_lower_is_better else 0.0
    best_val_metric = metrics_initial_best_val
    
    for e in range(initial_epoch, epochs):
        start_t = time.time()

        train_pbar_desc = f"epoch: {e+1}/{epochs}, train phase"
        train_log = train(model, device, criterion, optimizer, dataloaders["train"], accumulation_steps, train_pbar_desc)
        train_loss = train_log["train_loss"]

        val_pbar_desc = f"epoch: {e+1}/{epochs}, validation phase" 
        val_log = validation(model, device, criterion, metrics, dataloaders["val"], val_pbar_desc)
        val_loss = val_log["val_loss"]
        val_metric = val_log[metrics_monitor]
        
        make_scheduler_step(scheduler, val_metric)

        end_t = time.time()
        spended_t = end_t - start_t

        current_lr = [pg for pg in optimizer.param_groups][0]["lr"]

        if writer is not None:
            writer.add_scalar("time", spended_t, e)
            writer.add_scalar('train_loss', train_loss, e)
            writer.add_scalar('lr', current_lr, e)
            for k, v in val_log.items():
                writer.add_scalar(k, v, e)
                
            if writer_add_visualizations:
                writer.add_image(
                    "visualizations",
                    generate_visualization(model, device, dataloaders["val"]),
                    e,
                    dataformats="HW",
                )
            
        report = f"epoch: {e+1}/{epochs}, time: {spended_t}, train loss: {train_loss}, \n"\
               + f"val loss: {val_loss}, val_metric: {val_metric}"
        print(report)

        
        metric_improved = (metrics_lower_is_better and val_metric < best_val_metric) \
                       or (not metrics_lower_is_better and val_metric > best_val_metric)
        
        if metric_improved and model_folder is not None:
            best_val_metric = val_metric            
            
            if model_name is not None:
                checkpoint_name = f"{model_name}-epoch-{e}-ckpt.pt" 
            else:
                checkpoint_name = f"epoch-{e}-ckpt.pt"
                
            if remove_previous_ckpt:
                if model_name is not None:
                    pattern = f"{model_name}-epoch-[0-9]*-ckpt.pt"  
                else:
                    pattern = f"epoch-[0-9]*-ckpt.pt"
    
                for fn in model_folder.glob(pattern):
                    fn.unlink()
            
            torch.save({
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, model_folder / checkpoint_name)
            
            print("checkpoint saved")
        print()
        
        

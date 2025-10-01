import os
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import tqdm

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    # set model to evaluation mode
    model.eval()
    
    losses, accuracies = [], []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc=f'Evaluating test set'):
            inputs = send_inputs_to_device(inputs, device=device)
            outputs = model(**inputs)
            
            losses.append(outputs.loss.item())
            accuracies.append(outputs.accuracy.item())
    
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)
    
    # set model back to training mode
    model.train()
    
    return loss, accuracy


def send_inputs_to_device(dictionary, device):
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value for key, value in dictionary.items()}


# save checkpoints during training (on GPU)
# def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, checkpoint_name):
#     checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }, checkpoint_path)
#     print(f"Checkpoint saved to {checkpoint_path}")

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, checkpoint_dir, checkpoint_name):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


# save the final model (on CPU)
def save_final_model(model, final_model_path):
    model.to('cpu')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    
# load a checkpoint (for resuming training) [on GPU]
# def load_checkpoint(checkpoint_path, model, optimizer):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print(f"Resumed from epoch {epoch}, loss: {loss}")
#     return model, optimizer, epoch, loss

def load_checkpoint(checkpoint_path, model, optimizer, strict=False):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)  # Default to 0 if missing
    train_loss = checkpoint.get('train_loss', None)  # Default to None if missing
    val_loss = checkpoint.get('val_loss', None)
    train_acc = checkpoint.get('train_acc', None)
    val_acc = checkpoint.get('val_acc', None)

    print(f"Resumed from epoch {epoch}")
    if train_loss is not None and val_loss is not None:
        print(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
    if train_acc is not None and val_acc is not None:
        print(f"Train Acc: {train_acc}, Val Acc: {val_acc}")

    return model, optimizer, epoch


# Load the final model (on CPU)
def load_final_model(model, final_model_path):
    state_dict = torch.load(final_model_path, map_location=torch.device("cpu"))
    new_state_dict = {k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model
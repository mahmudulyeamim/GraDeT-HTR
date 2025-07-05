import tqdm
from utils import send_inputs_to_device, evaluate_model, load_checkpoint, save_checkpoint, save_final_model
from dataset import split_data
from model import DTrOCRLMHeadModel
from config import DTrOCRConfig
from torch.utils.data import DataLoader
import torch
import multiprocessing as mp

# Build dataset and dataloader
images_dir = "../sample_train/images"
labels_file = "../sample_train/labels/label.csv"

config = DTrOCRConfig()

train_dataset, validation_dataset = split_data(images_dir, labels_file, config)

print(f"Train size: {len(train_dataset)}")
# print(f"Validation size: {len(validation_dataset)}")
print(f"Test size: {len(validation_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=mp.cpu_count())
# validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=mp.cpu_count())
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=mp.cpu_count())

# attemt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print('using device: ', device)

# ensures reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
torch.set_float32_matmul_precision('high')

# Model
model = DTrOCRLMHeadModel(config)
model = torch.compile(model)
model.to(device)

use_amp = True
scaler = torch.amp.GradScaler(device=device, enabled=use_amp)
# pre train
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
# fine tune
# optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-6)

# Training
EPOCHS = 10
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []

for epoch in range(EPOCHS):
    epoch_losses, epoch_accuracies = [], []
    for inputs in tqdm.tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch + 1}'):
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # send inputs to same device as model
        inputs = send_inputs_to_device(inputs, device=device)
        
        # forward pass
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            outputs = model(**inputs)
        
        # calculate gradients
        scaler.scale(outputs.loss).backward()
        
        # update weights
        scaler.step(optimizer)
        scaler.update()
        
        epoch_losses.append(outputs.loss.item())
        epoch_accuracies.append(outputs.accuracy.item())
    
    # store loss and metrics
    train_losses.append(sum(epoch_losses) / len(epoch_losses))
    train_accuracies.append(sum(epoch_accuracies) / len(epoch_accuracies))
    
    # tests loss and accuracy
    validation_loss, validation_accuracy = evaluate_model(model, validation_dataloader, device=device)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)
    
    print(f"Epoch: {epoch + 1} - Train loss: {train_losses[-1]}, Train accuracy: {train_accuracies[-1]}, Validation loss: {validation_losses[-1]},  Validation accuracy: {validation_accuracies[-1]}")

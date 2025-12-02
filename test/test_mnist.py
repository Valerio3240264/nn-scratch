import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
from pathlib import Path

simulated_batch = 100

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class MNISTDataset(Dataset):
    """Custom dataset for MNIST CSV data"""
    def __init__(self, csv_file, max_samples=None):
        print(f"Loading dataset from {csv_file}...")
        self.data = pd.read_csv(csv_file)
        if max_samples is not None:
            self.data = self.data.iloc[:max_samples]
        
        # Extract labels and features
        self.labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values / 255.0  # Normalize to [0, 1]
        
        # Add bias term (column of ones)
        self.features = np.hstack([self.features, np.ones((len(self.features), 1))])
        
        print(f"Successfully loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return features, label

class MLP(nn.Module):
    """Multi-Layer Perceptron matching the CUDA implementation"""
    def __init__(self, input_size=785, hidden_sizes=[256, 128], output_size=10):
        super(MLP, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Hidden layers with ReLU
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=False))  # No bias since we add it to input
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer (linear)
        layers.append(nn.Linear(prev_size, output_size, bias=False))
        
        self.network = nn.Sequential(*layers)
        
        print(f"Network architecture: {input_size} -> {' -> '.join(map(str, hidden_sizes))} -> {output_size}")
    
    def forward(self, x):
        return self.network(x)

def calculate_accuracy(model, dataloader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return 100.0 * correct / total

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with detailed timing"""
    model.train()
    
    # Timing accumulators
    total_forward_time = 0.0
    total_loss_time = 0.0
    total_backward_time = 0.0
    total_optimizer_time = 0.0
    num_batches = 0
    
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        if (batch_idx + 1) % simulated_batch == 0:    
            optimizer.zero_grad()
        
        # Forward pass (timed)
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        outputs = model(features)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        forward_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        total_forward_time += forward_time
        
        # Compute loss (timed)
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        loss = criterion(outputs, labels)/simulated_batch
        
        if device == 'cuda':
            torch.cuda.synchronize()
        loss_time = (time.perf_counter() - start_time) * 1000
        total_loss_time += loss_time
        
        # Backward pass (timed)
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        loss.backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        backward_time = (time.perf_counter() - start_time) * 1000
        total_backward_time += backward_time
        
        # Optimizer step (timed)
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        if (batch_idx+1) % simulated_batch == 0:
            optimizer.step()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        optimizer_time = (time.perf_counter() - start_time) * 1000
        total_optimizer_time += optimizer_time
        
        # Accumulate loss
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        num_batches += 1
    


    avg_loss = total_loss / total_samples
    
    # Print timing statistics
    print(f"\n--- Timing Statistics (Epoch {epoch + 1}) ---")
    print(f"Mean Forward Pass Time:    {total_forward_time / num_batches:.4f} ms")
    print(f"Mean Compute Loss Time:    {total_loss_time / num_batches:.4f} ms")
    print(f"Mean Backward Pass Time:   {total_backward_time / num_batches:.4f} ms")
    print(f"Mean Optimizer Step Time:  {total_optimizer_time / num_batches:.4f} ms")
    print(f"Total Forward Time:        {total_forward_time:.4f} ms")
    print(f"Total Compute Loss Time:   {total_loss_time:.4f} ms")
    print(f"Total Backward Pass Time:  {total_backward_time:.4f} ms")
    print(f"Total Optimizer Step Time: {total_optimizer_time:.4f} ms")
    total_time = total_forward_time + total_loss_time + total_backward_time + total_optimizer_time
    print(f"Total Epoch Time:          {total_time:.4f} ms ({total_time / 1000:.2f} seconds)")
    print(f"Training Loss: {avg_loss:.6f}")
    print("------------------------------------\n")
    
    return avg_loss

def main():
    # HYPERPARAMETERS (matching CUDA implementation)
    total_samples = 42000
    training_samples = 32000
    test_samples = 10000
    input_size = 785  # 784 pixels + 1 bias
    hidden_sizes = [256, 128]
    output_size = 10
    num_epochs = 5
    batch_size = 1
    learning_rate = 0.01
    
    print("=" * 60)
    print("MNIST Training with PyTorch")
    print("=" * 60)
    
    # Load full dataset
    dataset_path = Path("./test/dataset/train.csv")
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    full_dataset = MNISTDataset(dataset_path, max_samples=total_samples)
    
    # Split into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [training_samples, total_samples - training_samples],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("\nCreating neural network...")
    model = MLP(input_size, hidden_sizes, output_size).to(device)
    print("Network created successfully!")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Includes softmax
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # TESTING BEFORE TRAINING
    print("\n=== TESTING BEFORE TRAINING ===")
    accuracy_before = calculate_accuracy(model, val_loader, device)
    print(f"Accuracy before training: {accuracy_before:.2f}%")
    
    # TRAINING PHASE
    print("\n=== TRAINING PHASE ===")
    total_training_start = time.perf_counter()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} started")
        epoch_start = time.perf_counter()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
    
    total_training_time = time.perf_counter() - total_training_start
    
    # FINAL TESTING
    print("\n=== FINAL TESTING ===")
    accuracy_after = calculate_accuracy(model, val_loader, device)
    print(f"Final accuracy: {accuracy_after:.2f}%")
    print(f"Improvement: {accuracy_after - accuracy_before:.2f}%")
    
    print("\n" + "=" * 60)
    print(f"TOTAL TRAINING TIME: {total_training_time:.2f} seconds")
    print("=" * 60)
    print("\nPyTorch training completed successfully!")

if __name__ == "__main__":
    main()


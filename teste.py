import os
import argparse
from datetime import datetime
import json
import yaml
import tqdm
from PIL import Image
import timm
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
 
import wandb
 
 
class MNISTDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image, label
 
class Model(nn.Module):
    """Example of a model using PyTorch
 
    This specific model is a pretrained ResNet18 model from torch hub
    with a custom head.
 
    """
 
    def __init__(self):
        super().__init__()
 
        self.base = timm.create_model('resnet50', pretrained=True)

        print("self.base", self.base)
        self.base.fc = nn.Sequential(
            nn.Dropout(), nn.Linear(2048, 2)
        )

 
    def forward(self, x):
        x = self.base(x)
        return x
 
 
def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    hyperparameters: dict,
    args: object,
    device: str,
):
    """Train model
 
    Args:
        model (nn.Module): Model to train
        train_dataloader (DataLoader): Dataloader to use for training
        test_dataloader (DataLoader): Dataloader to use for testing
        hyperparameters (dict): Dictionary with hyperparameters
        device (str): Device to use when training the model
    """
    # Create optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=hyperparameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyperparameters["scheduler_step_size"],
        gamma=hyperparameters["scheduler_gamma"],
    )
    criterion = nn.CrossEntropyLoss()
    epoch = checkpoint['epoch']
    for epoch in range(epoch, hyperparameters["epochs"]):
        # Set model to train mode
        model.train()
 
        # Create tqdm progress bar
        with tqdm.tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for (x, y) in train_dataloader:
                # Move batch to device
                x = x.to(device)
                y = y.to(device)
 
                # Forward pass
                y_hat = model(x)
 
                # Compute loss
                loss = criterion(y_hat, y)
 
                # Zero gradients
                optimizer.zero_grad()
 
                # Backward pass
                loss.backward()
 
                # Update weights
                optimizer.step()
 
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
 
                # Log loss to wandb
                #wandb.log({"loss
                # ": loss.item(), "lr": scheduler.get_last_lr()[0]})
 
        # Update learning rate
        scheduler.step()
 
        # Evaluate model
        test(model, test_dataloader, device)
 
        # Save model, scheduler and optimizer checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(args.out_dir, "last.pt"),
        )
 
 
def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    final_eval: bool = False,
):
    """Test model by plotting the embeddings in 2D space
 
    Args:
        model (nn.Module): Model to test
        dataloader (DataLoader): Dataloader to use for testing
    """
    # Create Loss function
    criterion = nn.CrossEntropyLoss()
 
    # Set model to eval mode
    model.eval()
 
    logs_dict = {}
 
    losses = []
 
    # List to store predictions and labels
    predictions = []
    labels = []
 
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for (x, y) in dataloader:
            # Move batch to device
            x = x.to(device)
            y = y.to(device)
 
            # Forward pass
            output = model(x)
            prediction = torch.argmax(output, dim=1)
 
            losses.append(criterion(output, y).item())
            predictions.append(prediction.cpu().numpy())
            labels.append(y.cpu().numpy())
 
        # Concatenate embeddings into a np array of shape: (num_samples, embedding_dim)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
 
        # Compute loss
        loss = np.mean(losses)
        logs_dict["Test Loss"] = loss
 
        # Compute accuracy
        accuracy = (predictions == labels).sum() / labels.shape[0]
        logs_dict["Test Accuracy"] = accuracy
 
        if final_eval:
            # Compute Confusion Matrix
            cm = confusion_matrix(labels, predictions)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            cm_display.figure_.savefig(
                os.path.join(args.out_dir, "confusion_matrix.png")
            )
            logs_dict["Confusion Matrix"] = wandb.Image(
                os.path.join(args.out_dir, "confusion_matrix.png")
            )
 
        # Log to wandb
        #wandb.log(logs_dict)
 
 
if __name__ == "__main__":
    # Define args
    parser = argparse.ArgumentParser(
        description="Example of good practices in an ML script"
    )
    parser.add_argument(
        "--hyperparameters",
        default="example.yml",
        type=str,
        help="Path to yaml file with hyperparameters",
    )
    parser.add_argument(
        "--out_dir",
        default="./output",
        type=str,
        help="Path to the output directory where the model will be saved",
    )
    parser.add_argument(
        "--data_path",
        default="/data",
        type=str,
        help="Path to the data",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Whether to preload the data in memory",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--prefetch_factor",
        default=2,
        type=int,
        help="Number of batches loaded in advance by each worker.",
    )
    parser.add_argument(
        "--experiment_name",
        default="example",
        type=str,
        help="Name to identify the experiment",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use when training the model",
    )
    args = parser.parse_args()
    with open(args.hyperparameters, "r") as f:
        hyperparameters = yaml.safe_load(f)
 
    # Create output directory
    args.out_dir = os.path.join(args.out_dir, args.experiment_name)
    if os.path.isdir(args.out_dir):
        args.experiment_name = (
            args.experiment_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        args.out_dir = os.path.join(
            args.out_dir,
            args.experiment_name,
        )
    os.makedirs(args.out_dir, exist_ok=True)
 
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
 
    # Set seed
    torch.manual_seed(hyperparameters["seed"])
    np.random.seed(hyperparameters["seed"])
 
    # Create model
    model = Model()
 
    # Move model to device
    model.to(device)
 
    # Create data augmentations
    data_transform = T.Compose(
        [   
            T.ToTensor(),
            T.Normalize([0.4178, 0.4176, 0.4176], [0.1461, 0.1461, 0.1460])
        ]
    )
    file_path = "clinical_data.json"
    with open(file_path, "r") as json_file:
         data = json.load(json_file)
    print(len(data))
    image_paths = []
    labels = []
    for i in data:
        image_paths = image_paths +  ["data/main_branch/ideal/images/" + i["filename"]]
        if i["ifr"] <0.89:
            labels = labels + [1]
        else:
            labels = labels + [0]

    # Create datasets
    train_dataset = MNISTDataset(image_paths,labels, data_transform)

 
    test_dataset = MNISTDataset(image_paths,labels, data_transform)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.prefetch_factor,
    )
    """
    for images, labels in train_dataloader:
        print(images)
        image_testing = images[0]
        label_testing = labels[0]
        break
    image_testing.to(device)
    label_testing.to(device)
    model.eval()
    print(torch.argmax(model(image_testing)))
    #print(train_dataset.__getitem__(0))
    """
    """
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for inputs, _ in data_loader:
        mean += inputs.mean((0, 2, 3))
        std += inputs.std((0, 2, 3))

    mean /= len(dataset)
    std /= len(dataset)

    print("Mean:", mean)
    print("Std:", std)
    """
    
    #print(train_dataset.__getitem__(0))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hyperparameters["batch_size"],
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # Create Logger
    """
    wandb.init(
        project="example",
        name=args.experiment_name,
        config=hyperparameters,
        save_code=True,
    )
    """
    mean = 0
    print(model)
    """
    for images, _ in train_dataloader:
        batch_samples = images.size(0)  # Batch size (first dimension of the tensor)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        print(mean)
        print (images.shape)
        print(images.size(0))
    mean = mean / len(train_dataloader.dataset)
    variance = 0
    stddev = 0
    for images, _ in train_dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        variance += ((images - mean.unsqueeze(1))**2).sum([0, 2])
    stddev = torch.sqrt(variance / (len(train_dataloader.dataset) * 512 * 512))  # Assuming image size is 32x32

    print("Mean:", mean)
    print("Standard Deviation:", stddev)
    """
    #Train model
    #Test model
    checkpoint = torch.load('last.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("epochhhhhhhh",checkpoint["epoch"])
    #train(model, train_dataloader, test_dataloader, hyperparameters, args, device)
    test(model, test_dataloader, device, final_eval=True)



    model.eval()
    model.to(device)
    #checkpoint = torch.load('last.pt', map_location=device)
    #print(checkpoint.keys())
    #model.load_state_dict(checkpoint['model'])

    image1 = Image.open("data/main_branch/ideal/images/251_-17.5_33.8_26_DA_24740063.png").convert('RGB')
    tensor_image1 = data_transform(image1)
    tensor_image1 = tensor_image1.unsqueeze(0).to(device)
    print(model(tensor_image1))



# Print the model output
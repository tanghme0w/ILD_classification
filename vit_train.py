# train clip model for CT classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPImageProcessor
import argparse
from data import CTDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def main(args):

    # Create model and move it to the appropriate device
    backbone_model = CLIPModel.from_pretrained(args.model_path, local_files_only=True).to(args.device)
    classifier_head = MLP(768, 512, 4).to(args.device)

    # Freeze the backbone model
    for param in backbone_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(classifier_head.parameters(), lr=args.lr)
    # Add the cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    processor = CLIPImageProcessor.from_pretrained(args.model_path, local_files_only=True)

    # Train dataset
    train_dataset = CTDataset(args.train_dir, args.metadata_file, processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    # Validation dataset
    val_dataset = CTDataset(args.val_dir, args.metadata_file, processor)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    # Initialize the GradScaler for mixed precision
    scaler = torch.amp.GradScaler()

    for epoch in range(args.epochs):
        classifier_head.train()
        epoch_loss = []
        train_loader = tqdm(train_loader, desc=f"Train Epoch {epoch}")

        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Use autocast for mixed precision
            with torch.amp.autocast(args.device):
                feature = backbone_model.get_image_features(images)
                outputs = classifier_head(feature)
                loss = loss_fn(outputs, labels)
                epoch_loss.append(loss.item())
            # Scale the loss and call backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Log the loss to wandb
            if args.wandb:
                wandb.log({"loss": loss})

            # Update tqdm description with the current loss
            train_loader.set_description(f"Train Epoch {epoch}, Loss: {np.mean(epoch_loss):.4f}")

        print(f"Train Epoch {epoch}, Loss: {np.mean(epoch_loss):.4f}")
        if args.wandb:
            wandb.log({"train_loss": np.mean(epoch_loss)})

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Validation
        backbone_model.eval()
        classifier_head.eval()
        val_loader = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
        count, correct, val_loss = 0, 0, []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)

                with torch.amp.autocast(args.device):
                    import ipdb; ipdb.set_trace()
                    feature = backbone_model.get_image_features(images)
                    outputs = classifier_head(feature)
                    # softmax the outputs
                    outputs = F.softmax(outputs, dim=1)
                    # get the predicted class
                    predicted_class = torch.argmax(outputs, dim=1)
                    # get loss and accuracy
                    loss = loss_fn(outputs, labels)
                    val_loss.append(loss.item())
                    count += len(labels)
                    correct += (predicted_class == labels).sum().item()
                if args.wandb:
                    wandb.log({"val_loss": loss.item(), "val_accuracy": correct / count})

        print(f"Validation Epoch {epoch}, Loss: {np.mean(val_loss):.4f}, Accuracy: {correct / count:.4f}")
        if args.wandb:
            wandb.log({"val_loss": np.mean(val_loss), "val_accuracy": correct / count})

        val_loader.set_description(f"Validation Epoch {epoch}, Loss: {np.mean(val_loss):.4f}, Accuracy: {correct / count:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_dir', type=str, default='/tanghaomiao/medai/data/train/ZR')
    parser.add_argument('--val_dir', type=str, default='/tanghaomiao/medai/data/val/ZR')
    parser.add_argument('--metadata_file', type=str, default='/tanghaomiao/medai/label4Tang.csv')
    parser.add_argument('--model_path', type=str, default='/tanghaomiao/medai/clip-vit-large-patch14')
    parser.add_argument('--wandb', action='store_true', default=False)
    args = parser.parse_args()

    if args.wandb:
        parser.add_argument('--exp_name', type=str, default=f'testrun_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        args = parser.parse_args()

        # Initialize wandb
        wandb.init(
            project="ILD_classification",
            name=args.exp_name,
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "train_dir": args.train_dir,
                "val_dir": args.val_dir,
                "model_path": args.model_path,
            },
            entity="allentang",
            tags=["ILD", "classification"],
        )

    main(args)

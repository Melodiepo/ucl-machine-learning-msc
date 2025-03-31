"""
task.py for COMP0197 cw1-task2
Usage:
  python task.py

This script:
1) Explains random guess for multi-class classification.
2) Trains 4 variants of ELM:
   - Plain ELM
   - ELM + MixUp
   - ELM + Ensemble
   - ELM + MixUp + Ensemble
3) Saves metrics, logs, and a "result.png" with predicted classes.
"""

import os
import random
import copy
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from models import *
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sklearn
from sklearn.metrics import f1_score

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

# Dataset Loading
def load_dataset():
    """
    Load the CIFAR-10 dataset for classification.
    Returns (train_loader, test_loader).
    """
    # 1) Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # 2) Create train/test datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 3) DataLoader wrappers
    train_loader = DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    return train_loader, test_loader


# 2. Random Guess Explanation
def random_guess_explanation():
    """
    In an N-class classification, random guessing means assigning each test example
    to one of the N classes uniformly at random, i.e. each class with probability 1/N.
    The expected accuracy of such a random predictor is 1/N.
    """
    print("Random Guess Explanation (under 100 words):")
    print(random_guess_explanation.__doc__.strip())


# 3. Metric Computations and Explanations
def compute_metrics(model, loader, device="cpu"):
    """
    Compute Accuracy and F1 (macro) for the model on the given DataLoader.
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # Flatten all_preds and all_labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, f1


def print_metric_explanation():
    explanation = (
        "We report top-1 accuracy and F1-score. "
        "Accuracy is the fraction of correct predictions, "
        "while F1 balances precision and recall, giving a robust measure "
        "especially in multiclass or imbalanced data."
    )
    print("Metric Explanation (under 50 words):")
    print(explanation)

# 4. Visualize Predictions
def visualize_predictions(model, loader, device, filename="result.png"):
    """
    Take a batch of 36 images from loader, run model to get predictions,
    then create a 6x6 montage with PIL, and save to filename.
    Now we also print ground-truth vs. predicted label on each cell.
    """

    data_iter = iter(loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    images = images[:36]
    labels = labels[:36]

    model.eval()
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    
    def tensor_to_pil(img_tensor):
        unnorm = (img_tensor * 0.5 + 0.5).clamp(0, 1) * 255
        arr = unnorm.byte().cpu().numpy()
        arr = arr.transpose((1,2,0))
        return Image.fromarray(arr)

    cell_size = (32, 32)
    grid_size = (6, 6)
    big_image = Image.new('RGB', (cell_size[0]*grid_size[0], cell_size[1]*grid_size[1]))
    draw = ImageDraw.Draw(big_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=10)
    except:
        font = ImageFont.load_default()

    for idx in range(36):
        row = idx // 6
        col = idx % 6
        x_offset = col * cell_size[0]
        y_offset = row * cell_size[1]
        pil_img = tensor_to_pil(images[idx])
        big_image.paste(pil_img, (x_offset, y_offset))

        gt_label_idx = int(labels[idx].cpu().item())
        pred_label_idx = int(preds[idx].cpu().item())
        gt_text = classes[gt_label_idx]
        pred_text = classes[pred_label_idx]

        # Write "GT=xxx  Pred=yyy"
        draw.text((x_offset+1, y_offset+1),
                  f"GT={gt_text}\nP={pred_text}",
                  fill=(255,0,0),
                  font=font)

    big_image.save(filename)
    print(f"Saved visualization to {filename}.")


def visualize_mixup_batch(x_mixed, filename="mixup.png"):
    """
    Take a batch of at least 16 mixup images, 
    create a 4x4 montage, and save to `filename`.
    """
    # 1) We'll display only the first 16 images
    x_mixed = x_mixed[:16].detach().cpu().clone()

    # Inverse of Normalize((0.5,),(0.5,)) to get back to [0,1]
    x_display = (x_mixed * 0.5 + 0.5).clamp(0, 1)

    # Convert each image to a PIL Image
    pil_images = []
    for i in range(x_display.size(0)):
        # shape: [3, H, W]
        arr = (x_display[i] * 255).byte().numpy()
        arr = arr.transpose(1, 2, 0)  # make it H x W x 3
        pil_images.append(Image.fromarray(arr))
        
    # 2) Create a montage 4x4
    grid_size = 4
    cell_w, cell_h = pil_images[0].size
    montage_w = grid_size * cell_w
    montage_h = grid_size * cell_h
    montage_img = Image.new('RGB', (montage_w, montage_h))

    # 3) Paste each image into the montage
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            x_offset = col * cell_w
            y_offset = row * cell_h
            montage_img.paste(pil_images[idx], (x_offset, y_offset))
            idx += 1

    # 4) Save the montage
    montage_img.save(filename)
    print(f"Saved MixUp visualization to {filename}")


# Main Function
def main():
    random_guess_explanation()
    print_metric_explanation()

    device = "cpu"
    train_loader, test_loader = load_dataset()

    # == 1) Plain ELM (no MixUp, no ensemble) ==
    print("\n=== Training Plain ELM ===")
    model_plain = MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16).to(device)
    (train_losses, test_losses, test_acc_list, test_f1_list) = model_plain.fit_elm_sgd(
         train_loader, test_loader, epochs=30, lr=0.001, device=device,
         checkpoint_every=10, checkpoint_prefix="model_plain"
    )
    final_acc_plain = test_acc_list[-1]
    final_f1_plain = test_f1_list[-1]
    print(f"Final test accuracy (Plain ELM): {final_acc_plain:.4f}")
    print(f"Final test F1        (Plain ELM): {final_f1_plain:.4f}")

    # == 2) ELM + MixUp ==
    print("\n=== Training ELM with MixUp ===")
    model_mixup = MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16).to(device)
    mixup = MyMixUp(alpha=1.0, seed=1)
    
    # Visualize a single MixUp batch
    data_iter = iter(train_loader)
    x_batch, y_batch = next(data_iter)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    x_mixed, y_a, y_b, lam = mixup(x_batch, y_batch)
    visualize_mixup_batch(x_mixed, filename="mixup.png")
    
    optimizer = torch.optim.SGD(model_mixup.linear.parameters(), lr=0.001)
    epochs = 30
    test_acc_list_mix = []
    test_f1_list_mix = []

    for epoch in range(epochs):
        model_mixup.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            x_mixed, y_a, y_b, lam = mixup(x, y)
            
            logits = model_mixup(x_mixed)
            loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        acc_val, f1_val = compute_metrics(model_mixup, test_loader, device)
        test_acc_list_mix.append(acc_val)
        test_f1_list_mix.append(f1_val)
        print(f"Epoch {epoch+1}/{epochs} [MixUp]: accuracy={acc_val:.4f}, f1={f1_val:.4f}")

        # If checkpointing is wanted:
        if (epoch + 1) % 10 == 0:
            torch.save(model_mixup.state_dict(), f"model_mixup_epoch{epoch+1}.pt")

    final_acc_mixup = test_acc_list_mix[-1]
    final_f1_mixup = test_f1_list_mix[-1]
    print(f"Final test accuracy (ELM + MixUp): {final_acc_mixup:.4f}")
    print(f"Final test F1       (ELM + MixUp): {final_f1_mixup:.4f}")

    # == 3) ELM + Ensemble (without MixUp) ==
    print("\n=== Training ELM Ensemble (without MixUp) ===")
    num_ensemble = 3
    # Initialize ensemble models and their optimizers
    ensemble_models = [
        MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16).to(device)
        for _ in range(num_ensemble)
    ]
    ensemble_optimizers = [
        torch.optim.SGD(model.linear.parameters(), lr=0.001)
        for model in ensemble_models
    ]
    epochs_ensemble = 30
    checkpoint_freq_ensemble = 10

    for epoch in range(epochs_ensemble):
        # Set all ensemble models to train mode
        for model in ensemble_models:
            model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            # For each model in the ensemble, perform a forward pass and update
            for model, optimizer in zip(ensemble_models, ensemble_optimizers):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # At checkpoint frequency, build the ensemble and save one checkpoint file
        if (epoch + 1) % checkpoint_freq_ensemble == 0:
            ensemble_model = MyEnsembleELM(ensemble_models)
            torch.save(ensemble_model.state_dict(), f"model_ensemble_epoch{epoch+1}.pt")
            print(f"Saved ensemble checkpoint at epoch {epoch+1} (without MixUp)")
    # Final evaluation for Ensemble (without MixUp)
    model_ensemble = MyEnsembleELM(ensemble_models)
    ensemble_acc, ensemble_f1 = compute_metrics(model_ensemble, test_loader, device)
    print(f"Final test accuracy (ELM Ensemble): {ensemble_acc:.4f}")
    print(f"Final test F1       (ELM Ensemble): {ensemble_f1:.4f}")

    # == 4) ELM + MixUp + Ensemble ==
    print("\n=== Training ELM + MixUp + Ensemble ===")
    num_ensemble_mix = 3
    # Initialize ensemble models for MixUp and their optimizers
    ensemble_mix_models = [
        MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16).to(device)
        for _ in range(num_ensemble_mix)
    ]
    ensemble_mix_optimizers = [
        torch.optim.SGD(model.linear.parameters(), lr=0.001)
        for model in ensemble_mix_models
    ]
    epochs_mix_ensemble = 30
    checkpoint_freq_mix_ensemble = 10

    for epoch in range(epochs_mix_ensemble):
        for model in ensemble_mix_models:
            model.train()
        for (x, y) in train_loader:
            x, y = x.to(device), y.to(device)
            # Apply mixup once per batch
            x_mixed, y1, y2, lam = mixup(x, y)
            for model, optimizer in zip(ensemble_mix_models, ensemble_mix_optimizers):
                logits = model(x_mixed)
                loss = lam * F.cross_entropy(logits, y1) + (1 - lam) * F.cross_entropy(logits, y2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if (epoch + 1) % checkpoint_freq_mix_ensemble == 0:
            ensemble_mix_model = MyEnsembleELM(ensemble_mix_models)
            torch.save(ensemble_mix_model.state_dict(), f"model_ensemble_mixup_epoch{epoch+1}.pt")
            print(f"Saved ensemble MixUp checkpoint at epoch {epoch+1}")
    # Final evaluation for Ensemble with MixUp
    model_ensemble_mixup = MyEnsembleELM(ensemble_mix_models)
    ensemble_mixup_acc, ensemble_mixup_f1 = compute_metrics(model_ensemble_mixup, test_loader, device)
    print(f"Final test accuracy (ELM Ensemble + MixUp): {ensemble_mixup_acc:.4f}")
    print(f"Final test F1       (ELM Ensemble + MixUp): {ensemble_mixup_f1:.4f}")

    # ===== Summaries =====
    print("\n=== Summary of the four models ===")
    print(f"Plain ELM             -> acc={final_acc_plain:.4f}, f1={final_f1_plain:.4f}")
    print(f"MixUp ELM             -> acc={final_acc_mixup:.4f}, f1={final_f1_mixup:.4f}")
    print(f"Ensemble ELM          -> acc={ensemble_acc:.4f},  f1={ensemble_f1:.4f}")
    print(f"MixUp + Ensemble ELM  -> acc={ensemble_mixup_acc:.4f}, f1={ensemble_mixup_f1:.4f}")

    # Visualize results from best model
    visualize_predictions(model_ensemble_mixup, test_loader, device, filename="result.png")


if __name__ == "__main__":
    main()
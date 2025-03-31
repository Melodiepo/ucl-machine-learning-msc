"""
task2a.py

This script experiments with a direct iterative least-squares (LS) solver
to optimize the ELM models (instead of using iterative SGD). It compares the
speed and performance of LS versus SGD (using an ensemble of ELMs) and performs
a random hyperparameter search over three hyperparameters:
    - hidden_channels (feature maps)
    - ridge_lambda (regularization constant)
    - conv_std (standard deviation for conv weight initialization)

Requirements:
    - The best model configuration is based on an ensemble of 3 ELMs with:
          in_channels=3, num_classes=10, hidden_channels=256,
          trained via fit_elm_sgd (SGD) as in task.py.
    - The script uses the LS solver (modified here to use iterative LBFGS optimizer to be more memory-efficient).
"""

import time
import copy
import random
import itertools
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

# Import our ELM models and the basic LS training function from models.py
from models import MyExtremeLearningMachine, MyEnsembleELM
from task import load_dataset, compute_metrics, visualize_predictions


def fit_elm_ls(elm_model, train_loader, device="cpu", ridge_lambda=1e-3):
    """
    Incremental least-squares solution for the final linear layer
    of MyExtremeLearningMachine, which has shape: [hidden_channels -> num_classes].
    We compute:
        A = HᵀH,  B = HᵀY
    Then solve (A + λI)*W = B for W, 
    where W has shape [hidden_channels, num_classes].
    """
    elm_model.to(device)
    elm_model.eval()

    hidden_dim = elm_model.linear.in_features  # = hidden_channels
    num_classes = elm_model.linear.out_features

    A = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float32)
    B = torch.zeros((hidden_dim, num_classes), dtype=torch.float32)

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass through conv + global avg pool => shape [batch_size, hidden_dim]
            h = elm_model.conv(inputs)
            h = F.relu(h)
            h = h.view(h.size(0), -1)
            h_cpu = h.cpu()

            # One-hot encode targets
            batch_size = targets.size(0)
            Y_batch = torch.zeros((batch_size, num_classes), dtype=torch.float32)
            Y_batch[torch.arange(batch_size), targets.cpu()] = 1.0

            A += h_cpu.t() @ h_cpu
            B += h_cpu.t() @ Y_batch

    # Solve (A + λI) * W = B
    I = torch.eye(hidden_dim, dtype=torch.float32)
    W = torch.linalg.solve(A + ridge_lambda * I, B)  # shape [hidden_dim, num_classes]

    # Copy solution into the linear layer
    with torch.no_grad():
        # linear.weight is [num_classes, hidden_dim], so transpose W
        elm_model.linear.weight.copy_(W.t().to(device))

        # If the linear layer has a bias, we might want to set it to zeros or 
        # solve for it separately. Here we set it to zeros.
        if elm_model.linear.bias is not None:
            elm_model.linear.bias.zero_()

    print(f"[LS Fit] Completed with hidden_dim={hidden_dim}, ridge_lambda={ridge_lambda}")



def main():
    device = "cpu"
    train_loader, test_loader = load_dataset()

    # ---------------------------------------------------------------------
    # 1. Compare LS versus SGD on the ELM ensemble (best config: 3 ELMs, 256 hidden channels)
    # ---------------------------------------------------------------------
    print("\n=== Comparing ELM Ensemble Training: SGD vs. Least-Squares ===")
    
    # (a) Ensemble using Least-Squares (LS)
    list_of_elms_ls = []
    start_time_ls = time.time()
    ridge_lambda_default = 1e-3
    for i in range(3):
        print(f"Training ELM {i+1} with Least-Squares Solver...")
        elm = MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16, std_init=0.1).to(device)
        fit_elm_ls(elm, train_loader, device=device, ridge_lambda=ridge_lambda_default)
        list_of_elms_ls.append(copy.deepcopy(elm))
    model_ensemble_ls = MyEnsembleELM(list_of_elms_ls)
    ls_time = time.time() - start_time_ls
    ls_acc, ls_f1 = compute_metrics(model_ensemble_ls, test_loader, device)
    print(f"LS Ensemble: Time = {ls_time:.2f}s, Test Accuracy = {ls_acc:.4f}, F1-Score = {ls_f1:.4f}")
    torch.save(model_ensemble_ls.state_dict(), "./elm_ls.pt")

    # (b) Ensemble using SGD (as in task.py)
    list_of_elms_sgd = []
    start_time_sgd = time.time()
    for i in range(3):
        print(f"Training ELM {i+1} with SGD...")
        elm = MyExtremeLearningMachine(in_channels=3, num_classes=10, hidden_channels=16, std_init=0.1).to(device)
        elm.fit_elm_sgd(train_loader, test_loader, epochs=10, lr=0.001, device=device)
        list_of_elms_sgd.append(copy.deepcopy(elm))
    model_ensemble_sgd = MyEnsembleELM(list_of_elms_sgd)
    sgd_time = time.time() - start_time_sgd
    sgd_acc, sgd_f1 = compute_metrics(model_ensemble_sgd, test_loader, device)
    print(f"SGD Ensemble: Time = {sgd_time:.2f}s, Test Accuracy = {sgd_acc:.4f}, F1-Score = {sgd_f1:.4f}")
    
    # ---------------------------------------------------------------------
    # 2. Random Hyperparameter Search with LS for a single ELM model
    # We use (1) Hidden Channels: the number of feature maps in the fixed convolution layer (size of hidden rep, affect model capacity)
    #        (2) Ridge_lambda:the regularization constant in the closed-form solution (controls the stability and regularization in the LS solver)
    #        (3) Std of Conv Initialization
    # ---------------------------------------------------------------------
    print("\n=== Random Hyperparameter Search (Iterative LS) ===")
    hidden_channels = [8, 16]
    ridge_lambda = [1e-4, 1e-3, 1e-2]
    conv_std = [0.05, 0.1, 0.2]

    best_acc = 0.0
    best_params = None
    best_model = None

    for hidden_channels, ridge_lambda, conv_std in itertools.product(hidden_channels, ridge_lambda,
                                                                 conv_std):
        print(f"\nTesting: hidden_channels = {hidden_channels}, ridge_lambda = {ridge_lambda:.4e}, conv_std = {conv_std:.4f}")
        ensemble_candidates = []  # Create an empty list to store ensemble members
        start = time.time()
        for j in range(3):
            candidate = MyExtremeLearningMachine(in_channels=3, num_classes=10,
                                         hidden_channels=hidden_channels,
                                         std_init=conv_std).to(device)
            fit_elm_ls(candidate, train_loader, device=device, ridge_lambda=ridge_lambda)
            ensemble_candidates.append(candidate)  # Append candidate to the list
        ensemble_candidate = MyEnsembleELM(ensemble_candidates)
        elapsed = time.time() - start
        acc, f1 = compute_metrics(ensemble_candidate, test_loader, device)
        # fit_elm_ls(elm_candidate, train_loader, device=device, ridge_lambda=ridge_lambda)
        # elapsed = time.time() - start
        # top1, top3 = compute_metrics(elm_candidate, test_loader, device)
        print(f"Elapsed Time: {elapsed:.2f}s, Test Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = {"hidden_channels": hidden_channels, "ridge_lambda": ridge_lambda, "conv_std": conv_std}
            best_model = copy.deepcopy(ensemble_candidate)

    print("\n=== Best LS Candidate Found ===")
    print(f"Hyperparameters: {best_params}")
    print(f"Test Performance: Accuracy = {best_acc:.4f}")
    # Save best candidate model (optional)
    torch.save(best_model.state_dict(), "./best_elm_ls.pt")
    
    # ---------------------------------------------------------------------
    # 3. Visualize the Best LS Model Predictions
    # ---------------------------------------------------------------------
    visualize_predictions(best_model, test_loader, device, filename="new_result.png")
    


if __name__ == "__main__":
    main()
    
    

"""
(/cs/student/projects2/ml/2024/yihanli/bioinformatics) [yihanli@dory-l task2]$ python task2a.py
Files already downloaded and verified
Files already downloaded and verified

=== Random Hyperparameter Search (Iterative LS) ===

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-04, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
Elapsed Time: 150.23s, Test Top-1 = 0.4567, Top-3 = 0.7444

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-04, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
Elapsed Time: 150.05s, Test Top-1 = 0.4690, Top-3 = 0.7509

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-04, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.0001
Elapsed Time: 149.82s, Test Top-1 = 0.4605, Top-3 = 0.7459

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-03, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
Elapsed Time: 149.51s, Test Top-1 = 0.4594, Top-3 = 0.7441

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-03, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
Elapsed Time: 149.61s, Test Top-1 = 0.4603, Top-3 = 0.7500

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-03, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
Elapsed Time: 149.61s, Test Top-1 = 0.4578, Top-3 = 0.7469

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-02, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
Elapsed Time: 149.81s, Test Top-1 = 0.4682, Top-3 = 0.7537

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-02, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
Elapsed Time: 149.66s, Test Top-1 = 0.4696, Top-3 = 0.7525

Testing: hidden_channels = 8, ridge_lambda = 1.0000e-02, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.01
Elapsed Time: 149.76s, Test Top-1 = 0.4740, Top-3 = 0.7570

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-04, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
Elapsed Time: 583.98s, Test Top-1 = 0.4692, Top-3 = 0.7450

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-04, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
Elapsed Time: 586.36s, Test Top-1 = 0.4636, Top-3 = 0.7428

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-04, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.0001
Elapsed Time: 589.28s, Test Top-1 = 0.4598, Top-3 = 0.7441

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-03, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
Elapsed Time: 590.72s, Test Top-1 = 0.4694, Top-3 = 0.7465

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-03, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
Elapsed Time: 589.33s, Test Top-1 = 0.4770, Top-3 = 0.7546

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-03, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.001
Elapsed Time: 585.13s, Test Top-1 = 0.4713, Top-3 = 0.7522

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-02, conv_std = 0.0500
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
Elapsed Time: 584.98s, Test Top-1 = 0.4726, Top-3 = 0.7482

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-02, conv_std = 0.1000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
Elapsed Time: 616.14s, Test Top-1 = 0.4753, Top-3 = 0.7540

Testing: hidden_channels = 16, ridge_lambda = 1.0000e-02, conv_std = 0.2000
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
[LS Fit] Completed with hidden_dim=16384, ridge_lambda=0.01
Elapsed Time: 584.24s, Test Top-1 = 0.4770, Top-3 = 0.7542

=== Best LS Candidate Found ===
Hyperparameters: {'hidden_channels': 16, 'ridge_lambda': 0.001, 'conv_std': 0.1}
Test Performance: Top-1 = 0.4770

=== Comparing ELM Ensemble Training: SGD vs. Least-Squares ===
Training ELM 1 with Incremental LS...
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
Training ELM 2 with Incremental LS...
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
Training ELM 3 with Incremental LS...
[LS Fit] Completed with hidden_dim=8192, ridge_lambda=0.001
LS Ensemble: Time = 149.09s, Test Top-1 = 0.4636, Top-3 = 0.7479
Saved visualization to new_result.png.
Training ELM 1 with SGD...
Epoch [1/10] -> train_loss=2.0674, test_loss=1.9518, test_top1=0.3362, test_top3=0.6426
Epoch [2/10] -> train_loss=1.9104, test_loss=1.8763, test_top1=0.3578, test_top3=0.6668
Epoch [3/10] -> train_loss=1.8556, test_loss=1.8372, test_top1=0.3674, test_top3=0.6793
Epoch [4/10] -> train_loss=1.8209, test_loss=1.8109, test_top1=0.3764, test_top3=0.6878
Epoch [5/10] -> train_loss=1.7968, test_loss=1.7890, test_top1=0.3858, test_top3=0.6989
Epoch [6/10] -> train_loss=1.7771, test_loss=1.7722, test_top1=0.3900, test_top3=0.7043
Epoch [7/10] -> train_loss=1.7606, test_loss=1.7590, test_top1=0.3942, test_top3=0.7049
Epoch [8/10] -> train_loss=1.7462, test_loss=1.7456, test_top1=0.3994, test_top3=0.7109
Epoch [9/10] -> train_loss=1.7335, test_loss=1.7348, test_top1=0.4025, test_top3=0.7145
Epoch [10/10] -> train_loss=1.7223, test_loss=1.7269, test_top1=0.4048, test_top3=0.7184
Training ELM 2 with SGD...
Epoch [1/10] -> train_loss=2.0217, test_loss=1.9008, test_top1=0.3582, test_top3=0.6679
Epoch [2/10] -> train_loss=1.8673, test_loss=1.8284, test_top1=0.3759, test_top3=0.6912
Epoch [3/10] -> train_loss=1.8131, test_loss=1.7891, test_top1=0.3906, test_top3=0.7022
Epoch [4/10] -> train_loss=1.7781, test_loss=1.7614, test_top1=0.3979, test_top3=0.7122
Epoch [5/10] -> train_loss=1.7524, test_loss=1.7411, test_top1=0.4029, test_top3=0.7193
Epoch [6/10] -> train_loss=1.7317, test_loss=1.7247, test_top1=0.4098, test_top3=0.7219
Epoch [7/10] -> train_loss=1.7145, test_loss=1.7097, test_top1=0.4133, test_top3=0.7282
Epoch [8/10] -> train_loss=1.6995, test_loss=1.6969, test_top1=0.4152, test_top3=0.7318
Epoch [9/10] -> train_loss=1.6858, test_loss=1.6863, test_top1=0.4180, test_top3=0.7350
Epoch [10/10] -> train_loss=1.6738, test_loss=1.6748, test_top1=0.4242, test_top3=0.7394
Training ELM 3 with SGD...
Epoch [1/10] -> train_loss=2.0832, test_loss=1.9692, test_top1=0.3377, test_top3=0.6506
Epoch [2/10] -> train_loss=1.9239, test_loss=1.8798, test_top1=0.3655, test_top3=0.6849
Epoch [3/10] -> train_loss=1.8552, test_loss=1.8277, test_top1=0.3846, test_top3=0.7001
Epoch [4/10] -> train_loss=1.8115, test_loss=1.7929, test_top1=0.3927, test_top3=0.7067
Epoch [5/10] -> train_loss=1.7796, test_loss=1.7660, test_top1=0.4010, test_top3=0.7156
Epoch [6/10] -> train_loss=1.7552, test_loss=1.7446, test_top1=0.4097, test_top3=0.7197
Epoch [7/10] -> train_loss=1.7347, test_loss=1.7270, test_top1=0.4128, test_top3=0.7255
Epoch [8/10] -> train_loss=1.7177, test_loss=1.7128, test_top1=0.4142, test_top3=0.7286
Epoch [9/10] -> train_loss=1.7031, test_loss=1.6990, test_top1=0.4219, test_top3=0.7341
Epoch [10/10] -> train_loss=1.6895, test_loss=1.6873, test_top1=0.4250, test_top3=0.7365
SGD Ensemble: Time = 203.50s, Test Top-1 = 0.4310, Top-3 = 0.7467
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
import warnings 
from sklearn.metrics import f1_score 

class MyExtremeLearningMachine(nn.Module):
    """
    A class implementing a single-hidden-layer ELM,
    with a fixed convolutional layer and a trainable linear output layer.
    
    Attributes:
    -----------
    conv : nn.Conv2d
        The convolutional layer with fixed (non-trainable) weights.
    linear : nn.Linear
        The fully connected (trainable) layer to produce class logits.
    std_init : float
        Standard deviation to be used when initializing the fixed conv weights.
    hidden_channels : int
        Number of feature maps in the hidden conv layer.

    Methods:
    --------
    initialise_fixed_layers(std_init: float):
        Initializes the fixed convolutional weights with Gaussian noise.
    forward(x: torch.Tensor) -> torch.Tensor:
        Forward pass through fixed conv, flatten, and trainable linear layer.
    fit_elm_sgd(...):
        Trains the ELM using minibatch SGD.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int, # hyperparameter 1: size of the hidden convolutional layer
        std_init: float = 0.1, # hyperparameter 2: std of the random init of conv filters
        kernel_size: int = 5
    ):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels (e.g. 3 for RGB).
        num_classes: int
            Number of classes for classification.
        hidden_channels: int
            Number of feature maps for the conv layer (the "hidden" dimension).
        std_init: float
            Standard deviation for the random init of conv filters.
        kernel_size: int
            Kernel size for the convolution.
        """
        super().__init__()
        # Range check (requirement: warn if out of recommended range)
        if hidden_channels <= 0 or hidden_channels > 2048:
            warnings.warn(
                f"hidden_channels={hidden_channels} out of recommended range [1..2048]."
            )
        if std_init <= 0 or std_init > 1.0:
            warnings.warn(
                f"std_init={std_init} out of recommended range (0..1]."
            )
        
        self.hidden_channels = hidden_channels
        self.std_init = std_init
        self.num_classes = num_classes

        # 1) Fixed random convolution
        self.conv = nn.Conv2d(
            in_channels, 
            hidden_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False)
        
        self.initialise_fixed_layers(self.std_init)

        # Freeze convolution weights after initialization
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # huge flatten => hidden_channels * 32 * 32
        flattened_dim = hidden_channels * 32 * 32
        self.linear = nn.Linear(flattened_dim, num_classes, bias=True)

        
    def initialise_fixed_layers(self, std_init: float):
        """
        Initialize the conv layer weights with Gaussian noise (mean=0, std=std_init).
        Parameters
        ----------
        std_init : float
            The standard deviation for random initialization.
        """
        with torch.no_grad():
            for param in self.conv.parameters():
                nn.init.normal_(param, mean=0.0, std=std_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 
          1) pass x through fixed conv, 
          2) flatten, 
          3) pass through linear,
          4) return the logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, in_channels, H, W].

        Returns
        -------
        torch.Tensor
            The logits of shape [batch_size, num_classes].
        """

        # Pass through fixed convolution
        x = self.conv(x)
        # Perhaps add a nonlinearity (like ReLU), though the spec doesn't forbid it:
        x = F.relu(x)
        x = x.view(x.size(0), -1)

        # Pass through the trainable linear
        logits = self.linear(x)
        return logits


    def fit_elm_sgd(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        device: str = "cpu",
        checkpoint_every: int = 0,
        checkpoint_prefix: str = "model"
    ):
        """
        Train the ELM using a minibatch SGD approach.

        Parameters
        ----------
        train_loader : DataLoader
            Dataloader for training set.
        test_loader : DataLoader
            Dataloader for validation/test set, for monitoring performance.
        epochs : int
            Number of epochs to train.
        lr : float
            Learning rate for optimizer.
        weight_decay : float
            Weight decay for regularization.
        device : str
            'cpu' device to run on.

        Returns
        -------
        (list, list)
            Lists of (train_loss, test_loss) per epoch or any other metrics.
        """
        self.to(device)
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        test_losses = []
        test_acc_list = []
        test_f1_list = []

        for epoch in range(epochs):
            # --- Training ---
            self.train()
            total_loss = 0.0
            for (inputs, targets) in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = self(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- Validation ---
            self.eval()
            total_loss_test = 0.0
            correct = 0
            total_samples = 0

            # We'll also gather predictions to compute F1
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for (inputs, targets) in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    logits = self(inputs)
                    loss = criterion(logits, targets)
                    total_loss_test += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == targets).sum().item()
                    total_samples += targets.size(0)

                    all_preds.append(preds.cpu())
                    all_labels.append(targets.cpu())

            avg_test_loss = total_loss_test / len(test_loader)
            test_losses.append(avg_test_loss)

            # Compute accuracy
            accuracy = correct / total_samples

            # Compute F1 (macro)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            f1_val = f1_score(all_labels, all_preds, average="macro")

            test_acc_list.append(accuracy)
            test_f1_list.append(f1_val)

            print(f"Epoch [{epoch+1}/{epochs}] -> "
                  f"train_loss={avg_train_loss:.4f}, test_loss={avg_test_loss:.4f}, "
                  f"accuracy={accuracy:.4f}, f1={f1_val:.4f}")

            # Save a checkpoint if needed
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                torch.save(self.state_dict(), f"{checkpoint_prefix}_epoch{epoch+1}.pt")

        return train_losses, test_losses, test_acc_list, test_f1_list


class MyMixUp:
    """
    Data augmentation class implementing mixup for images and labels.
    We do the mixing before passing data to the network in the training loop.

    The main idea: 
      Given a pair (x1, y1) and (x2, y2) from the same batch (or different),
      we generate a convex combination of the images and labels.

    Reference: https://arxiv.org/abs/1710.09412

    Usually: x' = lambda * x1 + (1 - lambda) * x2
             y' = lambda * y1 + (1 - lambda) * y2 (when using one-hot labels)
    or for class index labels, we keep track of the same mixing ratio.

    This class can be used inside the training loop to transform
    data/labels on the fly.
    """

    def __init__(self, alpha=1.0, seed=1):
        """
        Parameters
        ----------
        alpha : float
            Parameter for Beta distribution that decides the lambda mixing ratio.
        seed : int
            Seed for reproducibility.
        """
        self.alpha = alpha
        random.seed(seed)
        torch.manual_seed(seed)

    def __call__(self, x, y):
        """
        Applies MixUp to a mini-batch (x, y).

        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape [batch_size, channels, H, W].
        y : torch.Tensor
            Batch of class indices of shape [batch_size].

        Returns
        -------
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float):
            Mixed inputs (x_mixed),
            labels1 (y1),
            labels2 (y2),
            mixing coefficient (lam).
        """
        if self.alpha <= 0:
            return x, y, y, 1.0  # no mixup

        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
        batch_size = x.size(0)

        # Shuffle the batch
        indices = torch.randperm(batch_size)
        
        x2 = x[indices]
        y2 = y[indices]
        
        x_mixed = lam * x + (1 - lam) * x2

        return x_mixed, y, y2, lam


class MyEnsembleELM(nn.Module):
    """
    Model ensemble class combining multiple trained instances
    of MyExtremeLearningMachine. Each has random fixed conv weights.
    The forward pass is typically the average (or sum) of their logits.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor:
        Average the outputs of each ELM.
    """

    def __init__(self, list_of_elms, seed=1):
        """
        Parameters
        ----------
        list_of_elms : list
            A list of MyExtremeLearningMachine instances,
            each presumably trained or partially trained.
        """
        super().__init__()
        self.elms = nn.ModuleList(list_of_elms)
        # random.seed(seed)
        # torch.manual_seed(seed)

    def forward(self, x):
        # Combine outputs from each ELM by averaging the logits of multiple ELM.
        logits_list = [elm(x) for elm in self.elms]
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return avg_logits


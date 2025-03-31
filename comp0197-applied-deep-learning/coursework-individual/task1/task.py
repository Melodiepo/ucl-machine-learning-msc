"""
task.py

COMP0197 Individual Coursework - Task 1
Author: Melodie Li

==============================
This script implements:
1) logistic_fun(...): logistic model function with polynomial expansion
2) MyCrossEntropy: cross-entropy loss class
3) MyRootMeanSquare: RMSE loss class
4) fit_logistic_sgd(...): SGD training for logistic regression
5) Synthetic data generation (training + test)
6) Training for M in {1, 2, 3} with both cross-entropy and RMSE
7) Reporting metrics (loss & accuracy) on train and test

Usage:
  python task.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from itertools import combinations_with_replacement
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# 1) Polynomial Feature Expansion
class PolynomialFeatureTransform:
    """ Computes polynomial feature expansion of order M for input vector X in D-dimension. """
    def __init__(self, M, D):
        """
        Args:
            M: int, polynomial order
            D: int, input dimension
        """
        self.M = M
        self.D = D
        self.feature_indices = self._generate_feature_indices()

    def _generate_feature_indices(self):
        """ Precomputes index combinations for polynomial features. """
        feature_indices = []
        for m in range(self.M + 1):
            feature_indices.extend(combinations_with_replacement(range(self.D), m))
        return feature_indices

    def transform(self, X):
        """
        Transforms input X (N, D) to polynomial feature space (N, p).
        Args:
            X: torch.Tensor, shape (N, Dpy)
        Returns:
            Phi: torch.Tensor, shape (N, p) 
        """
        N = X.shape[0]
        p = len(self.feature_indices) 
        Phi = torch.ones((N, p), dtype=torch.float32, device=X.device) 

        for j, idx_tuple in enumerate(self.feature_indices[1:], start=1):
            Phi[:, j] = torch.prod(X[:, idx_tuple], dim=1)

        return Phi 
    

# 2) Logistic Regression Model
def logistic_fun(w, M, x):
    """
    Implements the logistic model function:
      y = σ( w^T φ^M(x) )
    where φ^M(x) is the polynomial feature expansion of x.
    
    Args:
        w:   torch.Tensor (p, ), weight vector
        M:   int, polynomial order
        x:   torch.Tensor (N, D), input data
        
    Returns:
        y:   torch.Tensor (N, ), positive class probabilities in [0,1]
    """
    # Polynomial transform
    poly_transform = PolynomialFeatureTransform(M, x.shape[1])
    Phi_x = poly_transform.transform(x)  # shape (N, p) polynomrial feature matrix
    
    # Linear combination
    z = Phi_x @ w                         # shape (N,) linear combination
    
    # Logistic (sigmoid) function
    y = torch.sigmoid(z)                  # shape (N,) Map to [0,1]
    return y


#3) nn.Module-based Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    """ Implements logistic regression with polynomial feature expansion. """

    def __init__(self, input_dim):
        """
        Args:
            input_dim: int, number of polynomial features (p)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False) 

    def forward(self, phi_x):
        """
        Args:
            phi_x: torch.Tensor, shape (batch_size, p), transformed polynomial features
        Returns:
            y_pred: torch.Tensor, shape (batch_size,), probability predictions
        """
        z = self.linear(phi_x).squeeze() # Squeeze to 1D
        return torch.sigmoid(z) # output the predicted probability


# 3) Loss classes
class MyCrossEntropy(nn.Module):
    """ Implements cross-entropy loss for binary classification. """

    def forward(self, y_pred, t):
        eps = 1e-6
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        ce = - (t * torch.log(y_pred) + (1 - t) * torch.log(1 - y_pred)) # Formula: -(t * log(y) + (1-t) * log(1-y))
        return ce.mean()

class MyRootMeanSquare(nn.Module):
    """ Implements RMSE loss for a binary label. """

    def forward(self, y_pred, t):
        mse = torch.mean((y_pred - t)**2) # Formula: 1/N * sum((y_pred - t)^2)
        return torch.sqrt(mse)
    
# 4) SGD Training Function via DataLoader
def fit_logistic_sgd(X, T, M, lr=0.005, batch_size=64, num_epochs=500, loss_fn=None, verbose=True):
    """
    Stochastic gradient descent training for logistic regression.
    
    Args:
        X: torch.Tensor, shape (N, D), training inputs
        T: torch.Tensor, shape (N,), binary labels {0,1}
        model: LogisticRegressionModel instance
        loss_fn: MyCrossEntropy() or MyRootMeanSquare()
        M: int, polynomial order
        lr: float, learning rate
        batch_size: int, mini-batch size
        num_epochs: int, number of epochs
        verbose: bool, whether to print progress
    
    Returns:
        model: trained LogisticRegressionModel
    """

    if loss_fn is None:
        loss_fn = MyCrossEntropy()
    # Transform dataset using polynomial features
    poly_transform = PolynomialFeatureTransform(M, X.shape[1])
    Phi_X = poly_transform.transform(X)
    
    # Initialize weight vector w
    w = torch.zeros(Phi_X.shape[1], dtype=torch.float32, requires_grad=True)
        
    # Prepare Dataloader
    dataset = TensorDataset(Phi_X, T)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Wraps phi and targets T into a mini-batch iterator for SGD

    # Optimizer (SGD)
    optimizer = optim.SGD([w], lr=lr, weight_decay=0.01)

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for phi_batch, t_batch in dataloader:
            optimizer.zero_grad() # zero out gradients
            y_pred = torch.sigmoid(phi_batch @ w) # forward pass
            loss = loss_fn(y_pred, t_batch) # compute loss
            loss.backward() # backward pass
            optimizer.step() # update weights
            total_loss += loss.item() # accumulate loss
        
        if verbose and (epoch % 50 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch+1}/{num_epochs}, Loss={total_loss/len(dataloader):.4f}")

    return w.detach()

# 5) Synthetic data generation
def generate_data(N, D, M_true, w_true, seed=1):
    """ Generates synthetic training and test datasets. """
    torch.manual_seed(seed)
    
    X = 10 * torch.rand(N, D) -5 # [-5, 5]
    
    y_probs = logistic_fun(w_true, M_true, X) # generate true prob as ground truth
   
    # Introduce strong noise 
    noise = torch.randn(N) * 1
    y_noisy = y_probs + noise
    T = (y_noisy >= 0.5).float() # i.e. Observed binary target: t=1 if noise-corrupted, t=0 otherwise
    gt = (y_probs >= 0.5).float()

    return X, T, gt


# 6) Main script
def main():
    # Setup ground-truth parameters
    M_true = 2
    D = 5
    
    p_true = sum(math.comb(D + m - 1, m) for m in range(M_true + 1))
    
    w_list = w_list = [(-1)**i * math.sqrt(i + 1) / p_true for i in range(p_true)]
    w_true = torch.tensor(w_list, dtype=torch.float32)

    # Generate train & test datasets
    trainX, trainT, train_GT = generate_data(200, D, M_true, w_true, seed=42)
    testX, testT, test_GT = generate_data(100, D, M_true, w_true, seed=999)

    loss_names = ["CrossEntropy", "RMSE"]
    loss_classes = [MyCrossEntropy(), MyRootMeanSquare()]
    
    # Train models for M ∈ {1, 2, 3}
    for loss_name, loss_class in zip(loss_names, loss_classes):
        print(f"TRAINING WITH {loss_name} LOSS")
        for M in [1, 2, 3]:
            print(f"== Training logistic regression with M={M} ==")
            w_hat = fit_logistic_sgd(trainX, trainT, M, lr=0.005, batch_size=64,
                                     num_epochs=500, loss_fn=loss_class, verbose=True)
        
            # Evaluate on training and test sets
            train_phi = PolynomialFeatureTransform(M, D).transform(trainX)
            test_phi = PolynomialFeatureTransform(M, D).transform(testX)
        
            # Compute predicted probabilities
            train_probs = torch.sigmoid(train_phi @ w_hat).detach().numpy()
            test_probs = torch.sigmoid(test_phi @ w_hat).detach().numpy()
        
            # Convert to binary predictions (threshold = 0.5)
            train_pred = (train_probs >= 0.5).astype(int)
            test_pred = (test_probs >= 0.5).astype(int)

            # Compute metric (Accuracy) on:
            # 1. between predicted label on train and the true train label (without noise)  
            train_acc = (train_pred == train_GT.numpy()).mean()
            
            # 2. between predicted label on test and the true test label (without noise)  
            test_acc = (test_pred == test_GT.numpy()).mean()
            
            print(f"[{loss_name}] M={M}: TrainAcc={train_acc:.4f}, TestAcc={test_acc:.4f}")  
    print("Chosen Metric Justification:")
    print("We use accuracy as the main evaluation metric because the dataset is relatively balanced (~51% class balance) and the decision threshold is fixed at 0.5. Accuracy directly reflects the proportion of correctly classified samples, making it intuitive and suitable for this binary classification task.")
    print("\nComparison of Model Prediction vs. Observed Training Data:")
    print("The accuracy on training predictions evaluates how well the model learned the underlying true pattern from data with noise, compared against ground truth label without noise. Test accuracy indicates the model's ability to generalize this learned pattern to unseen data. Ideally, train and test accuracies should be similar; lower test accuracy signals potential overfitting, while similar or higher test accuracy suggests good generalization.")

if __name__ == "__main__":
    main()
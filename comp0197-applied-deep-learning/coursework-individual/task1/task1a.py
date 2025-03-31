import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from itertools import combinations_with_replacement
from torch.utils.data import DataLoader, TensorDataset

########################################
# 1) Utility functions (data & features)
########################################

def num_poly_features(D, order):
    """
    Returns how many total polynomial features (including constant term)
    are in a D-dim input, for polynomials of EXACT order `order`.
    For example, order=2 => all quadratic terms.
    
    If we want sum_{m=0..order}, adapt accordingly below.
    """
    return math.comb(D + order - 1, order)

def build_poly_features_batch(X, order):
    """
    Build polynomial features of EXACT degree=order for each row in X.
    X: shape (N, D)
    Returns: Phi: shape (N, p)  (no constant term in this EXACT approach)
    
    If we want the constant term included, just add it.
    For clarity, we demonstrate how to do EXACT order. 
    """
    N, D = X.shape
    combos = list(combinations_with_replacement(range(D), order))  # e.g. D=3, order=2 => (0,0), (0,1), (0,2), (1,1), ...
    p = len(combos)
    
    # If order=0 => just 1 term "1.0"
    # If order=1 => combos=[(0,),(1,),(2,)...], etc.
    if order == 0:
        # Just return a single column of 1.0s
        return torch.ones((N, 1), dtype=X.dtype, device=X.device)
    
    # We will create a feature matrix of shape (N, p)
    Phi = torch.ones((N, p), dtype=X.dtype, device=X.device)
    for j, combo in enumerate(combos):
        # combo is something like (0,2) meaning x0*x2
        # We'll multiply the relevant columns
        Phi[:, j] = 1.0
        for c in combo:
            Phi[:, j] *= X[:, c]

    return Phi

def generate_data(N, D, M_true, seed=42):
    """
    Generates synthetic data with a known polynomial order M_true.
    We'll build 'true' polynomial weights at EXACT order=M_true,
    then logistic function to define classification labels.
    """
    torch.manual_seed(seed)
    X = 10.0 * torch.rand(N, D) - 5.0  # random in [-5, 5]
    
    p_true = num_poly_features(D, M_true)
    w_list = w_list = [(-1)**i * math.sqrt(i + 1) / p_true for i in range(p_true)]
    w_true = torch.tensor(w_list, dtype=torch.float32)
    
    # Build polynomial features of EXACT M_true
    Phi = build_poly_features_batch(X, M_true)  # shape (N, p_true)
    
    # Linear logits
    z = Phi @ w_true  # shape (N,)
    # Scale up so we get a decent spread
    z = z - z.mean()  # center
    if z.std() > 1e-8:
        z = z / z.std()  # normalize
    z *= 3.0
    
    # Probability
    y_prob = torch.sigmoid(z)
    
    # Noise
    noise = 1 * torch.randn(N)
    y_noisy = y_prob + noise
    T = (y_noisy >= 0.5).float()
    
    return X, T

class MyCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, t):
        # y_pred in [0,1], t in {0,1}
        eps = 1e-6
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        loss = - (t * torch.log(y_pred) + (1 - t)*torch.log(1 - y_pred))
        return loss.mean()

########################################
# 2) Model: SingleOrderChooser
########################################

class SingleOrderChooser(nn.Module):
    """
    This model attempts to pick exactly 1 polynomial order among {1..M_max},
    but does so via a *mixture* over all possible orders during training.
    
    We have:
      - a trainable logit vector theta of shape (M_max,)
      - a separate weight vector w_m for each order m=1..M_max
    The forward pass computes a "mixture of logistic outputs", weighted by
    softmax(theta).
    
    At inference, we do argmax(softmax(theta)) to pick a single integer M.
    """
    def __init__(self, D, M_max):
        super().__init__()
        self.D = D
        self.M_max = M_max
        
        # Logits for each of the M=1..M_max "choices"
        # We'll produce p(m) = softmax(theta)[m]
        self.theta = nn.Parameter(torch.zeros(M_max))
        
        # For each possible M, define a weight vector w_m
        # We'll assume EXACT polynomial order M (no sum_{k=0..M}),
        # so p_m = num_poly_features(D, M).
        # If we want to include a constant term, we can do so
        # by adding order=0 or just appending a column of 1.0, etc.
        self.w_list = nn.ParameterList()
        for m in range(1, M_max+1):
            p_m = num_poly_features(D, m)
            # Create w_m of shape (p_m,), init small random
            w_m = nn.Parameter(0.01 * torch.randn(p_m))
            self.w_list.append(w_m)

    def forward(self, X):
        """
        X: shape (N, D)
        Returns: shape (N,) = mixture of logistic outputs
        """
        # p(m) = softmax(theta)
        probs = torch.softmax(self.theta, dim=0)  # shape (M_max,)
        
        # We'll accumulate a mixture of predicted probabilities
        # across each possible M.
        y_mix = torch.zeros(X.shape[0], device=X.device)
        
        for m_idx in range(self.M_max):
            m_order = m_idx + 1
            # Build EXACT polynomial features for this order
            Phi_m = build_poly_features_batch(X, m_order)  # shape (N, p_m)
            z_m = Phi_m @ self.w_list[m_idx]  # shape (N,)
            y_m = torch.sigmoid(z_m)
            
            y_mix += probs[m_idx] * y_m
        
        return y_mix  # shape (N,)

    def pick_best_order(self):
        """
        Returns an integer 1..M_max chosen by argmax(softmax(theta)).
        """
        with torch.no_grad():
            probs = torch.softmax(self.theta, dim=0)
            best_idx = torch.argmax(probs).item()
        return best_idx + 1

########################################
# 3) Training function
########################################

def fit_single_order_sgd(X, T, model, loss_fn, lr=0.01, batch_size=32, num_epochs=500, verbose=True):
    """
    Standard SGD to train the SingleOrderChooser model.
    """
    dataset = TensorDataset(X, T)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_batch, t_batch in loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, t_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if verbose and (epoch % 100 == 0 or epoch == num_epochs-1):
            print(f"Epoch {epoch+1}/{num_epochs}, loss={avg_loss:.4f}")
    return model

########################################
# 4) Main
########################################

def main():
    # We'll define M_true = 2, but the model doesn't know that.
    D = 5
    M_true = 2
    M_max = 5  # We let the model consider orders from 1..5

    # Generate synthetic data
    trainX, trainT = generate_data(N=200, D=D, M_true=M_true, seed=42)
    testX, testT = generate_data(N=100, D=D, M_true=M_true, seed=999)
    
    # Build the model
    model = SingleOrderChooser(D, M_max)
    ce_loss = MyCrossEntropy()

    print("== Training SingleOrderChooser (mixture over M=1..5) ==")
    fit_single_order_sgd(
        trainX, trainT,
        model,
        loss_fn=ce_loss,
        lr=0.01, batch_size=32, num_epochs=500, verbose=True
    )

    # Evaluate on train/test with the mixture
    def evaluate_mixture(X, T):
        with torch.no_grad():
            y_pred = model(X)  # shape (N,)
            loss_val = ce_loss(y_pred, T).item()
            pred_class = (y_pred >= 0.5).float()
            acc_val = (pred_class == T).float().mean().item()
        return loss_val, acc_val

    train_loss, train_acc = evaluate_mixture(trainX, trainT)
    test_loss, test_acc = evaluate_mixture(testX, testT)

    print(f"\nMixture final: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    print(f"Mixture final: test_loss={test_loss:.4f},  test_acc={test_acc:.4f}\n")

    # Now pick the single best integer M
    chosen_M = model.pick_best_order()
    print(f"*** Model picks M = {chosen_M} ***")

    # Compute the metrics on the chosen M:
    def evaluate_single_M(X, T, chosen_M):
        w_chosen = model.w_list[chosen_M - 1]  # which weight vector
        Phi_chosen = build_poly_features_batch(X, chosen_M)  # EXACT order
        with torch.no_grad():
            z = Phi_chosen @ w_chosen
            y_pred = torch.sigmoid(z)
            loss_val = ce_loss(y_pred, T).item()
            pred_class = (y_pred >= 0.5).float()
            acc_val = (pred_class == T).float().mean().item()
        return loss_val, acc_val

    train_loss_M, train_acc_M = evaluate_single_M(trainX, trainT, chosen_M)
    test_loss_M, test_acc_M = evaluate_single_M(testX, testT, chosen_M)
    print(f"Single M={chosen_M}: train_loss={train_loss_M:.4f}, train_acc={train_acc_M:.4f}")
    print(f"Single M={chosen_M}: test_loss={test_loss_M:.4f},  test_acc={test_acc_M:.4f}")

if __name__ == "__main__":
    main()

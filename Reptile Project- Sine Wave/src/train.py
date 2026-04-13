import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from data import sine_task
from model import SinWave

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def inner_adapt(
        model: nn.Module,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        lr_inner: float = 0.01,
        inner_steps: int = 10
) -> nn.Module:
    adapted_model = copy.deepcopy(model)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=lr_inner)

    adapted_model.train()
    for _ in range(inner_steps):
        optimizer.zero_grad()
        y_pred = adapted_model(x_support)
        loss = loss_fn(y_pred, y_support)
        loss.backward()
        optimizer.step()
    
    return adapted_model

def reptile_update(
        model: nn.Module,
        adapted_model: nn.Module,
        lr_outer: float = 0.1
) -> None:
    with torch.no_grad():
        for param, adapted_param in zip(model.parameters(), adapted_model.parameters()):
            param.data += lr_outer * (adapted_param.data - param.data)

def evaluate(model: nn.Module, num_tasks: int = 100) -> float:
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for _ in range(num_tasks):
        x_support, y_support, x_query, y_query = sine_task()
        adapted_model = inner_adapt(model, x_support, y_support)
        adapted_model.eval()
        with torch.no_grad():
            y_pred = adapted_model(x_query)
            loss = loss_fn(y_pred, y_query)
            total_loss += loss.item()
    return total_loss / num_tasks

def train() -> None:
    model = SinWave()
    num_iterations = 7000

    mse_history = []
    steps = []
    for iteration in range(num_iterations):
        x_support, y_support, _, _ = sine_task()
        adapted_model = inner_adapt(model, x_support, y_support)
        reptile_update(model, adapted_model)

        if iteration % 100 == 0:
            mse = evaluate(model, num_tasks=200)
            print(f"Iteration {iteration}, MSE: {mse:.4f}")

            mse_history.append(mse)
            steps.append(iteration)

    plot_adaptation(model)
    plot_mse_curve(steps, mse_history)

def plot_adaptation(model):
    model.eval()
    x_support, y_support, x_query, y_query = sine_task()
    x_plot = torch.linspace(-5, 5, 100).unsqueeze(1)

    #before adaptation
    with torch.no_grad():
        y_pred_before = model(x_plot)
    
    #after adaptation
    adapted_model = inner_adapt(model, x_support, y_support)
    adapted_model.eval()
    with torch.no_grad():
        y_pred_after = adapted_model(x_plot)
    
    #TRUE FUNCTION
    A = (y_support.max() - y_support.min()) / 2

    with torch.no_grad():
        y_true = adapted_model(x_plot)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(x_query.numpy(), y_query.numpy(), color='red', label='Query Points' )
    plt.plot(x_plot.numpy(), y_pred_before.numpy(), color='blue', label='Before Adaptation')
    plt.plot(x_plot.numpy(), y_pred_after.numpy(), color='green', label='After Adaptation')
    
    plt.scatter
    plt.legend()
    plt.title("Reptile Adaptation on Sine Wave Regression")
    save_path = os.path.join(PLOTS_DIR, "adaptation_plot.png")
    plt.savefig(save_path)
    print(f"Saved adaptation plot → {save_path}")

    plt.show()

def plot_mse_curve(steps, mse_history):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mse_history, marker='o')
    plt.title("MSE Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.grid()
    save_path = os.path.join(PLOTS_DIR, "mse_curve.png")
    plt.savefig(save_path)
    print(f"Saved MSE curve → {save_path}")
    plt.show()

if __name__ == "__main__":
    train()
            
import math
import torch
import numpy as np


def sine_task(
        k_support: int = 20,
        K_query: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates a sine wave regression task.

    Args:
        k_support: Number of support points.
        K_query: Number of query points.

    Returns:
        A tuple containing the support inputs, support targets, query inputs, and query targets.
    """
    # Sample amplitude and phase
    A = np.random.uniform(0.1, 5.0)
    phi = np.random.uniform(0.0, 2* math.pi)

    # Sample support and query inputs
    x_support = np.random.uniform(-5.0, 5.0, size=(k_support, 1))
    x_query = np.random.uniform(-5.0, 5.0, size=(K_query, 1))

    # Compute targets
    y_support = A * np.sin(x_support + phi)
    y_query = A * np.sin(x_query + phi)

    return (
        torch.tensor(x_support, dtype=torch.float32),
        torch.tensor(y_support, dtype=torch.float32),
        torch.tensor(x_query, dtype=torch.float32),
        torch.tensor(y_query, dtype=torch.float32)
    )
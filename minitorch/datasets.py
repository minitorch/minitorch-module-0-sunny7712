import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable


def make_pts(N: int) -> List[Tuple[float, float]]:
    """
    Generate N random 2D points with coordinates between 0 and 1.
    
    Args:
        N: Number of points to generate
        
    Returns:
        List of tuples containing (x_1, x_2) coordinates
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    A data structure representing a set of 2D points with binary classifications.
    
    Attributes:
        N: Number of points in the dataset
        X: List of (x_1, x_2) coordinates for each point
        y: List of binary classifications (0 or 1) for each point
    """
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Create a dataset with a simple vertical decision boundary at x_1 = 0.5.
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Create a dataset with a diagonal decision boundary at x_1 + x_2 = 0.5.
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Create a dataset with two vertical decision boundaries at x_1 = 0.2 and x_1 = 0.8.
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Create a dataset with an XOR-like pattern of decision boundaries.
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Create a dataset with a circular decision boundary centered at (0.5, 0.5).
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Create a dataset with a spiral pattern decision boundary.
    
    Args:
        N: Number of points to generate
        
    Returns:
        Graph object containing the dataset
    """
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets: dict[str, Callable[[int], Graph]] = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}

from abc import abstractmethod
from enum import Enum
import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import display
from collections import defaultdict

class Point():
    """Represents a possible location of the ant.
    """
    def __init__(self, x:int, y:int):
        self.x=x                # x coordinate
        self.y=y                # y coordinate
        self.p=0                # probability of the ant being here
        self.neighbours = []    # neighbouring points

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)

    def __str__(self):
        return self.__repr__()

def is_neighbour(p1:Point, p2:Point) -> bool:
    """Check if the ant can go to p2 from p1 in the next step."""
    if p1.x == p2.x:
        if p1.y == p2.y - 10:
            return True
        if p1.y == p2.y + 10:
            return True
    if p1.y == p2.y:
        if p1.x == p2.x - 10:
            return True
        if p1.x == p2.x + 10:
            return True
    return False

def termination_1(x:int, y:int) -> bool:
    """Boundary condition in scenario #1"""
    if np.absolute(x) >= 20 or np.absolute(y) >= 20:
        return True
    return False

def termination_2(x:int, y:int) -> bool:
    """Boundary condition in scenario #2"""
    if -y-x+10<=0:
        return True
    return False

def termination_3(x:int, y:int) -> bool:
    """Boundary condition in scenario #3"""
    if ((x-2.5)/30)**2 + ((y-2.5)/40)**2 - 1 > 0:
        return True
    return False

def plot_boundaries(termination:Callable, bound:int=100, n_points:int=100):
    """Plot the food boundaries. bound and n_points determine the size and granularity of the plot."""
    x = np.linspace(-bound, bound, n_points)
    y = np.linspace(-bound, bound, n_points)
    points_x = []
    points_y = []
    for x_i in x:
        for y_i in y:
            if termination(x_i, y_i):
                points_x.append(x_i)
                points_y.append(y_i)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(points_x, points_y, s=20)
    ax.plot(0,0, "+k", markersize=12)
    return ax

def plot_points(termination:Callable, bound:int=100, title="Figure") -> List[Point]: 
    """plot the possible points the ant can visit before reaching food.

    Args:
        termination (Callable): function implementing the boundary condition
        bound (int, optional): semi-width of the area centered on the anthill we want to sweep. Defaults to 100.
        display (bool, optional): set to True to display points. Defaults to False.
        title (str, optional): give a title to the figure. Defaults to "Figure".

    Returns:
        list: list of points the ant can visit before reaching food containing their neighbours and initialised with probability 0.
    """
    # Get all the possible points.
    x = np.arange(-bound, bound, 10)
    y = np.arange(-bound, bound, 10)
    points = []
    for x_i in x:
        for y_i in y:
            if not termination(x_i, y_i):
                points.append(Point(x_i,y_i))

    # Assign the neighbours.
    for p1 in points:
        for p2 in points:
            if is_neighbour(p1,p2):
                p1.neighbours.append(p2)


    for p in points:
        ax = plt.subplot(1, 1, 1)
        ax.scatter(p.x, p.y, color="r", s=50)
    ax.plot(0,0, "+k", markersize=20)
    ax.set_title(title)

    return points


class Actions(Enum):
    """Action space and its corresponding coordinate change."""
    UP    = (0,  10)
    DOWN  = (0, -10)
    RIGHT = (10,  0)
    LEFT  = (-10, 0)

    def numpy(self) -> np.array:
        """Return the action as an np.array for convenience."""
        return np.array(self.value)


def simulation(boundary: Callable, iters, title: str = ''):

    points = defaultdict(lambda: 0)
    p = np.array([0,0], dtype=np.int32)
    points[p.tobytes()] = 1

    res = [0]
    for i in range(1, iters+1):
        new_points = defaultdict(lambda: 0)
        for p in points:
            prob = points[p]
            p = np.frombuffer(p, dtype=np.int32)
            if not boundary(p[0], p[1]):
                for a in Actions:
                    new_p = p + a.numpy()
                    new_points[new_p.tobytes()] += prob*0.25

        points = new_points
        prob_ = 0
        for p in points:
            x, y = np.frombuffer(p, dtype=np.int32)
            if boundary(x, y):
                prob_ += points[p]
        res.append(res[-1] + prob_ * i)


    plt.plot(res)
    plt.ylabel('seconds')
    plt.xlabel('iterations')
    plt.title(title)

    return res
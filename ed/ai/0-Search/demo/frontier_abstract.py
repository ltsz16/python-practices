"""
An abstract class is a class that serves as a blueprint for other classes but cannot be instantiated directly. It defines a common interface (methods and attributes) that subclasses must implement.

Key characteristics:
Cannot be instantiated: You can't create an object from an abstract class directly
Defines required methods: Uses abstract methods that subclasses must override
Enforces structure: Ensures all subclasses implement certain functionality
Provides common code: Can include regular methods with implementation that subclasses inherit
In Python:
You create abstract classes using the abc module (Abstract Base Class):

Any class that inherits from Frontier must implement the remove() method, or it also becomes abstract.

Frontier is abstract — it defines the interface for frontier data structures
StackFrontier and QueueFrontier are concrete implementations that must override remove()
Both inherit common methods like add(), contains_state(), and empty() from Frontier
This design ensures that any frontier type follows the same contract while allowing different removal strategies (stack = LIFO, queue = FIFO)."""

import abc

class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action

class Frontier(metaclass=abc.ABCMeta): # abstract base class for frontier data structures
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    @abc.abstractmethod
    def remove(self):
        """Remove a node from the frontier."""
        pass

class StackFrontier(Frontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

class QueueFrontier(Frontier):
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node
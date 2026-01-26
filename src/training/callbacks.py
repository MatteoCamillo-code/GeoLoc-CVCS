from dataclasses import dataclass

@dataclass
class EarlyStopping:

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """
        Returns True if should stop.
        """
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

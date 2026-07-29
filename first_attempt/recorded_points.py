from dataclasses import dataclass, field


@dataclass
class RecordedPoints:
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    best_y: list = field(default_factory=list)

    def add(self, x, y):
        self.X.append(x)
        self.y.append(y)
        self.best_y.append(min(self.y))
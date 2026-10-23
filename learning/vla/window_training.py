"""Dependency-light sampling and deadline rules for bounded supervised runs."""
from bisect import bisect_right
import math


class CommandWindows:
    """Every command is eligible, including masked terminal chunks."""

    def __init__(self, records):
        self.records = records
        self.ends = []
        for record in records:
            steps = record["steps"]
            if type(steps) is not int or steps < 1:
                raise ValueError("Positive integer episode length required")
            self.ends.append(steps + (self.ends[-1] if self.ends else 0))
        if not self.ends:
            raise ValueError("Empty training split")

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        offset = self.ends[episode - 1] if episode else 0
        return {"index": index, "episode": self.records[episode]["episode"],
                "step": index - offset}

    def probes(self, maximum=64):
        # Stratified timeline probes, not privileged phase labels as actor inputs.
        candidates, offset = [], 0
        for record in self.records:
            for fraction in (.02, .15, .30, .45, .60, .75, .90, .99):
                step = min(int(record["steps"] * fraction), record["steps"] - 1)
                candidates.append(self.locate(offset + step))
            offset += record["steps"]
        if len(candidates) <= maximum:
            return candidates
        return [candidates[int(i * len(candidates) / maximum)] for i in range(maximum)]


def validate_budget(seconds, steps, batch_size, learning_rate, ceiling=14400):
    if ceiling not in (14400, 28800) or not math.isfinite(seconds) or not 0 < seconds <= ceiling:
        raise ValueError("Runtime must be positive and within the explicitly authorized four/eight-hour ceiling")
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive steps and batch size required")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-3:
        raise ValueError("Invalid learning rate")

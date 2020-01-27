from abc import ABC, abstractmethod
import collections, itertools
import numpy as np

class Recorder(ABC):
    """
        record when the counter is passing current bar and trigger is false.

        - rcdr.record(delta_step, ...) will advance the counter and record if
            conditions are satisfied
        - rcdr.report() will return a record.

        record(delta_step, ...)
    """

    def __init__(self, name, record_interval=0, starting_at=0):
        self.trigger = False
        self.interval = record_interval
        self.next_bar = starting_at
        self.counter = 0
        self.name = name
        super().__init__()


    def update(self, delta_step):
        r"""
        update counter
        """
        self.counter += delta_step
        if self.counter > self.next_bar:
            self.next_bar += self.interval
            return True
        return False


    def __repr__(self):
        return self.__class__.__name__


    @abstractmethod
    def record(self, delta_step, output, target, loss, *args, **kwargs):
        r"""
            update the recorder
        """


    @abstractmethod
    def report(self):
        r"""
            return the calculation
        """

    @abstractmethod
    def reset(self):
        r"""
            reset the internals
        """


class StepRecorder(Recorder):
    def __init__(self, name, record_interval=0, starting_at=0):
        super().__init__(name, record_interval, starting_at)
        self.step = []

    def record(self, delta_step, output, target, loss, *args, **kwargs):
        if self.update(delta_step):
            self.step.append(self.counter)

    def report(self):
        return self.step

    def reset(self):
        self.step = []


class AccuracyRecorder(Recorder):

    def __init__(self, name, record_interval=0, starting_at=0):
        super().__init__(name, record_interval, starting_at)
        self.correct = 0
        self.tot = 0

    def record(self, delta_step, output, target, *args, **kwargs):
        if not self.update(delta_step):
            return
        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(target.view_as(pred)).sum().item()
        self.tot += len(target)

    def report(self):
        if self.tot == 0:
            return -1
        return self.correct / self.tot

    def reset(self):
        self.correct = 0
        self.tot = 0


class AccuracyPerClassRecorder(Recorder):
    r"""
        TP/P per class. true-positive over positive
        see BalancedAccuracy paper. (average of accuracy per class)
        https://ong-home.my/papers/brodersen10post-balacc.pdf
    """

    def __init__(self, name, record_interval=0, starting_at=0):
        super().__init__(name, record_interval, starting_at)
        self.cnt, self.cor = None, None

    def record(self, delta_step, output, target, *args, **kwargs):
        r"""
            output is a tensor of size (Bsz, ClsSz)
            target is a tensor of size (Bsz)
            return: accuracy counts per class
        """
        if not self.update(delta_step):
            return

        if self.cnt is None:
            num_class = output.size(1)
            self.cnt = np.zeros(num_class, dtype=np.int64)
            self.cor = np.zeros(num_class, dtype=np.int64)

        np.add.at(self.cnt, target.cpu().numpy(), 1)
        np.add.at(self.cor, target.cpu().numpy(), \
                  (output.max(1)[1] == target).cpu().numpy())

    def report(self):
        return (self.cor / self.cnt).tolist()

    def reset(self):
        if self.cnt is None:
            return
        self.cnt.fill(0)
        self.cor.fill(0)


class LossRecorder(Recorder):

    def __init__(self, name, record_interval=0, starting_at=0):
        super().__init__(name, record_interval, starting_at)
        self.data = []

    def record(self, delta_step, output, target, loss, *args, **kwargs):
        if self.update(delta_step):
            self.data.append(loss.item())

    def report(self):
        return self.data

    def reset(self):
        self.data = []


class AverageLossRecorder(Recorder):

    def __init__(self, name, record_interval=0, starting_at=0):
        super().__init__(name, record_interval, starting_at)
        self.counter, self.accumulator = 0, 0

    def record(self, delta_step, output, target, loss, *args, **kwargs):
        if self.update(delta_step):
            self.counter += 1
            self.accumulator += loss.item()

    def report(self):
        return self.accumulator / self.counter

    def reset(self):
        self.counter, self.accumulator = 0, 0

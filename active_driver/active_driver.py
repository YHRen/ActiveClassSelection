from abc import ABC, abstractmethod
import collections, itertools
import itertools as its
import numpy as np

class ActiveClassDriver(ABC):
    r"""
        ActiveClassDriver is responsible for providing a budget plan for
        selecting how many instances from each class.
    """

    def __init__(self, num_classes: int, budgets_per_stage: int or list):
        r"""
            num_classes: int
            budgets_per_stage: <int> total number of samples per stage
                             : list<int> : a[i] number of samples at stage i
        """

        self.num_classes = num_classes
        self.budgets_per_stage = budgets_per_stage


    @abstractmethod
    def step(self, *args, **kwargs):
        r"""
            gether info of the past stage training.
            and plan ahead for the next stage.
            the implementation depends on the method.
        """


    @abstractmethod
    def get_plan(self, *args, **kwargs):
        r"""
            generate a sampling plan represented as a dictionary:
                {class_id : number_of_samples}.
        """


class UniformRandomDriver(ActiveClassDriver):
    r"""
        UniformRandomDriver is for providing a budget plan for
        selecting the same amount of samples per class.
    """


    def __init__(self, num_classes: int, budgets_per_stage: int or list):
        r"""
            num_classes: int
            budgets_per_stage: <int> total number of samples per stage
                             : list<int> : a[i] number of samples at stage i

            requirements: budgets_per_stage is a multiplier of num_classes
        """
        super().__init__(num_classes, budgets_per_stage)
        if isinstance(budgets_per_stage, int):
            self.budgets_per_class = budgets_per_stage // num_classes
            self.idx = -1
        elif isinstance(budgets_per_stage, list):
            self.budgets_per_class = [x//num_classes for x in
                                      budgets_per_stage]
            self.idx = 0


    def step(self, *args, **kwargs):
        if self.idx >= 0:
            self.idx += 1


    def get_plan(self, *args, **kwargs):
        if self.idx >= 0:
            bgt = self.budgets_per_class[self.idx]
        else:
            bgt = self.budgets_per_class
            # self.idx remains -1

        return dict(zip(range(self.num_classes), its.repeat(bgt)))



class InverseAccuracyDriver(ActiveClassDriver):
    r"""
        InverseAccuracyDriver provides a budget plan
        based on the validation accuracy of a class.
    """


    def __init__(self, num_classes: int, budgets_per_stage: int or list):
        r"""
            num_classes: int
            budgets_per_stage: <int> total number of samples per stage
                             : list<int> : a[i] number of samples at stage i
        """
        super().__init__(num_classes, budgets_per_stage)
        if isinstance(budgets_per_stage, int):
            tmp = budgets_per_stage // num_classes
            self.idx = -1 #fixed budget
        elif isinstance(budgets_per_stage, list):
            tmp = budgets_per_stage[0] // num_classes
            self.idx = 0

        self.current_budgets = [tmp]*num_classes

    def __get_budget_per_stage(self):
        if self.idx >= 0:
            bgt = self.budgets_per_stage[self.idx]
        else:
            bgt = self.budgets_per_stage
        return bgt


    def step(self, valid_acc_per_cls, *args, **kwargs):
        r"""
            valid_acc_per_cls: list<float>
        """
        assert len(valid_acc_per_cls) == self.num_classes
        inv_acc = []
        sum_acc = 0.0
        tot_bgt = self.__get_budget_per_stage()
        for acc in valid_acc_per_cls:
            if acc == 0: ## avoid divide-by-zero
                tmp = self.num_classes
                inv_acc.append(tmp)
                sum_acc += tmp
            else:
                inv_acc.append(1/acc)
                sum_acc += 1/acc

        prob = [ia/sum_acc for ia in inv_acc]
        bgts = [int(p*tot_bgt) for p in prob]

        if tot_bgt - sum(bgts) > 0:
            residue = tot_bgt - sum(bgts)
            selected = np.random.choice(range(self.num_classes), residue, prob)
            for k in selected:
                bgts[k] += 1

        self.current_budgets = bgts

    def get_plan(self, *args, **kwargs):
        return dict(zip(range(self.num_classes),
                        self.current_budgets))

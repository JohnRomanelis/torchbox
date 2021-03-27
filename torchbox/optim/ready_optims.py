from .optim import Optimizer, StatefulOptimizer
from .stats import AverageGrad, AverageSqrGrad, StepCount
from .steppers import adam_step, weight_decay
from ..utils.core import listify

from functools import partial

__all__ = ['adam_opt']

def adam_opt(xtra_step=None, **kwargs):
    return partial(StatefulOptimizer, steppers=[adam_step, weight_decay]+listify(xtra_step), 
                    stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)


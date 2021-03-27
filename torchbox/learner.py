import torch
from torch import tensor
from .callbacks import TrainEvalCallback
from .exceptions import CancelBatchException, CancelEpochException, CancelTrainException
from .utils.core import listify



def param_getter(m): return m.parameters()

class Learner():
    def __init__(self, model, 
                       data, 
                       loss_func, 
                       opt_func, 
                       lr, 
                       splitter=param_getter, 
                       cbs=None, #callbacks 
                       cb_funcs=None):

        self.model, self.data, self.loss_func, self.opt_func, self.lr, self.splitter = model, data, loss_func, opt_func, lr, splitter
        self.in_train, self.logger, self.opt = False, print, None

        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs([cbf() for cbf in listify(cb_funcs)])

    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)
    
    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)
    
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb;                          self('begin_batch')
            self.pred = self.model(self.xb);                    self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb);     self('after_loss')              # NOTE: This way the learner will calculate the loss for the 
            if not self.in_train: return                                                        # the validation set too (may be interesting to monitor in some cases)
            self.loss.backward();                               self('after_backward')          
            self.opt.step();                                    self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                            self('after_cancel_batch')
        finally:                                                self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl): self.one_batch(i, xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def do_begin_fit(self, epochs):
        self.epochs,self.loss = epochs, tensor(0.)
        self('begin_fit')

    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, cbs=None, reset_opt=False):
        self.add_cbs(cbs)
        # Note: to create the opt_func use the functools.partial
        if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                if not self.do_begin_epoch(epoch): self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self('begin_validate'): self.all_batches()
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.remove_cbs(cbs)


    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x:x._order): res = cb(cb_name) or res 
        return res 



class DictLearner(Learner):
    """
        A learner that instead of receiving pairs of xb, yb from the dataloader, 
        receives a dictionary (or a class) containing all the information.
    """

    def one_batch(self, i, example):
        try:
            self.iter = i
            self.example = example;                                  self('begin_batch')
            self.pred = self.model(self.example);                    self('after_pred')
            self.loss = self.loss_func(self.pred, self.example);     self('after_loss')              # NOTE: This way the learner will calculate the loss for the 
            if not self.in_train: return                                                             # the validation set too (may be interesting to monitor in some cases)
            self.loss.backward();                                    self('after_backward')          
            self.opt.step();                                         self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                                 self('after_cancel_batch')
        finally:                                                     self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, example in enumerate(self.dl): self.one_batch(i, example)
        except CancelEpochException: self('after_cancel_epoch')

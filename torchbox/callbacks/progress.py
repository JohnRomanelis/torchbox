import time
from fastprogress.fastprogress import format_time
from fastprogress.fastprogress import master_bar, progress_bar
import numpy as np
from functools import partial
from .callback import Callback
from ..utils.core import listify


class ProgressCallback(Callback):
    _order = -1
    
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        # Changing the logger to the progress bar to handle the messege printing 
        # along with the progress display
        self.run.logger = partial(self.mbar.write, table=True)
        
    def after_fit     (self): self.mbar.on_iter_end()
    def after_batch   (self): 
        self.pb.update(self.iter)
        # how often to update the bar
        if self.iter % 5 == 1:
            # creating a graph
            losses = self.recorder.losses
            iters = np.arange(len(losses))
            graph = [[iters, losses]]
            x_bounds = [self.epoch * len(self.dl), (self.epoch+1) * len(self.dl)]
            y_bounds = [0.0, self.upper_bound]
            self.mbar.update_graph(graph, x_bounds, y_bounds)
        
    def begin_epoch   (self): 
        self.set_pb()
        losses = self.recorder.losses
        self.upper_bound = losses[-1] + losses[-1] / 2 if len(losses) > 0 else 170
         
    def begin_validate(self): self.set_pb()
        
    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)



class AvgStats():
    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train
        
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumulate(self, run):
        bn = run.example['anchors'].shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
        
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else seld.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg.stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)
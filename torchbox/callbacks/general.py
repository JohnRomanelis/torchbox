from .callback import Callback

class TrainEvalCallback(Callback):

    def begin_fit(self):
        self.run.n_epochs=0
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
    
    def begin_epoch(self):
        self.run.n_epochs=self.epochs
        self.run.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False



class SkipValidation(Callback):
    
    def begin_validate(self):
        return True


class PrintLoss(Callback):
    
    def after_epoch(self):
        if isinstance(self.loss, LossTensor):
            pprint(self.loss.sublosses)
        else:
            print(self.loss)
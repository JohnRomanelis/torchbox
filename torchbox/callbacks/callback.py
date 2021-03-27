import re
from ..utils.core import camel2snake

class Callback():
    _order = 0

    def set_runner(self, run): self.run = run
    # This way we get access to the valiable and functions of the self.run
    # So we can run: self.run.in_train or self.in_train
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callbacks$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False



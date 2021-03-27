

class DataBunch():
    """
    DataBunch :
        A class for easy access to the used dataloaders and datasets
        Calls:
             -> .train_dl : dataloader for training dataset
             -> .valid_dl : dataloader for validation dataset
             -> .train_ds : training dataset
             -> .valid_ds : validation dataset
    """
    
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c
    
    # Get the Datasets (not the dataloaders)
    @property
    def train_ds(self): return self.train_dl.dataset
    @property
    def valid_ds(self): return self.valid_dl.dataset

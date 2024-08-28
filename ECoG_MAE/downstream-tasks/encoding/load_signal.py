# Here we want to load the signal for the grid electrodes and organize it in an 8*8 matrix. We also want to perform filtering here - we might need to think about how to do it
# such that it is the same as the filtering we perform for model training - since there we do it using mne, which we can't here, since the preprocessed signal here are mat files.

from loader import ECoGDataset

class EncodingDataset(ECoGDataset):
    def _load_word_data():
        # load dataframe with conversation data.
        pass
    
    
    def _load_grid_data(self):
        # load grid data from files with high gamma
        pass
    
    
    def __iter__(self):
        # Data should be loaded with one example per word given lag.
        # Return word embedding and neural data for model.
        pass


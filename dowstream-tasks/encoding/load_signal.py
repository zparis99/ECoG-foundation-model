# Here we want to load the signal for the grid electrodes and organize it in an 8*8 matrix. We also want to perform filtering here - we might need to think about how to do it
# such that it is the same as the filtering we perform for model training - since there we do it using mne, which we can't here, since the preprocessed signal here are mat files.

import pandas as pd
import numpy as np
from scipy.io import loadmat


def load_signal():

    return

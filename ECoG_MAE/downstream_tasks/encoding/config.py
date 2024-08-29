from dataclasses import dataclass
from typing import Optional

from config import ECoGDataConfig

@dataclass
class EncodingDataConfig(ECoGDataConfig):
    # Path to the dataframe containing conversation data (i.e. embeddings, word onsets and offsets, etc)
    conversation_data_df_path: Optional[str] = None
    
    # The glob path to access electrode data within dataset_path. Should include placeholder for electrode_id {elec_id} See default for sample value.
    # TODO: In the future include options for subject id and conv id but keeping it simple for now with just electrode id.
    electrode_glob_path: str = "NY*_*_Part*_conversation*_electrode_preprocess_file_{elec_id}.mat"

    # Determines where relative to the word onset in ms the neural signal should be read from.
    # Lag = 0 means neural signal will be read starting at word onset.
    # Lag = -2000 means neural signal will be read starting at 2 seconds before word onset.
    # Lag = 100 means neural signal will be read starting at 100 ms before word onset.
    lag: int = 0
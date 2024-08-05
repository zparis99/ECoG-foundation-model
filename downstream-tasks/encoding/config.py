from dataclasses import dataclass, field

from ECoG_MAE.config import ECoGDataConfig

@dataclass
class EncodingDataConfig(ECoGDataConfig):
    # The subject ID the data belongs to.
    sid: int = 798
    # The electrode ID's to grab data from.
    electrodes: list[int] = field(default_factory=lambda: [i + 1 for i in range(64)])
    # The conversation id to train the model over.
    conversation_id: int = 1
    # The number of folds to try training over.
    fold__num: int = 10
    # The word embedding models to use for encoding.
    align_with: list[str] = ["gpt2"]
    # The path to the dataframe which contains word embedding and timing data.
    df_path: str = "word-embeddings/gpt2-layer-8-emb.pkl"
    # If true then filter out non-words (i.e. punctuation)
    exclude_non_words: bool = False
    # If set then all words that appear outside of the most <min_word_freq> common words in the conversation will be filtered out.
    min_word_freq: int = 0
    # The type of embedding to use. Should be a model name in align_with.
    align_with: str = "gpt2"
    # A string representing how to modify data.
    datum_mod: str = ""

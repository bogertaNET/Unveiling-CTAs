import os

UNVEILINGCTAS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/.."
TIMESTAMPED = True

RAW_DATA_PATH = os.path.join(UNVEILINGCTAS_PATH, "data/raw_final.json")
TLD_REGEX_PATH = os.path.join(UNVEILINGCTAS_PATH, "tld_regex")

if TIMESTAMPED == False:
    WORD2ID_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/word2id_nontimestamped.joblib"
    )
    ID2WORD_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/id2word_nontimestamped.joblib"
    )
    MAX_LEN_OTHERS_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/max_length_others_nontimestamped.joblib"
    )
    MAX_LEN_MAIN_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/max_length_main_nontimestamped.joblib"
    )
    DF_OTHERS_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/data_others_nontimestamped.parquet"
    )
    DF_MAIN_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/data_nontimestamped.parquet"
    )
else:
    WORD2ID_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/word2id_timestamped.joblib"
    )
    ID2WORD_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/id2word_timestamped.joblib"
    )
    MAX_LEN_OTHERS_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/max_length_others_timestamped.joblib"
    )
    MAX_LEN_MAIN_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/supportfiles/max_length_main_timestamped.joblib"
    )
    DF_OTHERS_OUT_PATH = os.path.join(
        UNVEILINGCTAS_PATH, "data/data_others_timestamped.parquet"
    )
    DF_MAIN_OUT_PATH = os.path.join(UNVEILINGCTAS_PATH, "data/data_timestamped.parquet")

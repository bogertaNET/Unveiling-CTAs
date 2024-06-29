import pandas as pd
from config import (DF_MAIN_OUT_PATH, DF_OTHERS_OUT_PATH, ID2WORD_OUT_PATH,
                    MAX_LEN_MAIN_OUT_PATH, MAX_LEN_OTHERS_OUT_PATH,
                    WORD2ID_OUT_PATH)
from joblib import dump, load
from tensorflow.keras.preprocessing.sequence import pad_sequences


class UtilizeData:
    def __init__(self, actions):
        self.actions = actions
        self.actors = []
        self.sequences = []
        self.run()

    def uniq_word_create(self):
        UNIQ_WORD = []
        for actor in self.actions.keys():
            for sentence in self.actions[actor]:
                for word in sentence:
                    UNIQ_WORD.append(word)
        UNIQ_WORD = list(set(UNIQ_WORD))

        return UNIQ_WORD

    def word_token_dump(self):
        UNIQ_WORD = self.uniq_word_create()
        self.word2id = {}
        self.id2word = {}

        for word in UNIQ_WORD:
            self.word2id[word] = len(self.word2id)
            self.id2word[len(self.word2id) - 1] = word

        dump(self.word2id, WORD2ID_OUT_PATH)
        dump(self.id2word, ID2WORD_OUT_PATH)

        return self

    def create_df(self):
        for actor in self.actions.keys():
            for i in range(len(self.actions[actor])):
                sentence = []
                for word in self.actions[actor][i]:
                    sentence.append(self.word2id[word])
                self.actions[actor][i] = sentence

        for actor in self.actions.keys():
            for sequence in self.actions[actor]:
                self.sequences.append(sequence)
                self.actors.append(actor)

        self.df = pd.DataFrame(
            {"actor": self.actors, "victim_sequence": self.sequences}
        )
        max_seq = max(len(seq) for seq in self.df["victim_sequence"])
        print(f"Max Sequence {max_seq}")
        print()
        print()

        return self

    def purge_sequences(self, max=256):
        indexes = []
        for idx, row in self.df.iterrows():
            if len(row["victim_sequence"]) > max:
                indexes.append(idx)

        print(f"Before Purge - DF Length: {len(self.df)}")
        print()
        self.df = self.df.drop(index=indexes)
        self.df = self.df.reset_index(drop=True)

        print(f"After Purge - DF Length: {len(self.df)}")
        print()

        return self

    def merge_sf(self):
        sf_sequences = []
        sf_label = []
        tobedropped = []
        for idx, row in self.df.iterrows():
            if "plki" in row["actor"] or "joke" in row["actor"]:
                tobedropped.append(idx)
                sf_sequences.append(row["victim_sequence"])
                sf_label.append("sf")

        self.df = self.df.drop(index=tobedropped)
        self.df = self.df.reset_index(drop=True)

        sfdf = pd.DataFrame({"actor": sf_label, "victim_sequence": sf_sequences})

        sfdf = sfdf.sample(n=1500)
        sfdf = sfdf.reset_index(drop=True)
        self.df = pd.concat([sfdf, self.df])
        self.df = self.df.reset_index(drop=True)
        print("Main - Actor Value Counts:")
        print(self.df["actor"].value_counts())
        print()
        print()
        self.df_others = self.df

        return self

    def create_others(self, threshold=300):
        victimc = dict(self.df["actor"].value_counts())
        other_sequence = []
        other_label = []
        dropidx = []
        for idx, row in self.df.iterrows():
            actor = row["actor"]
            if victimc[actor] < threshold:
                other_sequence.append(row["victim_sequence"])
                other_label.append("other")
                dropidx.append(idx)

        self.df = self.df.drop(index=dropidx)
        self.df = self.df.reset_index(drop=True)

        otherdf = pd.DataFrame(
            {"actor": other_label, "victim_sequence": other_sequence}
        )

        self.df = pd.concat([otherdf, self.df])
        self.df = self.df.reset_index(drop=True)
        print("Others - Actor Value Counts:")
        print(self.df["actor"].value_counts())
        print()
        print()

        victimc = dict(self.df_others["actor"].value_counts())
        dropidx = []
        for idx, row in self.df_others.iterrows():
            actor = row["actor"]
            if victimc[actor] > threshold or victimc[actor] < 15:
                dropidx.append(idx)
        self.df_others = self.df_others.drop(index=dropidx)
        self.df_others = self.df_others.reset_index(drop=True)
        print(self.df_others["actor"].value_counts())

        return self

    def pad_others(self):
        max_length = max(len(seq) for seq in self.df_others["victim_sequence"])
        padded_sequences = list(
            pad_sequences(
                self.df_others["victim_sequence"],
                maxlen=max_length,
                padding="post",
                value=len(self.word2id),
            )
        )

        self.df_others["victim_sequence"] = padded_sequences

        dump(max_length, MAX_LEN_OTHERS_OUT_PATH)

        return self

    def pad(self):
        max_length = max(len(seq) for seq in self.df["victim_sequence"])
        padded_sequences = list(
            pad_sequences(
                self.df["victim_sequence"],
                maxlen=max_length,
                padding="post",
                value=len(self.word2id),
            )
        )

        self.df["victim_sequence"] = padded_sequences

        dump(max_length, MAX_LEN_MAIN_OUT_PATH)

        return self

    def to_parquet(self):
        self.df.to_parquet(DF_MAIN_OUT_PATH, index=False)
        self.df_others.to_parquet(DF_OTHERS_OUT_PATH, index=False)

        return self

    def run(self):
        self.word_token_dump().create_df().purge_sequences().merge_sf().create_others().pad().pad_others().to_parquet()

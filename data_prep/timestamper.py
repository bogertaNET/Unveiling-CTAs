import copy
from datetime import timedelta

import inflect


class TimeStamper:
    def __init__(self, data):
        self.data = data
        self.deepcopied = copy.deepcopy(data)
        self.eng = inflect.engine()

        self.timestamper()

    def get_time_hh_mm_ss(self, sec):
        td_str = str(timedelta(seconds=sec))

        x = td_str.split(":")
        string = (
            "wait: "
            + self.eng.number_to_words(x[0])
            + " hours "
            + self.eng.number_to_words(x[1])
            + " minutes "
            + self.eng.number_to_words(x[2])
            + " seconds"
        )
        return string

    def timer(self, a, b):
        diff = int((a - b))
        x = self.get_time_hh_mm_ss(diff)
        return x

    def timestamper(self):
        for ta in self.data.keys():
            for vic in self.data[ta].keys():
                victim = copy.deepcopy(self.data[ta][vic])
                prev = -1
                for cmd in victim:
                    if prev == -1:
                        prev = int(cmd["when"])
                        continue
                    diff = self.timer(int(cmd["when"]), prev)
                    if "minus" in diff:
                        diff = "wait: zero hours zero minutes zero seconds"
                    data = {"data": diff}
                    ind = self.deepcopied[ta][vic].index(cmd)

                    self.deepcopied[ta][vic].insert(ind, data)
                    prev = int(cmd["when"])

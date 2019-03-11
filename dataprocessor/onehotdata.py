from dataprocessor.data import Data


class OnehotData(Data):
    def __init__(self, df):
        super().__init__(df)

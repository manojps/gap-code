from dataprocessor.data import Data


class OnehotData(Data):
    """ This class prepares data for training. """

    def __init__(self, df):
        """
        Constructs a new OnehotData object

        :param df: A Pandas dataframe
        :type df: pandas.core.frame.DataFrame
        :return: Returns nothing
        """
        super().__init__(df)

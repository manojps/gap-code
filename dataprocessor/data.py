from __future__ import unicode_literals, print_function
import pandas as pd


class Data(object):
    """ This class is the parent data class. """

    def __init__(self, df):
        """
        Constructs a new Data object

        :param df: A Pandas dataframe
        :type df: pandas.core.frame.DataFrame
        :return: Returns nothing
        """
        self.df = df
        # Create labels column
        self.df['labels'] = 'Neither'
        self.df.loc[self.df['A-coref'] == True, 'labels'] = 'A'
        self.df.loc[self.df['B-coref'] == True, 'labels'] = 'B'

    @property
    def sample_count(self):
        """
        Get number of samples in dataset.

        :return: Numer of rows in dataframe
        :rtype: int
        """
        return self.df.shape[0]

    @property
    def feature_names(self):
        """
        Get names of features.

        :return: List of column names in a Pandas dataframe
        :rtype: numpy.ndarray
        """
        return self.df.columns.values

    def column_value_counts(self, target_column, new_column):
        """
        Get value counts of each categorical variable. Store this data in
        a dataframe. Also add a column with relative percentage of each
        categorical variable.

        :param target_column: Name of the column in the original dataframe
            (string)
        :param new_column: Name of the new column where the frequency
            counts are stored
        :type target_column: str
        :type new_column: str
        :return: A Pandas dataframe containing the frequency counts
        :rtype: pandas.core.frame.DataFrame
        """
        df_value_counts = self.df[target_column].value_counts()
        df = pd.DataFrame(df_value_counts)
        df.columns = [new_column]
        df[new_column+'_%'] = 100*df[new_column] \
            / df[new_column].sum()
        return df

    def column_summary(self, target_column, new_column):
        """
        Compute column summary and return as a dataframe.

        :param target_column: Name of the column in the original dataframe
            (string)
        :param new_column: Name of the new column where the frequency counts
            are stored
        :type target_column: str
        :type new_column: str
        :return: A Pandas dataframe containing the summary
        :rtype: pandas.core.frame.DataFrame
        """
        temp = pd.DataFrame(self.df[target_column].describe())
        temp.columns = [new_column]
        temp = temp.round(2)
        return temp

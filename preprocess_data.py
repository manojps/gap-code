import pandas as pd
from dataprocessor.data import Data
from dataprocessor.edadata import EdaData


def get_feature_names(df):
    """ Get feature names in the dataset """
    print("Feature names: {} \n".format(df.feature_names))
    return


def get_sample_count(train, valid, test):
    """ Get sample sizez of training, validation and test data """
    print(" Training set: {} \n Validation set: {} \n Test Set: {}\n".
          format(train.sample_count, valid.sample_count,
                 test.sample_count))
    return


def pronoun_frequency_distribution(train, valid, test):
    """ Get frequency distribution of pronouns in each set """
    df_train = train.column_value_counts('Pronoun', 'Train')
    df_test = test.column_value_counts('Pronoun', 'Test')
    df_valid = valid.column_value_counts('Pronoun', 'Validation')
    # Merge dataframes by index
    pronoun_count = pd.concat([df_train, df_valid, df_test], axis=1)
    # Replace Nan with 0 (zero)
    pronoun_count = pronoun_count.fillna(0)
    # Rounding decimals to two digits after
    pronoun_count = pronoun_count.round(2)
    print(pronoun_count.sort_values(by=['Train'], ascending=False))
    return


def pronoun_count_summary(train, valid, test):
    """ Get summary of pronoun frequency in text. """
    train.store_pronoun_count()
    valid.store_pronoun_count()
    test.store_pronoun_count()
    # Get summary of pronoun count and compare side by side
    df_train = train.column_summary('Pronoun-count', 'Train')
    df_test = test.column_summary('Pronoun-count', 'Test')
    df_valid = valid.column_summary('Pronoun-count', 'Validation')

    pronoun_count_summary = pd.concat([df_train, df_valid, df_test], axis=1)
    print(pronoun_count_summary)
    return


if __name__ == '__main__':

    # Read training, validation and test datasets
    train = Data(pd.read_csv('..\\gap-coreference\\gap-development.tsv',
                             sep='\t'))
    valid = Data(pd.read_csv('..\\gap-coreference\\gap-validation.tsv',
                             sep='\t'))
    test = Data(pd.read_csv('..\\gap-coreference\\gap-test.tsv', sep='\t'))

    print("Train id: {}".format(id(train)))
    get_feature_names(train)
    get_sample_count(train, valid, test)
    pronoun_frequency_distribution(train, valid, test)

    # Get pronoun count in text and store it in a new column
    train2 = EdaData(pd.read_csv('..\\gap-coreference\\gap-development.tsv',
                                 sep='\t'))
    valid2 = EdaData(pd.read_csv('..\\gap-coreference\\gap-validation.tsv',
                                 sep='\t'))
    test2 = EdaData(pd.read_csv('..\\gap-coreference\\gap-test.tsv', sep='\t'))

    train2.person_name_count()
    print(train2.feature_names)
    pronoun_count_summary(train2, valid2, test2)
    # print(train2.df.head())

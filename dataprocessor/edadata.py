import re
from spacy import load
from spacy.symbols import PERSON
from dataprocessor.data import Data


class EdaData(Data):
    """ This class pre-process data for EDA. """

    nlp = load('en_core_web_sm')

    def __init__(self, df):
        """
        Constructs a new EdaData object

        :param df: A Pandas dataframe
        :type df: pandas.core.frame.DataFrame
        :return: Returns nothing
        """
        super().__init__(df)
        self.df['Text-len'] = self.df.Text.str.len()
        self.df['Word-count'] = self.df.Text.str.split().str.len()

    def store_pronoun_count(self):
        """
        Count all pronouns in each row and store it in a new column named
        'Pronoun-count'.

        :return: Returns nothing
        :rtype: None
        """

        def count_all_pronoun(text):
            """
            Count all pronouns in passed text.

            :param text: A string
            :type text: str
            :return: Number of pronouns in the text string
            :rtype: int
            """
            return len(re.findall(r"\b(He|Her|His|She|he|her|hers|him|his|she)\b",
                                  text))

        self.df['Pronoun-count'] = self.df['Text'].apply(count_all_pronoun)
        return

    def text_len(self):
        """
        Count text length in each row and stroe it in a new column named
        "Text-len".

        :return: Returns nothing
        :rtype: None
        """
        self.df['Text-len'] = self.df.Text.str.len()
        return

    def pronoun_first_word(self):
        """
        Find if the target pronoun is the frst word in the sentence, and store
        it in a column in the dataframe.

        :return: Returns nothing
        :rtype: None
        """
        def pronoun_capital(pronoun_text):
            """
            Find if the passed pronoun starts with a capital letter.

            :param pronoun_text: The target pronoun
            :type pronoun_text: str
            :return: Returns true or false
            :rtype: int
            """
            z = re.match('(H\w+)', pronoun_text)
            if z is None:
                return 0
            else:
                return 1

        self.df['Pronoun-first-word'] = self.df['Text'].apply(pronoun_capital)
        return

    def word_count(self):
        """
        Count and store words in "Text" for each row of the dataframe.

        :return: Returns nothing
        :rtype: None
        """
        self.df['Word-count'] = self.df.Text.str.split().str.len()
        return

    def person_name_count(self):
        """
        Identify total number of person names (not unique) used in the text,
        and store it in a new dataframe column named 'Person-name-count'.
        """
        def person_entity_count(text):
            """
            Extract person names and count the total number of person names.
            Not completely accurate, as it is based on heuristics.

            :param text: Text sample in the dataset
            :type text: str
            :return: Rturns the number of person names in the text
            :rtype: int
            """
            doc = EdaData.nlp(text)
            p_count = 0
            # person = []
            for ent in doc.ents:
                if ent.label == PERSON:
                    p_count += 1
                    # person.append(ent.text)
            return p_count

        self.df['Person-name-count'] = self.df['Text'].\
            apply(person_entity_count)
        return

    @staticmethod
    def unique_pronouns(df):
        """
        Get the list of unique pronouns in the dataset.

        :param df: A Pandas dataframe
        :type df: pandas.core.frame.DataFrame
        :return: List of dataframe indexes
        :rtype: list
        """
        return df.index.tolist()

from scripts.helpers import normalize_column_name
import pandas as pd

class ScaleScorer:
    """
    A class to encapsulate the logic for detecting and scoring behavioral scales.
    """

    def __init__(self, df, user_column_mappings):
        """
        Initialize the ScaleScorer with the input DataFrame and column mappings.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing columns for multiple behavioral measures.
        user_column_mappings : dict
            Dictionary mapping scales to their corresponding column mappings.
        """
        self.df = df.copy()
        self.user_column_mappings = user_column_mappings
        self.question_columns_to_drop = []

        # Mapping scale names to their scoring functions
        # TODO - add more as needed
        self.scoring_functions = {
            'MODTAS': self.score_modtas,
            'TIPI': self.score_tipi,
            'VVIQ': self.score_vviq,
            'DPES-Awe': self.score_dpes_awe,
            'MAIA': self.score_maia,
            'Ego-Dissolution': self.score_ego_dissolution,
            'SMES': self.score_smes,
            'Emotional-Breakthrough': self.score_emotional_breakthrough,
            'Psychological-Insight': self.score_psychological_insight,
            'WCS-Connectedness-To-World-Spirituality': self.score_wcs_connectedness_to_world_spirituality,
            'WCS-Connectedness-To-Others': self.score_wcs_connectedness_to_others,
            'WCS-Connectedness-To-Self': self.score_wcs_connectedness_to_self
        }

    def calculate_all_scales(self, mid_processing=False):
        """
        Calculate all available scale scores for the input DataFrame.

        Parameters:
        ----------
        mid_processing : bool, optional
            If True, retains the original question columns. Default is False.

        Returns:
        -------
        pd.DataFrame
            DataFrame with calculated scale scores and question columns removed unless mid_processing is True.
        """
        for scale_name, scoring_fn in self.scoring_functions.items():
            if scale_name not in self.user_column_mappings:
                print(f"[DEBUG] No mapping found for {scale_name}, skipping.")
                continue

            column_mapping = self.user_column_mappings[scale_name]
            normalized_columns = [normalize_column_name(col) for col in column_mapping.values()]
            matching_columns = [col for col in self.df.columns if col in normalized_columns]

            print(f"[DEBUG] Matching columns for {scale_name}: {matching_columns}")

            if matching_columns:
                self.df[scale_name] = scoring_fn(matching_columns)
                self.question_columns_to_drop.extend(matching_columns)
            else:
                print(f"Warning: No matching columns found for {scale_name}.")

        print(f"[INFO] Dropping columns: {self.question_columns_to_drop}")
        if not mid_processing:
            self.df = self.df.drop(columns=self.question_columns_to_drop, errors='ignore')

        return self.df

    def score_modtas(self, columns):
        return self.df[columns].mean(axis=1)

    def score_tipi(self, columns):
        def recode_reverse_score(item_score):
            return 8 - item_score

        self.df[columns[1]] = self.df[columns[1]].apply(recode_reverse_score)
        self.df[columns[3]] = self.df[columns[3]].apply(recode_reverse_score)
        return self.df[columns].mean(axis=1)

    def score_vviq(self, columns):
        return self.df[columns].mean(axis=1)

    def score_dpes_awe(self, columns):
        return self.df[columns].sum(axis=1)

    def score_maia(self, columns):
        return self.df[columns].sum(axis=1)

    def score_ego_dissolution(self, columns):
        return self.df[columns].sum(axis=1)

    def score_smes(self, columns):
        return self.df[columns].sum(axis=1)

    def score_emotional_breakthrough(self, columns):
        return self.df[columns].sum(axis=1)

    def score_psychological_insight(self, columns):
        return self.df[columns].sum(axis=1)

    def score_wcs_connectedness_to_world_spirituality(self, columns):
        return self.df[columns].mean(axis=1)

    def score_wcs_connectedness_to_others(self, columns): # TODO - add warning in app that will tell users that the columns must match exactly as seen below
        """
        Calculate the WCS Connectedness to Others score

        Parameters:
        -----------
        columns : list
            A list with the column names

        Returns:
        --------
        pd.Series
            A series containing the calculated connectedness scores for each row.
        """
        # Extract relevant columns directly from the DataFrame
        col1 = self.df[columns[0]]  # 'I felt trapped in my mind.'
        col2 = self.df[columns[1]]  # 'I felt alone.'
        col3 = self.df[columns[2]]  # 'I felt connected to friends and/or family.'
        col4 = self.df[columns[3]]  # 'I felt connected to a community.'
        col5 = self.df[columns[4]]  # 'I felt unwelcome amongst others.'
        col6 = self.df[columns[5]]  # 'I felt separate from the world around me.'

        # Apply the formula
        connectedness_score = ((10 - col1) + (10 - col2) + col3 + col4 + (10 - col5) + (10 - col6)) / 6

        return connectedness_score

    def score_wcs_connectedness_to_self(self, columns):
        """
        Calculate the WCS Connectedness to Self score.

        Parameters:
        -----------
        columns : list
            A list with the column names.

        Returns:
        --------
        pd.Series
            A series containing the calculated connectedness scores for each row.
        """

        # Extract the relevant columns from the DataFrame
        cols = [self.df[col] for col in columns]

        # Combine columns into a DataFrame and calculate the row-wise mean
        connectedness_score = pd.concat(cols, axis=1).mean(axis=1)

        return connectedness_score

    # TODO - add more scoring functions
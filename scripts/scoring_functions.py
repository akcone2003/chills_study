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
            'WCS-Connectedness-To-Self': self.score_wcs_connectedness_to_self,
            'WCS': self.score_wcs
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
        results = []  # Store individual scale DataFrames for concatenation

        for scale_name, scoring_fn in self.scoring_functions.items():
            if scale_name not in self.user_column_mappings:
                print(f"[DEBUG] No mapping found for {scale_name}, skipping.")
                continue

            # Retrieve the original column mapping from user input
            column_mapping = self.user_column_mappings[scale_name]

            # Normalize both the DataFrame columns and the user-mapped columns
            normalized_mapping = [normalize_column_name(col) for col in column_mapping.values()]
            normalized_df_columns = [normalize_column_name(col) for col in self.df.columns]

            # Find the matching columns (after normalization)
            matching_columns = [col for col in normalized_mapping if col in normalized_df_columns]

            print(f"[DEBUG] Matching columns for {scale_name}: {matching_columns}")

            if matching_columns:
                # Call the appropriate scoring function
                score_result = scoring_fn(matching_columns)

                if isinstance(score_result, pd.DataFrame):
                    # If the scoring function returns a DataFrame, store it for concatenation
                    results.append(score_result)
                else:
                    # If it returns a Series, add it directly to the DataFrame
                    self.df[scale_name] = score_result
                    self.question_columns_to_drop.extend(matching_columns)
            else:
                print(f"Warning: No matching columns found for {scale_name}.")

        # Concatenate any DataFrames returned by the scoring functions
        if results:
            scores_df = pd.concat(results, axis=1)
            self.df = pd.concat([self.df, scores_df], axis=1)

        print(f"[INFO] Dropping columns: {self.question_columns_to_drop}")

        if not mid_processing:
            # Drop question columns from the DataFrame unless mid-processing is active
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
            A series containing the connectedness scores for each row.
        """

        # Combine columns and calculate row-wise mean
        return self.df[columns].mean(axis=1)

    def score_wcs_connectedness_to_others(self, columns):
        """
        Calculate the WCS Connectedness to Others score.

        Parameters:
        -----------
        columns : list
            A list with the column names.

        Returns:
        --------
        pd.Series
            A series containing the connectedness scores for each row.
        """

        # Apply the formula: ((10 - col1) + (10 - col2) + col3 + col4 + (10 - col5) + (10 - col6)) / 6
        score = (
            (10 - self.df[columns[0]]) +
            (10 - self.df[columns[1]]) +
            self.df[columns[2]] +
            self.df[columns[3]] +
            (10 - self.df[columns[4]]) +
            (10 - self.df[columns[5]])
        ) / 6

        return score

    def score_wcs_connectedness_to_world_spirituality(self, columns):
        """
        Calculate the WCS Connectedness to World/Spirituality score.

        Parameters:
        -----------
        columns : list
            A list with the column names.

        Returns:
        --------
        pd.Series
            A series containing the connectedness scores for each row.
        """

        # Calculate row-wise mean for the provided columns
        return self.df[columns].mean(axis=1)

    def score_wcs(self, columns):
        """
        Calculate the WCS scores for Self, Others, and World/Spirituality,
        and return all four scores including the total WCS score.

        Parameters:
        -----------
        columns : list
            A single list containing all column names in the following order:
            - Connectedness to Self (6 columns)
            - Connectedness to Others (6 columns)
            - Connectedness to World/Spirituality (7 columns)

        Returns:
        --------
        pd.DataFrame
            A DataFrame with four columns: 'Self Score', 'Others Score',
            'World/Spirituality Score', and 'Total WCS Score'.
        """
        # Ensure the list has exactly 19 columns
        if len(columns) != 19:
            raise ValueError(f"Expected 19 columns, but got {len(columns)}")

        print(f"\n\n[DEBUG] Columns passed in: {columns}")

        # Split the columns into the three categories
        self_columns = columns[:6]  # First 6 columns
        others_columns = columns[6:12]  # Next 6 columns
        world_columns = columns[12:]  # Last 7 columns

        # Debug: Print the columns assigned to each group
        print("\n\n[DEBUG] Self Columns:", self_columns)
        print("\n\n[DEBUG] Others Columns:", others_columns)
        print("\n\n[DEBUG] World/Spirituality Columns:", world_columns)

        # Calculate Connectedness to Self score
        self_score = self.score_wcs_connectedness_to_self(self_columns)

        # Calculate Connectedness to Others score
        others_score = self.score_wcs_connectedness_to_others(others_columns)

        # Calculate Connectedness to World/Spirituality score
        world_score = self.score_wcs_connectedness_to_world_spirituality(world_columns)

        # Calculate the total WCS score (avg of all three scores)
        total_score = (self_score + others_score + world_score) / 3

        # Combine all scores into a DataFrame
        scores_df = pd.DataFrame({
            'WCS-Connectedness-To-Self': self_score,
            'WCS-Connectedness-To-Others': others_score,
            'WCS-Connectedness-To-World-Spirituality': world_score,
            'Total WCS': total_score
        })

        return scores_df

    # TODO - add more scoring functions
from scripts.helpers import normalize_column_name
import pandas as pd
import numpy as np


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
            'WCS': self.score_wcs,
            'Religiosity': self.score_religiosity,
            'Big-Five': self.score_big_five,
            'KAMF': self.score_kamf
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
            print(f"\n\n[DEBUG] User Passed Columns: {column_mapping}")

            # Normalize both the DataFrame columns and the user-mapped columns
            normalized_mapping = [normalize_column_name(col) for col in column_mapping.values()]
            normalized_df_columns = [normalize_column_name(col) for col in self.df.columns]

            # Find the matching columns (after normalization)
            matching_columns = [col for col in normalized_mapping if col in normalized_df_columns]

            print(f"\n\n[DEBUG] Matching columns for {scale_name}: {matching_columns}")

            if matching_columns:
                # Call the appropriate scoring function
                score_result = scoring_fn(matching_columns)

                if isinstance(score_result, pd.DataFrame):
                    # If the scoring function returns a DataFrame, store it for concatenation
                    results.append(score_result)
                    self.question_columns_to_drop.extend(matching_columns)
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
        """
        Calculate the MODTAS (Modified Tellegen Absorption Scale) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the MODTAS questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated mean MODTAS score for each row.
        """
        return self.df[columns].mean(axis=1)

    def score_tipi(self, columns):
        """
        Calculate the TIPI (Ten-Item Personality Inventory) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the TIPI questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated mean TIPI score for each row with reverse-coded items.
        """
        def recode_reverse_score(item_score):
            """Reverse score for TIPI items."""
            return 8 - item_score

        self.df[columns[1]] = self.df[columns[1]].apply(recode_reverse_score)
        self.df[columns[3]] = self.df[columns[3]].apply(recode_reverse_score)
        return self.df[columns].mean(axis=1)

    def score_vviq(self, columns):
        """
        Calculate the VVIQ (Vividness of Visual Imagery Questionnaire) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the VVIQ questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated mean VVIQ score for each row.
        """
        return self.df[columns].mean(axis=1)

    def score_dpes_awe(self, columns):
        """
        Calculate the DPES Awe (Dispositional Positive Emotion Scale - Awe) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Awe questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Awe scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_maia(self, columns):
        """
       Calculate the MAIA (Multidimensional Assessment of Interoceptive Awareness) score.

       Parameters:
       -----------
       columns : list
           A list with the column names associated with the MAIA questions.

       Returns:
       --------
       pd.Series
           A series containing the calculated sum of MAIA scores for each row.
       """
        return self.df[columns].sum(axis=1)

    def score_ego_dissolution(self, columns):
        """
        Calculate the Ego Dissolution score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the Ego Dissolution questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of Ego Dissolution scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_smes(self, columns):
        """
       Calculate the SMES (Self-Memory System) score.

       Parameters:
       -----------
       columns : list
           A list with the column names associated with the SMES questions.

       Returns:
       --------
       pd.Series
           A series containing the calculated sum of SMES scores for each row.
       """
        return self.df[columns].sum(axis=1)

    def score_emotional_breakthrough(self, columns):
        """
        Calculate the Emotional Breakthrough score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the Emotional Breakthrough questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of Emotional Breakthrough scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_psychological_insight(self, columns):
        """
       Calculate the Psychological Insight score.

       Parameters:
       -----------
       columns : list
           A list with the column names associated with the Psychological Insight questions.

       Returns:
       --------
       pd.Series
           A series containing the calculated sum of Psychological Insight scores for each row.
       """
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

    def score_big_five(self, columns):
        """
        Calculate subcategory and main Big Five scores.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all subcategory scores, the five big five traits.
        """
        # Ensure the columns are correctly passed
        if len(columns) != 60:
            raise ValueError(f"Expected 60 columns, but got {len(columns)}")

        # Unpack the 60 columns directly
        (
            c1, c2, c3, c4, c5,
            c6, c7, c8, c9, c10, c11, c12,
            c13, c14,
            c15, c16, c17, c18,
            c19, c20, c21, c22,
            c23, c24, c25, c26,
            c27, c28, c29,
            c30, c31, c32,
            c33, c34, c35,
            c36, c37, c38, c39,
            c40, c41, c42,
            c43, c44, c45, c46, c47, c48, c49, c50,
            c51, c52, c53, c54,
            c55, c56,
            c57, c58, c59, c60
        ) = columns

        # Calculate subcategory scores
        # Negative Affect, Self-Reproach, and Neuroticism
        negative_affect = (6 - self.df[c1]) + self.df[c11] + (6 - self.df[c16]) + (6 - self.df[c31]) + (6 - self.df[c46])
        self_reproach = self.df[[c6, c21, c26, c36, c41, c51, c56]].sum(axis=1)
        neuroticism = negative_affect + self_reproach

        # Positive Affect, Sociability, Activity, and Extraversion
        positive_affect = self.df[c7] + (6 - self.df[c12]) + self.df[c37] + (6 - self.df[c42])
        sociability = self.df[c2] + self.df[c17] + (6 - self.df[c27]) + (6 - self.df[c57])
        activity = self.df[[c22, c32, c47, c52]].sum(axis=1)
        extraversion = positive_affect + sociability + activity

        # Aesthetic Interest, Intellectual Interest, Unconventionality, and Openness
        aesthetic_interest = self.df[c13] + (6 - self.df[c23]) + self.df[c43]
        intellectual_interest = (6 - self.df[c48]) + self.df[c53] + self.df[c58]
        unconventionality = (6 - self.df[c3]) + (6 - self.df[c8]) + (6 - self.df[c18]) + (6 - self.df[c38])
        openness = aesthetic_interest + intellectual_interest + unconventionality

        # Nonantagonistic Orientation, Prosocial Orientation, and Agreeableness
        nonantagonistic_orientation = (
                (6 - self.df[c9]) + (6 - self.df[c14]) + self.df[c19] +
                (6 - self.df[c24]) + (6 - self.df[c29]) + (6 - self.df[c44]) +
                (6 - self.df[c54]) + (6 - self.df[c59])
        )
        prosocial_orientation = self.df[c4] + self.df[c34] + (6 - self.df[c39]) + (6 - self.df[c49])
        agreeableness = nonantagonistic_orientation + prosocial_orientation

        # Orderliness, Goal-Striving, Dependability, and Conscientiousness
        orderliness = self.df[c5] + self.df[c10] + (6 - self.df[c15]) + (6 - self.df[c30]) + (6 - self.df[c55])
        goal_striving = self.df[c25] + self.df[c35] + self.df[c60]
        dependability = self.df[c20] + self.df[c40] + (6 - self.df[c45]) + self.df[c50]
        conscientiousness = orderliness + goal_striving + dependability

        # Create a DataFrame with all subcategories and main categories
        scores_df = pd.DataFrame({
            'Negative Affect': negative_affect,
            'Self-Reproach': self_reproach,
            'Neuroticism': neuroticism,
            'Positive-Affect': positive_affect,
            'Sociability': sociability,
            'Activity': activity,
            'Extraversion': extraversion,
            'Aesthetic Interest': aesthetic_interest,
            'Intellectual Interest': intellectual_interest,
            'Unconventionality': unconventionality,
            'Openness': openness,
            'Nonantagonistic-Orientation': nonantagonistic_orientation,
            'Prosocial-Orientation': prosocial_orientation,
            'Agreeableness': agreeableness,
            'Orderliness': orderliness,
            'Goal-Striving': goal_striving,
            'Dependability': dependability,
            'Conscientiousness': conscientiousness
        })

        # Return the final DataFrame with all scores
        return scores_df

    def score_religiosity(self, columns):
        """
        Calculate the Religiosity score.

        Parameters:
        -----------
        columns : list
            A list with the column names.

        Returns:
        --------
        pd.Series
            A series containing the religiosity scores for each row.
        """
        # Religiosity scores require 7 columns
        if len(columns) != 7:
            raise ValueError(f"Expected 7 columns, but got {len(columns)}")

        return self.df[columns].sum(axis=1)

    def score_kamf(self, columns):
        """
        Calculate the KAMF score for each question and apply transformations as specified.

        Parameters:
        ----------
        columns : list
            List of column names in the following order:
            - KAMF_1 (When was the last time you felt moved or touched?)
            - KAMF_2 (How often do you feel moved or touched?)
            - KAMF_3_1, KAMF_3_2, KAMF_3_3, KAMF_3_4 (Sub-questions of item 3)
            - KAMF_4 (How easily do you get moved or touched?)

        Returns:
        -------
        pd.DataFrame
            DataFrame with calculated KAMF scores for each row.
        """
        # Ensure the correct number of columns are passed
        if len(columns) != 7:
            raise ValueError(f"Expected 7 columns, but got {len(columns)}")

        # Extract each question into variables
        kamf_1 = self.df[columns[0]]
        kamf_2 = self.df[columns[1]]
        kamf_3_1 = self.df[columns[2]]
        kamf_3_2 = self.df[columns[3]]
        kamf_3_3 = self.df[columns[4]]
        kamf_3_4 = self.df[columns[5]]
        kamf_4 = self.df[columns[6]]

        # Apply transformations for KAMF_1r and KAMF_4r
        kamf_1r = (kamf_1 * 1.75) - 0.75
        kamf_4r = (kamf_4 * 1.166) - 0.166

        # Compute the KAMF total, which is the average amongst the items
        total_items = pd.concat([
            kamf_1, kamf_1r, kamf_2, kamf_3_1, kamf_3_2, kamf_3_3, kamf_3_4, kamf_4, kamf_4r
        ], axis=1)

        kamf_total = total_items.mean(axis=1)

        # Create a DataFrame to store all the calculated values
        scores_df = pd.DataFrame({
            'KAMF_1': kamf_1,
            'KAMF_1r': kamf_1r,
            'KAMF_2': kamf_2,
            'KAMF_3_1': kamf_3_1,
            'KAMF_3_2': kamf_3_2,
            'KAMF_3_3': kamf_3_3,
            'KAMF_3_4': kamf_3_4,
            'KAMF_4': kamf_4,
            'KAMF_4r': kamf_4r,
            'KAMF_Total': kamf_total
        })

        return scores_df

    # TODO - add more scoring functions

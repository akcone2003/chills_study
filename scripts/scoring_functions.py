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
            'WCS': self.score_wcs,
            'Religiosity': self.score_religiosity,
            'Big-Five': self.score_big_five
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
            c1, c2, c3, c4, c5,  # Negative Affect
            c6, c7, c8, c9, c10, c11, c12,  # Self-Reproach
            c13, c14,  # Neuroticism (combination of negative affect and self-reproach)
            c15, c16, c17, c18,  # Positive Affect
            c19, c20, c21, c22,  # Sociability
            c23, c24, c25, c26,  # Activity
            c27, c28, c29,  # Extraversion
            c30, c31, c32,  # Aesthetic Interest
            c33, c34, c35,  # Intellectual Interest
            c36, c37, c38, c39,  # Unconventionality
            c40, c41, c42,  # Openness
            c43, c44, c45, c46, c47, c48, c49, c50,  # Nonantagonistic Orientation
            c51, c52, c53, c54,  # Prosocial Orientation
            c55, c56,  # Agreeableness
            c57, c58, c59, c60  # Conscientiousness
        ) = columns

        # Calculate subcategory scores
        negative_affect = (6 - self.df[c1]) + self.df[c2] + (6 - self.df[c3]) + (6 - self.df[c4]) + (6 - self.df[c5])
        self_reproach = self.df[[c6, c7, c8, c9, c10, c11, c12]].sum(axis=1)
        neuroticism = negative_affect + self_reproach

        positive_affect = self.df[c15] + (6 - self.df[c16]) + self.df[c17] + (6 - self.df[c18])
        sociability = self.df[c19] + self.df[c20] + (6 - self.df[c21]) + (6 - self.df[c22])
        activity = self.df[[c23, c24, c25, c26]].sum(axis=1)
        extraversion = positive_affect + sociability + activity

        aesthetic_interest = self.df[c30] + (6 - self.df[c31]) + self.df[c32]
        intellectual_interest = (6 - self.df[c33]) + self.df[c34] + self.df[c35]
        unconventionality = (6 - self.df[c36]) + (6 - self.df[c37]) + (6 - self.df[c38]) + (6 - self.df[c39])
        openness = aesthetic_interest + intellectual_interest + unconventionality

        nonantagonistic_orientation = (
                (6 - self.df[c43]) + (6 - self.df[c44]) + self.df[c45] +
                (6 - self.df[c46]) + (6 - self.df[c47]) + (6 - self.df[c48]) +
                (6 - self.df[c49]) + (6 - self.df[c50])
        )
        prosocial_orientation = self.df[c51] + self.df[c52] + (6 - self.df[c53]) + (6 - self.df[c54])
        agreeableness = nonantagonistic_orientation + prosocial_orientation

        orderliness = self.df[c5] + self.df[c10] + (6 - self.df[c15])

        conscientiousness = self.df[[c57, c58, c59, c60]].sum(axis=1)

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


    # TODO - add more scoring functions

from scripts.utils import normalize_column_name, get_score_from_mapping, ORDERED_KEYWORD_SET
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
            # Trait Measures
            'TIPI': self.score_tipi,
            'VVIQ': self.score_vviq,
            'MAIA': self.score_maia,
            'Ego-Dissolution': self.score_ego_dissolution,
            'SMES': self.score_smes,
            'Emotional_Breakthrough': self.score_emotional_breakthrough,
            'Psychological_Insight': self.score_psychological_insight,
            'WCS_Connectedness_To_World_Spirituality': self.score_wcs_connectedness_to_world_spirituality,
            'WCS_Connectedness_To_Others': self.score_wcs_connectedness_to_others,
            'WCS_Connectedness_To_Self': self.score_wcs_connectedness_to_self,
            'WCS': self.score_wcs,
            'Religiosity': self.score_religiosity,
            'NEO-FFI-3_Five_Factor_Inventory': self.score_neo_ffi_3,
            'Cloninger_Self_Transcendence_Subscale': self.score_csts,
            'Self-Transcendence_Scale': self.score_sts,
            'Early_Maladaptive_Schema_(EMS)_Young_Schema_Questionnaire_Short_Form_3_(YSQ-S3)': self.score_ems_ysq3S3,
            'Multidimensional_Iowa_Suggestibility_Scale_(MISS)': self.score_miss,
            'Short_Suggestibility_Scale_(SSS)': self.score_sss,
            'Warwick-Edinburgh_Mental_Wellbeing_Scale_(WEMWBS)': self.score_wemwbs,
            'Cognitive_and_Affective_Mindfulness_Scale_Revised_(CAMS-R)': self.score_cams_r,
            # Measuring Experience-Drive Trait Changes
            'DPES-Joy': self.score_dpes_joy,
            'DPES-Love': self.score_dpes_love,
            'DPES-Pride': self.score_dpes_pride,
            'DPES-Awe': self.score_dpes_awe,
            'DPES-Amusement': self.score_dpes_amusement,
            'DPES-Compassion': self.score_dpes_compassion,
            'MODTAS': self.score_modtas,
            'KAMF': self.score_kamf,
            'MAAS': self.score_maas,
            'Five_Facet_Mindfulness_Questionnaire_(FFMQ)': self.score_ffmq,
            'Positive_Negative_Affect_Schedule_(PANAS)': self.score_panas,
            # Outcome Measures
            'Toronto_Mindfulness_Scale': self.score_toronto_mind_scale,
            # Resilience, Flexibility, Burnout
            'Copenhagen_Burnout_Inventory_(CBI)': self.score_cbi,
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
            'WCS_Connectedness_To-Self': self_score,
            'WCS_Connectedness_To_Others': others_score,
            'WCS_Connectedness_To_World_Spirituality': world_score,
            'Total WCS': total_score
        })

        return scores_df

    def score_neo_ffi_3(self, columns):
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
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
            c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26,
            c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
            c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54,
            c55, c56, c57, c58, c59, c60
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
            'Self_Reproach': self_reproach,
            'Neuroticism': neuroticism,
            'Positive_Affect': positive_affect,
            'Sociability': sociability,
            'Activity': activity,
            'Extraversion': extraversion,
            'Aesthetic_Interest': aesthetic_interest,
            'Intellectual_Interest': intellectual_interest,
            'Unconventionality': unconventionality,
            'Openness': openness,
            'Nonantagonistic_Orientation': nonantagonistic_orientation,
            'Prosocial_Orientation': prosocial_orientation,
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

    def score_dpes_joy(self, columns):
        """
        Calculate the DPES Awe (Dispositional Positive Emotion Scale - Joy) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Joy questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Joy scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_dpes_love(self, columns):
        """
        Calculate the DPES Awe (Dispositional Positive Emotion Scale - Love) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Love questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Love scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_dpes_pride(self, columns):
        """
        Calculate the DPES Awe (Dispositional Positive Emotion Scale - Pride) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Pride questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Pride scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_dpes_amusement(self, columns):
        """
        Calculate the DPES Amusement (Dispositional Positive Emotion Scale - Amusement) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Amusement questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Amusement scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_dpes_compassion(self, columns):
        """
        Calculate the DPES Compassion (Dispositional Positive Emotion Scale - Compassion) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the DPES Compassion questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated sum of DPES Compassion scores for each row.
        """
        return self.df[columns].sum(axis=1)

    def score_maas(self, columns):
        """
        Calculate the MAAS (Mindful Attention Awareness Scale) score.

        Parameters:
        -----------
        columns : list
            A list with the column names associated with the MAAS questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated MAAS scores for each row.
        """
        return self.df[columns].mean(axis=1)

    def score_ffmq(self, columns):
        """
        Calculate subcategory and total FFMQ scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the FFMQ questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all subcategory scores and the total FFMQ score.
        """
        # Ensure the correct number of columns (39 questions expected)
        if len(columns) != 39:
            raise ValueError(f"Expected 39 columns, but got {len(columns)}")

        # Unpack all 39 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
            q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32, q33, q34, q35, q36, q37, q38, q39
        ) = columns

        # Define helper for reverse scoring
        def reverse(q):
            return 6 - self.df[q]

        # Calculate subscale scores
        # Observing Score
        observing = self.df[[q1, q6, q11, q15, q20, q26, q31, q36]].sum(axis=1)

        # Describing Score
        describing = (
                self.df[[q2, q7, q27, q32, q37]]  # Regular scores
                + reverse(q12) + reverse(q16) + reverse(q22)  # Reverse-scored
        ).sum(axis=1)

        # Acting With Awareness Score
        acting_with_awareness = (
                reverse(q5) + reverse(q8) + reverse(q13) + reverse(q18) +
                reverse(q23) + reverse(q28) + reverse(q34) + reverse(q38)
        ).sum(axis=1)

        # Non-judging Score
        nonjudging = (
                reverse(q3) + reverse(q10) + reverse(q14) + reverse(q17) +
                reverse(q25) + reverse(q30) + reverse(q35) + reverse(q39)
        ).sum(axis=1)

        # Non-reactivity score
        nonreactivity = self.df[[q4, q9, q19, q21, q24, q29, q33]].sum(axis=1)

        # Calculate the total FFMQ score
        total_score = observing + describing + acting_with_awareness + nonjudging + nonreactivity

        # Create a DataFrame with all subscale and total scores
        scores_df = pd.DataFrame({
            'Observing': observing,
            'Describing': describing,
            'Acting_with_Awareness': acting_with_awareness,
            'Nonjudging': nonjudging,
            'Nonreactivity': nonreactivity,
            'Total_FFMQ_Score': total_score
        })

        return scores_df

    def score_panas(self, columns):
        """
        Calculate subcategory and total PANAS (Positive Negative Affect Schedule) scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the PANAS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all subcategory scores and the total PANAS score.
        """
        # Ensure the correct number of columns (39 questions expected)
        if len(columns) != 20:
            raise ValueError(f"Expected 20 columns, but got {len(columns)}")

        # Unpack all 20 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
        ) = columns

        # Calculating sub scores
        positive = self.df[q1, q3, q5, q10, q12, q14, q16, q17, q19].sum(axis=1)

        negative = self.df[q2, q4, q6, q7, q8, q11, q13, q15, q18, q20].sum(axis=1)

        total_panas = positive + negative

        # Create a DataFrame with all subscale and total scores
        scores_df = pd.DataFrame({
            'Positive_PANAS': positive,
            'Negative_PANAS': negative,
            'PANAS': total_panas
        })

        return scores_df

    def score_csts(self, columns):
        """
        Calculate Cloninger Self-Transcendence Subscale (CSTS) scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the CSTS questions.

        Returns:
        --------
        pd.Series
            Series containing CSTS score.
        """
        if len(columns) != 15:
            raise ValueError(f"Expected 15 columns but got {len(columns)}")

        return self.df[columns].sum(axis=1) / 10

    def score_sts(self, columns):
        """
        Calculate Self-Transcendence Scale scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the STS questions.

        Returns:
        --------
        pd.Series
            Series containing STS score.
        """
        if len(columns) != 15:
            raise ValueError(f"Expected 15 columns but got {len(columns)}")

        return self.df[columns].sum(axis=1)

    def score_ems_ysq3S3(self, columns):
        """
        Calculate the 18 EMS schema scores based on the YSQ-S3.

        Parameters:
        -----------
        columns : list
            A list containing the column names of the 90 YSQ-S3 questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the scores for each of the 18 schemas.
        """
        # Ensure that the correct number of columns is provided
        if len(columns) != 90:
            raise ValueError(f"Expected 90 columns, but got {len(columns)}")

        # Unpack the 90 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
            q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32, q33, q34, q35, q36, q37, q38, q39, q40,
            q41, q42, q43, q44, q45, q46, q47, q48, q49, q50,
            q51, q52, q53, q54, q55, q56, q57, q58, q59, q60,
            q61, q62, q63, q64, q65, q66, q67, q68, q69, q70,
            q71, q72, q73, q74, q75, q76, q77, q78, q79, q80,
            q81, q82, q83, q84, q85, q86, q87, q88, q89, q90
        ) = columns

        # Define the items belonging to each EMS schema
        schema_map = {
            "Abandonment": [q1, q2, q3, q4, q5],
            "Mistrust/Abuse": [q6, q7, q8, q9, q10],
            "Emotional_Deprivation": [q11, q12, q13, q14, q15],
            "Defectiveness/Shame": [q16, q17, q18, q19, q20],
            "Social_Isolation/Alienation": [q21, q22, q23, q24, q25],
            "Dependence/Incompetence": [q26, q27, q28, q29, q30],
            "Vulnerability_to_Harm_or_Illness": [q31, q32, q33, q34, q35],
            "Enmeshment/Undeveloped_Self": [q36, q37, q38, q39, q40],
            "Failure": [q41, q42, q43, q44, q45],
            "Entitlement/Grandiosity": [q46, q47, q48, q49, q50],
            "Insufficient_Self-Control": [q51, q52, q53, q54, q55],
            "Subjugation": [q56, q57, q58, q59, q60],
            "Self-Sacrifice": [q61, q62, q63, q64, q65],
            "Approval-Seeking/Recognition-Seeking": [q66, q67, q68, q69, q70],
            "Negativity/Pessimism": [q71, q72, q73, q74, q75],
            "Emotional_Inhibition": [q76, q77, q78, q79, q80],
            "Unrelenting_Standards/Hypercriticalness": [q81, q82, q83, q84, q85],
            "Punitiveness": [q86, q87, q88, q89, q90]
        }

        # Calculate the mean score for each schema
        schema_scores = {
            schema: self.df[questions].mean(axis=1) for schema, questions in schema_map.items()
        }

        # Convert the scores into a DataFrame
        scores_df = pd.DataFrame(schema_scores)

        return scores_df

    def score_sss(self, columns):
        """
        Calculate the Short Suggestibility Scale (SSS) score.

        Parameters:
        -----------
        columns : list
            List of column names corresponding to the 95 MISS questions.

        Returns:
        --------
        pd.Series
            Series containing the SSS scores.
        """
        if len(columns) != 21:
            raise ValueError(f"Expected 21 columns but got {len(columns)}")

        return self.df[columns].sum(axis=1)

    def score_miss(self, columns):
        """
        Calculate the MISS scores for each subscale and the total suggestibility score.

        Parameters:
        -----------
        columns : list
            List of column names corresponding to the 95 MISS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing scores for each subscale and the total score.
        """
        # Ensure that the correct number of columns is provided
        if len(columns) != 95:
            raise ValueError(f"Expected 95 columns, but got {len(columns)}")

        # Unpack the 90 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
            q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32, q33, q34, q35, q36, q37, q38, q39, q40,
            q41, q42, q43, q44, q45, q46, q47, q48, q49, q50,
            q51, q52, q53, q54, q55, q56, q57, q58, q59, q60,
            q61, q62, q63, q64, q65, q66, q67, q68, q69, q70,
            q71, q72, q73, q74, q75, q76, q77, q78, q79, q80,
            q81, q82, q83, q84, q85, q86, q87, q88, q89, q90,
            q91, q92, q93, q94, q95
        ) = columns

        # Define subscale calculations
        # Consumer Suggestibility
        consumer = self.df[[q2, q10, q14, q20, q24, q32, q45, q51, q57, q63, q70]].sum(axis=1)
        # Persuadability
        persuadability = self.df[[q1, q5, q13, q22, q35, q44, q47, q62, q69, q75, q76, q82, q88, q89]].sum(axis=1)
        # Physiological Suggestibility
        physiological = self.df[[q11, q15, q25, q33, q52, q58, q64, q66, q68, q71, q77, q94]].sum(axis=1)

        # Physiological Reactivity
        physiological_reactivity = self.df[[q3, q12, q17, q21, q27, q31, q40, q43, q50, q60, q73, q85, q91]].sum(axis=1)

        # Peer Conformity
        peer_conformity = (
                self.df[[q4, q16, q29, q46, q53, q59, q65, q72, q78, q84, q90, q95]].sum(axis=1)
                - self.df[[q34, q39]].sum(axis=1) + 12
        )

        # Mental Control
        mental_control = self.df[[q6, q8, q18, q23, q28, q36, q41, q48, q55, q67, q74, q79, q80, q83, q92]].sum(axis=1)

        # Unpersuabability
        unpersuadability = self.df[[q7, q9, q19, q26, q30, q37, q38, q42, q49, q54, q56, q61, q81, q86, q87, q93]].sum(axis=1)

        # Short Suggestibility Scale (SSS)
        # Check if SSS is already in the DataFrame
        if 'Short_Suggestibility_Scale_(SSS)' in self.df.columns:
            print("SSS already exists in the DataFrame. Skipping SSS calculation.")
            sss = self.df['Short_Suggestibility_Scale_(SSS)']
        else:
            # Calculate the SSS if not present
            sss_columns = [q1, q14, q15, q27, q45, q51, q57, q58, q63, q66, q69, q73,
                           q75, q76, q77, q78, q84, q85, q90, q94, q95]
            sss = self.score_sss(sss_columns)

        # Total Suggestibility Score
        total_suggestibility = consumer + physiological + physiological_reactivity + persuadability + peer_conformity

        # Create a DataFrame with all scores
        scores_df = pd.DataFrame({
            'Consumer_Suggestibility': consumer,
            'Persuadability': persuadability,
            'Physiological_Suggestibility': physiological,
            'Physiological_Reactivity': physiological_reactivity,
            'Peer_Conformity': peer_conformity,
            'Mental_Control': mental_control,
            'Unpersuadability': unpersuadability,
            'Short_Suggestibility_Scale_(SSS)': sss,
            'Total_Suggestibility': total_suggestibility
        })

        return scores_df

    def score_wemwbs(self, columns):
        """
        Calculate the WEMWBS score.

        Parameters:
        -----------
        columns : list
            List of column names corresponding to the 14 WEMWBS questions.

        Returns:
        --------
        pd.Series
            Series containing the WEMWBS scores.
        """
        # Ensure correct number of columns
        if len(columns) != 14:
            raise ValueError(f"Expected 14 columns but got {len(columns)}")

        # Sum the responses to get the total WEMWBS score
        return self.df[columns].sum(axis=1)

    def score_cams_r(self, columns):
        """
        Calculate the CAMS-R score.

        Parameters:
        -----------
        columns : list
            List of column names corresponding to the 12 CAMS-R questions.

        Returns:
        --------
        pd.Series
            Series containing the CAMS-R scores.
        """
        # Ensure correct number of columns
        if len(columns) != 12:
            raise ValueError(f"Expected 12 columns but got {len(columns)}")

        # Unpack the 12 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12
        ) = columns

        # Reverse-scoring for items 2, 6, and 7: new_score = 6 - original_score
        reverse_scored = self.df[[q2, q6, q7]].apply(lambda x: 5 - x)

        # Replace original scores for these items with reversed scores
        df_corrected = self.df.copy()
        df_corrected[[q2, q6, q7]] = reverse_scored

        # Calculate total CAMS-R score by summing all 12 items
        cams_r_score = df_corrected[columns].sum(axis=1)

        return cams_r_score

    def score_toronto_mind_scale(self, columns):
        """
        Calculate the MISS scores for each subscale and the total suggestibility score.

        Parameters:
        -----------
        columns : list
            List of column names corresponding to the 95 MISS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing scores for each subscale and the total score.
        """
        if len(columns) != 13:
            raise ValueError(f"Expected 13 columns but got {len(columns)}")

        # Unpack the 12 questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13
        ) = columns

        # Scoring subscales
        curiosity = self.df[[q3, q5, q6, q10, q12, q13]].sum(axis=1)
        decentering = self.df[[q1, q2, q4, q7, q8, q9, q11]].sum(axis=1)
        # Calculate total
        total = curiosity + decentering

        scores_df = pd.DataFrame({
            'Curiosity': curiosity,
            'De-Centering': decentering,
            'Toronto_Mindfulness_Scale_Total': total
        })

        return scores_df

    def score_cbi(self, columns):
        """
        Score the Copenhagen Burnout Inventory (CBI) with inferred mappings.

        Parameters:
        ----------
        columns : list
            List of column names for the CBI questions.

        Returns:
        -------
        pd.DataFrame
            DataFrame with averaged scores for each CBI subscale and total.
        """

        # Infer mappings based on column content
        def infer_scale_mapping(column):
            sample_values = self.df[column].dropna().unique()
            if all(val.lower() in ORDERED_KEYWORD_SET['frequency_08'] for val in sample_values):
                return 'frequency_08'
            elif all(val.lower() in ORDERED_KEYWORD_SET['frequency_09'] for val in sample_values):
                return 'frequency_09'
            else:
                raise ValueError(f"Unexpected values in column '{column}'")
        # Unpack columns
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19
        ) = columns

        # Define subscale columns
        subscales = {
            'Personal_Burnout': self.df[[q1, q2, q3, q4, q5, q6]],
            'Work_Related_Burnout': self.df[[q7, q8, q9, q10, q13, q14, q15]],
            'Client_Related_Burnout': self.df[[q11, q12, q16, q17, q18, q19]]
        }

        # Calculate scores
        scored_subscales = {}
        for subscale, cols in subscales.items():
            # Use the inferred scale for the first column in each subscale
            scale_key = infer_scale_mapping(cols[0])
            scores = self.df[cols].applymap(lambda x: get_score_from_mapping(x, scale_key))
            scored_subscales[subscale] = scores.mean(axis=1)

        # Calculate total average
        scored_subscales['CBI_Total'] = pd.DataFrame(scored_subscales).mean(axis=1)

        return pd.DataFrame(scored_subscales)


# TODO - add multi dimensional health locus, POMS, NEOPI
# TODO - add burnout study behavioral surveys
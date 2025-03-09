from scripts.utils import normalize_column_name, get_score_from_mapping, ORDERED_KEYWORD_SET
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
            # Trait Measures
            'TIPI': self.score_tipi,
            'VVIQ': self.score_vviq,
            'MAIA': self.score_maia,
            'MAIA-S': self.score_maia_s,
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
            'NEO-PI-3_(Openness_to_Experience)': self.score_neo_pi_3,
            'Montgomery-Åsberg_Depression_Rating_Scale_(MADRS)': self.score_madrs,
            'Hamilton_Anxiety_Rating_Scale_(HAM-A)': self.score_ham_a,
            'State-Trait_Anxiety_Inventory_(STAI-State_Form)': self.score_stai,
            'Dispositional_Hope_Scale_(DHS)': self.score_dhs,
            'General_Self-Efficacy_Scale_(GSES)': self.score_gses,
            'Connor-Davidson_Resilience_Scale_(CD-RISC-10)': self.score_cd_risc_10,
            'Multidimensional_Health_Locus_of_Control_(MHLC)': self.score_mhlc,
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
            'PANAS_X': self.score_panas_x,
            '5-Dimensional_Altered States_of_Consciousness_Questionnaire_(5DASC)': self.score_5dasc,
            'Anxiety_Sensitivity_Index-3_(ASI-3_ASI-R)': self.score_asi3,
            'Subjective_Vitality_Scale(SVS)': self.score_svs,
            'Flow_State_Scale_(FFS)_(short-version)': self.score_fss_short,
            'Purpose_In_Life_Test_(PIL)': self.score_pil,
            'Perceived_Stress_Scale_(PSS)': self.score_pss,
            'Profile_of_Mood_States_(POMS)': self.score_poms,
            # Outcome Measures
            'Toronto_Mindfulness_Scale': self.score_toronto_mind_scale,
            # Resilience, Flexibility, Burnout
            'Copenhagen_Burnout_Inventory_(CBI)': self.score_cbi,
            'Dispositional_Resilience_\'Hardiness\'_Scale_(HARDY)': self.score_hardy,
            # Misc
            'Karolinska_Sleepiness_Scale_(KSS)': self.score_kss,
            'Wong-Baker_Pain_Scale': self.score_wb_pain,
            'Overall_Anxiety_Severity_and_Impairment_Scale_(OASIS)': self.score_oasis,
            'Sheehan_Disability_Scale_(SDS)': self.score_sds,
            'Brief_Symptom_Inventory-18_(BSI-18)': self.score_bsi_18,
            'Adverse_Childhood_Experience_(ACE)': self.score_ace,
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
        print("\n[DEBUG] Function: calculate_all_scales entered\n")
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
                    # Drop duplicate columns if they already exist
                    overlapping_columns = set(self.df.columns) & set(score_result.columns)
                    if overlapping_columns:
                        print(f"[INFO] Dropping overlapping columns: {overlapping_columns}")
                        self.df = self.df.drop(columns=list(overlapping_columns), errors='ignore')

                    # Append the new scores DataFrame
                    results.append(score_result)
                    self.question_columns_to_drop.extend(matching_columns)
                else:
                    # If it returns a Series, add it directly to the DataFrame
                    column_name = scale_name
                    if column_name in self.df.columns:
                        print(f"[INFO] Dropping existing column: {column_name}")
                        self.df = self.df.drop(columns=[column_name], errors='ignore')

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

        print("\n[DEBUG] Function: calculate_all_scales completed\n")

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
        Calculate the subscale scores for the Multidimensional Assessment of Interoceptive Awareness (MAIA).

        Parameters:
        -----------
        columns : list
            A list of column names corresponding to the 32 MAIA items, in the correct order.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the subscale scores for each MAIA subscale.
        """
        # Ensure the correct number of columns (32 items for MAIA expected)
        if len(columns) != 32:
            raise ValueError(f"Expected 32 columns, but got {len(columns)}")

        # Unpack the column names directly as q1, q2, ..., q32
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
            q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32
        ) = columns

        # Define subscales
        subscales = {
            'Noticing': [q1, q2, q3, q4],
            'Not-Distracting': [q5, q6, q7],  # Reverse q5, q6, q7
            'Not-Worrying': [q8, q9, q10],   # Reverse q8, q9
            'Attention_Regulation': [q11, q12, q13, q14, q15, q16, q17],
            'Emotional_Awareness': [q18, q19, q20, q21, q22],
            'Self-Regulation': [q23, q24, q25, q26],
            'Body_Listening': [q27, q28, q29],
            'Trusting': [q30, q31, q32]
        }

        # Reverse-score the specified items
        reversed_items = [q5, q6, q7, q8, q9]
        self.df[reversed_items] = 5 - self.df[reversed_items]

        # Calculate the average for each subscale
        subscale_scores = {
            subscale: self.df[items].mean(axis=1)
            for subscale, items in subscales.items()
        }

        # Create a DataFrame with the subscale scores
        scores_df = pd.DataFrame(subscale_scores)

        return scores_df

    def score_maia_s(self, columns):
        """
        Calculate the subscale scores for the MAIA-S using the MAIA scoring function.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame.
        columns : dict
            Dictionary mapping subscale names to their corresponding column names (shared with MAIA).

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the subscale scores for MAIA-S.
        """
        # Simply call score_maia with the same columns
        return self.score_maia(columns)


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

        Parameters:
        -----------
        columns : list
            A list with the column names.

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

    def score_neo_pi_3(self, columns):
        """
        Calculate the Openness to Experience scale

        Parameters:
        -----------
        columns : list
            A list with the column names.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the scores for scale
        """
        # Ensure the columns are correctly passed
        if len(columns) != 40:
            raise ValueError(f"Expected 60 columns, but got {len(columns)}")

        # Unpack questions
        (
            c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
            c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26,
            c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
            c40, c41, c42, c43, c44, c45, c46, c47, c48
        ) = columns

        # Calculate subscales
        fantasy = (self.df[c1] + (6 - self.df[c2]) + self.df[c3] +
                   (6 - self.df[c4]) + self.df[c5] + (6 - self.df[c6]) + (6 - self.df[c7])
                   + (6 - self.df[c8]))
        aesthetics = (6 - self.df[c9] + self.df[c10] + (6 - self.df[c11])
                      + self.df[c12] + (6 - self.df[c13]) + self.df[c14] + self.df[c15] + self.df[c16])
        feelings = (self.df[c17] + (6 - self.df[c18]) + self.df[c19] + (6 - self.df[c20]) + self.df[c21]
                    + (6 - self.df[c22]) + self.df[c23] + self.df[c24])
        actions = ((6 - self.df[c25]) + self.df[c26] + (6 - self.df[c27]) + self.df[c28] + (6- self.df[c29])
                   + self.df[c30] + (6 - self.df[c31]) + (6 - self.df[c32]))
        ideas = (self.df[c33] + (6 - self.df[c34]) + self.df[c35] + (6 - self.df[c36]) + self.df[c37]
                 + (6 - self.df[c38]) + self.df[c39] + self.df[c40])
        values = ((6 - self.df[c41]) + self.df[c42] + (6 - self.df[c43]) + self.df[c44]
                  + (6 - self.df[c45]) + self.df[c46] + (6 - self.df[c47]) + self.df[c48])

        # Create dataframe with subscales
        scores_df = pd.DataFrame({
            'Fantasy': fantasy,
            'Aesthetics': aesthetics,
            'Feelings': feelings,
            'Actions': actions,
            'Ideas': ideas,
            'Values': values
        })

        # Return dataframe
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

        # Define reverse scoring
        def reverse(q):
            return 6 - self.df[q]

        # Define subscale items
        observing_items = [columns[i - 1] for i in [1, 6, 11, 15, 20, 26, 31, 36]]
        describing_items = [columns[i - 1] for i in [2, 7, 27, 32, 37]]
        describing_reversed = [columns[i - 1] for i in [12, 16, 22]]
        acting_awareness_reversed = [columns[i - 1] for i in [5, 8, 13, 18, 23, 28, 34, 38]]
        nonjudging_reversed = [columns[i - 1] for i in [3, 10, 14, 17, 25, 30, 35, 39]]
        nonreactivity_items = [columns[i - 1] for i in [4, 9, 19, 21, 24, 29, 33]]

        # Calculate subscale scores
        observing = self.df[observing_items].sum(axis=1)
        describing = self.df[describing_items].sum(axis=1) + reverse(describing_reversed).sum(axis=1)
        acting_with_awareness = reverse(acting_awareness_reversed).sum(axis=1)
        nonjudging = reverse(nonjudging_reversed).sum(axis=1)
        nonreactivity = self.df[nonreactivity_items].sum(axis=1)

        # Calculate the total FFMQ score
        total_score = observing + describing + acting_with_awareness + nonjudging + nonreactivity

        # Create a DataFrame with all subscale and total scores
        scores_df = pd.DataFrame({
            'ffmq_observing': observing,
            'ffmq_describing': describing,
            'ffmq_acting_with_awareness': acting_with_awareness,
            'ffmq_nonjudging': nonjudging,
            'ffmq_nonreactivity': nonreactivity,
            'total_ffmq_score': total_score
        })

        return scores_df

    def score_panas(self, columns):
        """
        Calculate the Positive and Negative Affect Schedule (PANAS) scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the PANAS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing positive and negative affect scores.
        """
        # Debug the incoming columns
        print(f"[DEBUG] PANAS columns: {columns}")
        
        # Instead of direct unpacking, use a more flexible approach
        # Find the columns by matching partial names
        positive_affect_cols = []
        negative_affect_cols = []
        
        # Positive affect items
        positive_items = ["interested", "excited", "strong", "enthusiastic", "proud", 
                        "alert", "inspired", "determined", "attentive", "active"]
        
        # Negative affect items
        negative_items = ["distressed", "upset", "guilty", "scared", "hostile", 
                        "irritable", "ashamed", "nervous", "jittery", "afraid"]
        
        # Find matching columns
        for col in columns:
            col_lower = col.lower()
            
            # Check if it matches a positive affect item
            if any(item in col_lower for item in positive_items):
                positive_affect_cols.append(col)
                
            # Check if it matches a negative affect item
            if any(item in col_lower for item in negative_items):
                negative_affect_cols.append(col)
        
        print(f"[DEBUG] Positive affect columns found: {positive_affect_cols}")
        print(f"[DEBUG] Negative affect columns found: {negative_affect_cols}")
        
        # Calculate subscale scores
        if positive_affect_cols:
            positive_affect = self.df[positive_affect_cols].sum(axis=1)
        else:
            positive_affect = pd.Series(np.nan, index=self.df.index)
            
        if negative_affect_cols:
            negative_affect = self.df[negative_affect_cols].sum(axis=1)
        else:
            negative_affect = pd.Series(np.nan, index=self.df.index)
        
        # Create output dataframe
        results = pd.DataFrame({
            'PANAS_Positive_Affect': positive_affect,
            'PANAS_Negative_Affect': negative_affect
        })
        
        return results

    def score_panas_x(self, columns):
        """
        Calculate subcategory and total PANAS-X scores.

        Parameters:
        -----------
        columns : list
            A list of column names corresponding to the PANAS-X questions, in the correct order.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing all subcategory scores and the higher-order Positive and Negative Affect scores.
        """
        # Ensure the correct number of columns (60 items for PANAS-X expected)
        if len(columns) != 60:
            raise ValueError(f"Expected 60 columns, but got {len(columns)}")

        # Unpack columns directly
        (
            cheerful, sad, active, angry_at_self,
            disgusted, calm, guilty, enthusiastic,
            attentive, afraid, joyful, downhearted,
            bashful, tired, nervous, sheepish,
            sluggish, amazed, lonely, distressed,
            daring, shaky, sleepy, blameworthy,
            surprised, happy, excited, determined,
            strong, timid, hostile, frightened,
            scornful, alone, proud, astonished,
            relaxed, alert, jittery, interested,
            irritable, upset, lively, loathing,
            delighted, angry, ashamed, confident,
            inspired, bold, at_ease, energetic,
            fearless, blue, scared, concentrating,
            disgusted_with_self, shy, drowsy, dissatisfied_with_self
        ) = columns

        # Define the PANAS-X scales based on question names in the specified order
        positive_affect = [active, alert, attentive, determined, enthusiastic, excited, inspired, interested, proud, strong]
        negative_affect = [afraid, scared, nervous, jittery, irritable, hostile, guilty, ashamed, upset, distressed]

        fear = [afraid, scared, frightened, nervous, jittery, shaky]
        hostility = [angry, hostile, irritable, scornful, disgusted, loathing]
        guilt = [guilty, ashamed, blameworthy, angry_at_self, disgusted_with_self, dissatisfied_with_self]
        sadness = [sad, blue, downhearted, alone, lonely]
        joviality = [happy, joyful, delighted, cheerful, excited, enthusiastic, lively, energetic]
        self_assurance = [proud, strong, confident, bold, daring, fearless]
        attentiveness = [alert, attentive, concentrating, determined]
        shyness = [shy, bashful, sheepish, timid]
        fatigue = [sleepy, tired, sluggish, drowsy]
        serenity = [calm, relaxed, at_ease]
        surprise = [amazed, surprised, astonished]

        # Store scale definitions in a dictionary
        scales = {
            'panasx_positive_affect': positive_affect,
            'panasx_negative_affect': negative_affect,
            'panasx_fear': fear,
            'panasx_hostility': hostility,
            'panasx_guilt': guilt,
            'panasx_sadness': sadness,
            'panasx_joviality': joviality,
            'panasx_self-assurance': self_assurance,
            'panasx_attentiveness': attentiveness,
            'panasx_shyness': shyness,
            'panasx_fatigue': fatigue,
            'panasx_serenity': serenity,
            'panasx_surprise': surprise
        }

        # Calculate scores for each scale
        scores = {
            scale_name: self.df[items].sum(axis=1)
            for scale_name, items in scales.items()
        }

        # Create a DataFrame from the scores
        scores_df = pd.DataFrame(scores)

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

        # Unpack the 95 questions directly
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
            'miss_consumer_suggestibility': consumer,
            'miss_persuadability': persuadability,
            'miss_physiological_suggestibility': physiological,
            'miss_physiological_reactivity': physiological_reactivity,
            'miss_peer_conformity': peer_conformity,
            'miss_mental_control': mental_control,
            'miss_unpersuadability': unpersuadability,
            'miss_short_suggestibility_scale_(SSS)': sss,
            'miss_total_suggestibility': total_suggestibility
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
            'tms_curiosity': curiosity,
            'tms_de-centering': decentering,
            'toronto_mindfulness_scale_total': total
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
            'cbi_personal_burnout': self.df[[q1, q2, q3, q4, q5, q6]],
            'cbi_work_related_burnout': self.df[[q7, q8, q9, q10, q13, q14, q15]],
            'cbi_client_related_burnout': self.df[[q11, q12, q16, q17, q18, q19]]
        }

        # Calculate scores
        scored_subscales = {}
        for subscale, cols in subscales.items():
            # Use the inferred scale for the first column in each subscale
            scale_key = infer_scale_mapping(cols[0])
            scores = self.df[cols].applymap(lambda x: get_score_from_mapping(x, scale_key))
            scored_subscales[subscale] = scores.mean(axis=1, skipna=True)

        # Calculate total average
        scored_subscales['cbi_total'] = pd.DataFrame(scored_subscales).mean(axis=1, skipna=True)

        return pd.DataFrame(scored_subscales)

    def score_hardy(self, columns):
        """
        Score the Dispositional Resilience 'Hardiness' Scale (HARDY) with inferred mappings.

        Parameters:
        ----------
        columns : list
            List of column values for the HARDY questions (should be numerical values for scoring).

        Returns:
        -------
        pd.DataFrame
            DataFrame with corrected scores for each HARDY subscale and total.
        """
        # Ensure that the correct number of columns is provided
        if len(columns) != 45:
            raise ValueError(f"Expected 45 columns, but got {len(columns)}")

        # Unpack the questions directly
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
            q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32, q33, q34, q35, q36, q37, q38, q39, q40,
            q41, q42, q43, q44, q45
        ) = columns

        # Define forward and reverse lists for each subscale
        comm_forward = [q1, q8, q17, q25, q39]
        comm_reverse = [q7, q9, q18, q23, q24, q31, q37, q41, q44, q45]

        chal_forward = [q15, q21, q30, q33, q36]
        chal_reverse = [q5, q6, q12, q16, q20, q27, q32, q35, q38, q40]

        cont_forward = [q2, q13, q19, q22, q42]
        cont_reverse = [q3, q4, q10, q11, q14, q26, q28, q29, q34, q43]

        # Reverse scoring function - only takes one parameter
        def reverse_score(items):
            # For scale 0-3
            return 3 - self.df[items]

        # Compute subscale scores
        hardy_comm = (
                self.df[comm_forward].sum(axis=1) + reverse_score(comm_reverse).sum(axis=1)
        )
        hardy_chal = (
                self.df[chal_forward].sum(axis=1) + reverse_score(chal_reverse).sum(axis=1)
        )
        hardy_cont = (
                self.df[cont_forward].sum(axis=1) + reverse_score(cont_reverse).sum(axis=1)
        )

        # Compute total score
        hardy_tot = hardy_comm + hardy_chal + hardy_cont

        # Return as DataFrame for each subscale and total
        return pd.DataFrame({
            'HARDY_Communication_Score': hardy_comm,
            'HARDY_Challenge_Score': hardy_chal,
            'HARDY_Control_Score': hardy_cont,
            'HARDY_Total_Score': hardy_tot
        })

    def score_madrs(self, columns):
        """
        Calculate the total MADRS score based on individual item scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the MADRS items, in the correct order.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the total MADRS score.
        """
        # Ensure the correct number of columns (10 items for MADRS expected)
        if len(columns) != 10:
            raise ValueError(f"Expected 10 columns, but got {len(columns)}")

        # Calculate the total score as the sum of all items
        total_score = self.df[columns].sum(axis=1, skipna=True)

        # Create a DataFrame to hold the score
        scores_df = pd.DataFrame({
            'madrs_total_score': total_score
        })

        return scores_df

    def score_ham_a(self, columns):
        """
        Calculate the total HAM-A score based on individual item scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the HAM-A items, in the correct order.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the total HAM-A score and severity classification.
        """
        # Ensure the correct number of columns (14 items for HAM-A expected)
        if len(columns) != 14:
            raise ValueError(f"Expected 14 columns, but got {len(columns)}")

        # Calculate the total score as the sum of all items
        total_score = self.df[columns].sum(axis=1, skipna=True)

        conditions = [
            total_score < 17,
            (total_score >= 17) & (total_score <= 24),
            (total_score >= 25) & (total_score <= 30),
            total_score > 30
        ]
        choices = ['Mild', 'Mild to Moderate', 'Moderate to Severe', 'Severe']
        severity = np.select(conditions, choices, default='Unknown')

        # Create a DataFrame to hold the total score and severity classification
        scores_df = pd.DataFrame({
            'hama_total_score': total_score,
            'hama_severity': severity
        })

        return scores_df

    def score_stai(self, columns):
        """
        Calculate the total scores for the State Anxiety Inventory (SAI) and Trait Anxiety Inventory (TAI)
        based on the number of items provided.

        Parameters:
        -----------
        columns : list
            A list of column names corresponding to the STAI items. If 20 columns are provided, it calculates
            only the STAI-State score. If 40 columns are provided, it calculates both STAI-State and STAI-Trait scores.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the total score(s) for SAI and/or TAI.
        """
        # If only 20 columns are provided, assume they are STAI-State items
        if len(columns) == 20:
            (
                q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                q11, q12, q13, q14, q15, q16, q17, q18, q19, q20
            ) = columns

            # Define reversed items for STAI-State
            reversed_sai_items = [q1, q2, q5, q8, q10, q11, q15, q16, q19, q20]

            # Adjust reversed items: Transform `x` to `5 - x` (since the scale is 1 to 4)
            self.df[reversed_sai_items] = 5 - self.df[reversed_sai_items]

            # Calculate STAI-State total score
            stai_state_total = self.df[columns].sum(axis=1) + 50

            # Return a DataFrame with only the STAI-State score
            scores_df = pd.DataFrame({
                'STAI_State_Total_Score': stai_state_total
            })

        # If 40 columns are provided, calculate both STAI-State and STAI-Trait scores
        elif len(columns) == 40:
            (
                q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                q11, q12, q13, q14, q15, q16, q17, q18, q19, q20,
                q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
                q31, q32, q33, q34, q35, q36, q37, q38, q39, q40
            ) = columns

            # Define reversed items for both STAI-State and STAI-Trait
            reversed_sai_items = [q1, q2, q5, q8, q10, q11, q15, q16, q19, q20]
            reversed_tai_items = [q21, q26, q27, q30, q33, q36, q39]

            # Adjust reversed items
            self.df[reversed_sai_items] = 5 - self.df[reversed_sai_items]
            self.df[reversed_tai_items] = 5 - self.df[reversed_tai_items]

            # Calculate STAI-State and STAI-Trait total scores
            sai_columns = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
                           q11, q12, q13, q14, q15, q16, q17, q18, q19, q20]
            tai_columns = [q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
                           q31, q32, q33, q34, q35, q36, q37, q38, q39, q40]

            # Add constant scores for SAI and TAI
            stai_state_total = self.df[sai_columns].sum(axis=1) + 50
            stai_trait_total = self.df[tai_columns].sum(axis=1) + 35

            # Return a DataFrame with both scores
            scores_df = pd.DataFrame({
                'STAI_State_Total_Score': stai_state_total,
                'STAI_Trait_Total_Score': stai_trait_total
            })

        else:
            raise ValueError("Expected either 20 or 40 columns for scoring.")

        return scores_df

    def score_5dasc(self, columns):
        """
        Scores the 5D-ASC questionnaire, including subscales and main scales.

        Parameters:
        ----------
        columns : list
            List of column names corresponding to the 5D-ASC questions (Q1 to Q94).

        Returns:
        -------
        pd.DataFrame
            DataFrame with scores for all subscales and main scales.
        """
        # Ensure the correct number of columns
        if len(columns) != 94:
            raise ValueError(f"Expected 94 columns, but got {len(columns)}")

        # Unpack the question columns
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16,
            q17, q18, q19, q20, q21, q22, q23, q24, q25, q26, q27, q28, q29, q30,
            q31, q32, q33, q34, q35, q36, q37, q38, q39, q40, q41, q42, q43, q44,
            q45, q46, q47, q48, q49, q50, q51, q52, q53, q54, q55, q56, q57, q58,
            q59, q60, q61, q62, q63, q64, q65, q66, q67, q68, q69, q70, q71, q72,
            q73, q74, q75, q76, q77, q78, q79, q80, q81, q82, q83, q84, q85, q86,
            q87, q88, q89, q90, q91, q92, q93, q94
        ) = columns

        # Define subscale items
        subscale_items = {
            'Experience_of_Unity': [q18, q34, q41, q42, q52],
            'Spiritual_Experience': [q9, q81, q94],
            'Blissful_State': [q12, q86, q91],
            'Insightfulness': [q50, q69, q77],
            'Disembodiment': [q26, q62, q63],
            'Impaired_Control_and_Cognition': [q8, q27, q38, q47, q64, q67, q78],
            'Anxiety': [q32, q43, q44, q46, q56, q89],
            'Complex_Imagery': [q39, q72, q82],
            'Elementary_Imagery': [q14, q22, q33],
            'Audio-Visual_Synesthesiae': [q20, q23, q75],
            'Changed_Meaning_of_Percepts': [q28, q31, q54]
        }

        # Define main scale subscales
        main_scale_subscales = {
            'Oceanic_Boundlessness_(OB)': {
                'Positive_Derealization': [q1, q9, q18, q34, q57, q71, q87],
                'Positive_Depersonalization': [q16, q26, q62, q63],
                'Altered_Perception_of_Time_and_Space': [q36, q41, q52]
            },
            'Anxious_Ego_Dissolution_(AED)': {
                'Negative_Derealization': [q21, q43, q44, q46, q64, q85],
                'Thought_Disorder': [q27, q38, q67, q88],
                'Paranoid_Ideation': [q6, q56, q89]
            },
            'Visual_Restructuralization_(VR)': {
                'Simple_Hallucinations': [q14, q22, q33, q83],
                'Complex_Hallucinations': [q7, q39]
            },
            'Auditory_Alterations': {  # Items directly
                'Auditory_Alterations': [q4, q5, q11, q13, q19, q25, q30, q48, q49, q55, q65, q74, q76, q92, q93]
            },
            'Reduction_of_Vigilance': {  # Items directly
                'Reduction_of_Vigilance': [q2, q10, q15, q17, q24, q29, q37, q51, q61, q68, q84]
            }
        }

        # Calculate subscale scores row-wise
        subscale_scores = {}
        for subscale, items in subscale_items.items():
            subscale_scores[subscale] = self.df[items].mean(axis=1)  # Mean score for each row

        # Calculate main scale scores (sum of subscale scores)
        main_scale_scores = {}
        for scale, subscales in main_scale_subscales.items():
            subscale_scores_in_scale = []
            for subscale_name, items in subscales.items():
                subscale_scores_in_scale.append(self.df[items].mean(axis=1))  # Mean score for subscale
            main_scale_scores[scale] = sum(subscale_scores_in_scale)  # Sum of subscale scores for each row

        # Combine subscale and main scale scores
        result = {**subscale_scores, **main_scale_scores}

        # Return scores as a DataFrame
        return pd.DataFrame(result)

    def score_asi3(self, columns):
        """
        Scores the Anxiety Sensitivity Index-3 (ASI-3) questionnaire.

        Parameters:
        ----------
        responses : list
            List of responses (0-4) corresponding to the 18 ASI-3 items.

        Returns:
        -------
        pd.DataFrame
            DataFrame with scores for ASI-3 Total and the three subscales (Physical, Cognitive, Social).
        """
        print("\n[DEBUG] Entered function score_asi3\n")
        # Ensure the correct number of responses
        if len(columns) != 18:
            raise ValueError(f"Expected 18 responses, but got {len(columns)}")

        # Unpack responses into individual question variables
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12, q13, q14, q15, q16, q17, q18
        ) = columns

        # Define subscale item mappings
        physical_concerns_items = [q4, q12, q8, q7, q15, q3]
        cognitive_concerns_items = [q14, q18, q10, q16, q2, q5]
        social_concerns_items = [q9, q6, q11, q13, q17, q1]
        # Calculate subscale scores row-wise
        self.df['Physical_Concerns_Score'] = self.df[physical_concerns_items].sum(axis=1)
        self.df['Cognitive_Concerns_Score'] = self.df[cognitive_concerns_items].sum(axis=1)
        self.df['Social_Concerns_Score'] = self.df[social_concerns_items].sum(axis=1)

        # Calculate total score row-wise
        self.df['Total_Score'] = (
                self.df['Physical_Concerns_Score'] +
                self.df['Cognitive_Concerns_Score'] +
                self.df['Social_Concerns_Score']
        )

        # Return scores as a DataFrame
        return self.df[[
            'Total_Score',
            'Physical_Concerns_Score',
            'Cognitive_Concerns_Score',
            'Social_Concerns_Score'
        ]]

    def score_kss(self, column):
        """
        Score the Karolinska Sleepiness Scale (KSS).

        Parameters:
        ----------
        column : str
            The column name containing the KSS scores.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the KSS score.
        """
        kss_score = self.df[column]

        return kss_score

    def score_wb_pain(self, column):
        """
        Score the Wong-Baker Pain Scale.

        Parameters:
        ----------
        column : str
            The column name containing the WB Pain scores.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the WB Pain score.
        """
        pain_score = self.df[column]

        return pain_score

    def score_oasis(self, columns):
        """
        Scores the Overall Anxiety Severity and Impairment Scale (OASIS).

        Parameters:
        ----------
        columns : list
            List of column names corresponding to the 5 OASIS items.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the OASIS total score and severity classification.
        """
        # Ensure the correct number of columns
        if len(columns) != 5:
            raise ValueError(f"Expected 5 columns, but got {len(columns)}")

        # Calculate the total OASIS score
        total_score = self.df[columns].sum(axis=1)

        # Severity classification based on the total score
        severity = total_score.apply(
            lambda x: "Minimal" if x < 8 else
            "Elevated"
        )

        # Create a DataFrame to store the results
        result_df = pd.DataFrame({
            'OASIS_Total_Score': total_score,
            'OASIS_Severity': severity
        })

        return result_df

    def score_phq9(self, columns):
        """
        Scores the Patient Health Questionnaire (PHQ-9).

        Parameters:
        ----------
        columns : list
            List of column names corresponding to the 9 PHQ-9 items.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the PHQ-9 total score and severity classification.
        """
        # Ensure the correct number of columns
        if len(columns) != 9:
            raise ValueError(f"Expected 9 columns, but got {len(columns)}")

        # Calculate the total PHQ-9 score
        total_score = self.df[columns].sum(axis=1)

        # Classify severity based on the total score
        severity = total_score.apply(
            lambda x: 'Minimal depression' if 0 <= x <= 4 else
            'Mild depression' if 5 <= x <= 9 else
            'Moderate depression' if 10 <= x <= 14 else
            'Moderately severe depression' if 15 <= x <= 19 else
            'Severe depression'
        )

        # Create a DataFrame to store the results
        result_df = pd.DataFrame({
            'PHQ9_Total_Score': total_score,
            'PHQ9_Severity': severity
        })

        return result_df

    def score_sds(self, columns):
        """
        Scores the Sheehan Disability Scale (SDS).

        Parameters:
        ----------
        columns : list
            List of column names corresponding to the 3 SDS items: work/school, social life, and family life.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the total SDS score and flags for significant functional impairment in each domain.
        """
        # Ensure the correct number of columns
        if len(columns) != 3:
            raise ValueError(f"Expected 3 columns, but got {len(columns)}")

        # Extract the individual columns
        work_school = self.df[columns[0]]
        social_life = self.df[columns[1]]
        family_life = self.df[columns[2]]

        # Calculate the total SDS score
        total_score = work_school + social_life + family_life

        # Flag significant functional impairment (score ≥ 5 in any domain)
        work_school_flag = work_school.apply(lambda x: 1 if x >= 5 else 0)
        social_life_flag = social_life.apply(lambda x: 1 if x >= 5 else 0)
        family_life_flag = family_life.apply(lambda x: 1 if x >= 5 else 0)

        # Combine the results into a DataFrame
        result_df = pd.DataFrame({
            'SDS_Work_School': work_school,
            'SDS_Social_Life': social_life,
            'SDS_Family_Life': family_life,
            'SDS_Total_Score': total_score,
            'SDS_Work_School_Impairment': work_school_flag,
            'SDS_Social_Life_Impairment': social_life_flag,
            'SDS_Family_Life_Impairment': family_life_flag
        })

        return result_df

    def score_bsi_18(self, columns):
        # TODO
        pass

    def score_ace(self, columns):
        """
        Score the Adverse Childhood Experiences (ACE) Questionnaire.

        Parameters:
        ----------
        columns : list
            List of 10 column names corresponding to the ACE questionnaire items.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the ACE total score.
       """
        # Ensure the correct number of columns
        if len(columns) != 10:
            raise ValueError(f"Expected 10 columns, but got {len(columns)}")

        ace_score = self.df[columns].sum(axis=1)

        return ace_score
    
    def score_dhs(self, columns):
        """
        Score the Dispositional Hope Scale (DHS).

        Parameters:
        ----------
        columns : list
            List of 12 column names corresponding to the DHS questionnaire items.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the DHS total score.
       """
        # Ensure the correct number of columns
        if len(columns) != 12:
            raise ValueError(f"Expected 12 columns, but got {len(columns)}")
        
        # Unpack responses into individual question variables
        (
            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10,
            q11, q12
        ) = columns

        # Score subscales
        pathways_score = self.df[[q1, q4, q6, q8]].sum(axis=1)
        agency_score = self.df[[q2, q9, q10, q12]].sum(axis=1)

        # Calculate total DHS
        total_dhs = pathways_score + agency_score

        # Create a DataFrame with all scores
        scores_df = pd.DataFrame({
            'DHS_Pathways_Score': pathways_score,
            'DHS_Agency_Score': agency_score,
            'DHS_Total_Hope_Score': total_dhs
        })

        return scores_df
    
    def score_gses(self, columns):
        """
        Calculate the General Self-Efficacy Scale (GSES) score.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the GSES questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated GSES total scores.
        """
        # Ensure the correct number of columns
        if len(columns) != 10:
            raise ValueError(f"Expected 10 columns, but got {len(columns)}")

        # Calculate total GSES score by summing all items
        gses_total_score = self.df[columns].sum(axis=1)

        return gses_total_score
    
    def score_cd_risc_10(self, columns):
        """
        Calculate the Connor-Davidson Resilience Scale (CD-RISC-10) score.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the CD-RISC-10 questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated CD-RISC-10 total scores.
        """
        # Ensure the correct number of columns
        if len(columns) != 10:
            raise ValueError(f"Expected 10 columns, but got {len(columns)}")

        # Calculate total CD-RISC-10 score by summing all items
        cd_risc_10_total_score = self.df[columns].sum(axis=1)

        return cd_risc_10_total_score
    
    def score_poms(self, columns):
        """
        Calculate the Profile of Mood States (POMS) scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the POMS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing subscale scores and Total Mood Disturbance (TMD).
        """
        # Ensure the correct number of columns
        if len(columns) != 31:
            raise ValueError(f"Expected 31 columns, but got {len(columns)}")

        # Direct unpacking of columns in the order they'll be passed
        (
            tense, angry, worn_out, lively, confused, shaky, sad, active, grouchy, energetic,
            unworthy, uneasy, fatigued, annoyed, discouraged, nervous, lonely, muddled, exhausted, anxious,
            good_natured, gloomy, sluggish, weary, bewildered, furious, efficient, full_of_pep, bad_tempered, forgetful, 
            vigorous
        ) = columns

        # Group columns by subscale
        tension_cols = [tense, shaky, uneasy, nervous, anxious]
        depression_cols = [sad, unworthy, discouraged, lonely, gloomy]
        anger_cols = [angry, grouchy, annoyed, furious, bad_tempered]
        vigor_cols = [lively, active, energetic, good_natured, full_of_pep, vigorous]
        fatigue_cols = [worn_out, fatigued, exhausted, sluggish, weary]
        confusion_cols = [confused, muddled, bewildered, forgetful]

        # Calculate subscale scores
        tension_sum = self.df[tension_cols].sum(axis=1)
        depression_sum = self.df[depression_cols].sum(axis=1)
        anger_sum = self.df[anger_cols].sum(axis=1)
        vigor_sum = self.df[vigor_cols].sum(axis=1)
        fatigue_sum = self.df[fatigue_cols].sum(axis=1)
        confusion_sum = self.df[confusion_cols].sum(axis=1)
        
        # Calculate Total Mood Disturbance (TMD)
        tmd = tension_sum + depression_sum + anger_sum + fatigue_sum + confusion_sum - vigor_sum
        
        # Create output dataframe
        results = pd.DataFrame({
            'POMS_Tension': tension_sum,
            'POMS_Depression': depression_sum,
            'POMS_Anger': anger_sum,
            'POMS_Fatigue': fatigue_sum,
            'POMS_Confusion': confusion_sum,
            'POMS_Vigor': vigor_sum,
            'POMS_Total_Mood_Disturbance': tmd
        })
        
        return results
    
    def score_svs(self, columns):
        """
        Calculate the Subjective Vitality Scale (SVS) score.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the SVS questions.

        Returns:
        --------
        pd.Series
            A series containing the calculated SVS mean scores.
        """
        # Ensure the correct number of columns
        if len(columns) != 6:
            raise ValueError(f"Expected 6 columns, but got {len(columns)}")

        # Calculate SVS score by averaging all items
        svs_score = self.df[columns].mean(axis=1)

        return svs_score
    
    def score_fss_short(self, columns):
        """
        Calculate the Flow State Scale (FSS) short version scores.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the FSS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the ABA factor score, FP factor score, and total flow score.
        """
        # Ensure the correct number of columns
        if len(columns) != 9:
            raise ValueError(f"Expected 9 columns, but got {len(columns)}")

        # Define which items belong to which factor
        aba_items = [columns[0], columns[2], columns[5]]  # items 1, 3, 6
        fp_items = [columns[1], columns[3], columns[4], columns[6], columns[7], columns[8]]  # items 2, 4, 5, 7, 8, 9

        # Calculate factor scores
        aba_score = self.df[aba_items].sum(axis=1)
        fp_score = self.df[fp_items].sum(axis=1)
        
        # Calculate total flow score (sum of all items)
        total_flow_score = self.df[columns].sum(axis=1)

        # Create results DataFrame
        results = pd.DataFrame({
            'FSS_Absorption_by_Activity': aba_score,
            'FSS_Fluency_of_Performance': fp_score,
            'FSS_Total_Flow_Score': total_flow_score
        })

        return results
    
    def score_pil(self, columns):
        """
        Calculate the Purpose in Life Test (PIL) score.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the PIL questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the total PIL score and classification.
        """
        # Ensure the correct number of columns
        if len(columns) != 20:
            raise ValueError(f"Expected 20 columns, but got {len(columns)}")

        # Calculate the total PIL score by summing all items
        total_score = self.df[columns].sum(axis=1)
        
        # Classify PIL scores based on typical interpretation ranges
        # (Note: Adjust these ranges if different classification criteria are used)
        def classify_pil(score):
            if score <= 91:
                return "Low Purpose"
            elif score <= 112:
                return "Moderate Purpose"
            else:
                return "High Purpose"
        
        classification = total_score.apply(classify_pil)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'PIL_Total_Score': total_score,
            'PIL_Classification': classification
        })

        return results
    
    def score_pss(self, columns):
        """
        Calculate the Perceived Stress Scale (PSS) score.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the PSS questions.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the PSS total score and stress level classification.
        """
        # Ensure the correct number of columns
        if len(columns) != 10:
            raise ValueError(f"Expected 10 columns, but got {len(columns)}")

        # Unpack the 10 questions directly
        (q1, q2, q3, q4, q5, q6, q7, q8, q9, q10) = columns
        
        # Calculate regular scores (items 1, 2, 3, 6, 9, 10)
        regular_items = [q1, q2, q3, q6, q9, q10]
        regular_score = self.df[regular_items].sum(axis=1)
        
        # Calculate reversed scores (items 4, 5, 7, 8)
        reversed_items = [q4, q5, q7, q8]
        reversed_score = (4 - self.df[reversed_items]).sum(axis=1)
        
        # Calculate total PSS score
        total_score = regular_score + reversed_score
        
        # Classify stress levels
        def classify_stress(score):
            if score <= 13:
                return "Low Stress"
            elif score <= 26:
                return "Moderate Stress"
            else:
                return "High Stress"
        
        stress_classification = total_score.apply(classify_stress)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'PSS_Total_Score': total_score,
            'PSS_Stress_Level': stress_classification
        })
        
        return results
    
    def score_mhlc(self, columns):
        """
        Calculate the Multidimensional Health Locus of Control (MHLC) scores
        for both Form A and Form B.

        Parameters:
        -----------
        columns : list
            A list with the column names corresponding to the MHLC questions
            (36 total - Form A followed by Form B).

        Returns:
        --------
        pd.DataFrame
            DataFrame containing subscale scores for both Form A and Form B:
            Internality, Powerful Others Externality, and Chance Externality for each form.
        """
        # Ensure the correct number of columns
        if len(columns) != 36:
            raise ValueError(f"Expected 36 columns, but got {len(columns)}")

        # Unpack the 36 questions - first 18 for Form A, last 18 for Form B
        (
            # Form A questions (1-18)
            a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
            a11, a12, a13, a14, a15, a16, a17, a18,
            # Form B questions (19-36)
            b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
            b11, b12, b13, b14, b15, b16, b17, b18
        ) = columns
        
        # Calculate Form A subscale scores
        a_internality_score = self.df[[a1, a6, a8, a12, a13, a17]].sum(axis=1)
        a_powerful_others_score = self.df[[a3, a5, a7, a10, a14, a18]].sum(axis=1)
        a_chance_score = self.df[[a2, a4, a9, a11, a15, a16]].sum(axis=1)
        
        # Calculate Form B subscale scores
        b_internality_score = self.df[[b1, b6, b8, b12, b13, b17]].sum(axis=1)
        b_powerful_others_score = self.df[[b3, b5, b7, b10, b14, b18]].sum(axis=1)
        b_chance_score = self.df[[b2, b4, b9, b11, b15, b16]].sum(axis=1)
        
        # Create results DataFrame with separate scores for each form
        results = pd.DataFrame({
            'MHLC_FormA_Internality': a_internality_score,
            'MHLC_FormA_Powerful_Others': a_powerful_others_score,
            'MHLC_FormA_Chance': a_chance_score,
            'MHLC_FormB_Internality': b_internality_score,
            'MHLC_FormB_Powerful_Others': b_powerful_others_score,
            'MHLC_FormB_Chance': b_chance_score
        })
        
        return results
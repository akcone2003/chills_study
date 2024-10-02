import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import sys
from scripts.pipeline import (
    handle_missing_values, detect_outliers, generate_qa_report, preprocess_for_output,
    process_data_pipeline, detect_column_types
)


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataframe similar to a survey structure."""
        self.df = pd.DataFrame({
            'Age': [40, 45, np.nan, 30],  # Numerical with a missing value
            'Gender': ['Female', 'Male', 'Female', 'Trans-Male'],  # Categorical
            'Commitment': ['I will provide my best answers'] * 4,  # Free text
            'Experience_Meditation': [2, 1, 3, 2],  # Numerical
            'Frequency_Exercise': ['Sometimes', 'Never', 'Often', 'Rarely'],  # Ordinal
            'Moved_By_Language': ['Sometimes', 'Never', 'Sometimes', 'Rarely'],  # Categorical
            'Political_Orientation': [3, 2, 2, 5],  # Numerical (Likert scale)
            'Often_Feel_Awe': [2, 4, 5, np.nan]  # Numerical with a missing value
        })

    def test_handle_missing_values(self):
        """Test that missing values are correctly handled."""
        df_result = handle_missing_values(self.df.copy())

        # Check numerical columns are filled with the mean
        self.assertEqual(df_result['Age'].iloc[2], (40 + 45 + 30) / 3)  # Mean of column 'Age'
        self.assertEqual(df_result['Often_Feel_Awe'].iloc[3], (2 + 4 + 5) / 3)  # Mean of column 'Often_Feel_Awe'

        # Check categorical columns are filled with 'Missing'
        self.assertEqual(df_result['Gender'].iloc[2], 'Female')  # No missing value in Gender here, so no change
        self.assertEqual(df_result['Moved_By_Language'].iloc[3], 'Rarely')  # No missing in Moved_By_Language

    def test_detect_column_types(self):
        """Test automatic column type detection (nominal, ordinal, free text)."""
        column_types = detect_column_types(self.df)

        # Check that the detected columns match expected types
        self.assertIn('Gender', column_types['nominal'])
        self.assertIn('Frequency_Exercise', column_types['ordinal'])
        self.assertIn('Commitment', column_types['free_text'])
        self.assertNotIn('Age', column_types['nominal'])  # Age should not be a nominal category

    def test_detect_outliers(self):
        """Test outlier detection in numerical columns."""
        outliers_count = detect_outliers(self.df, 'Experience_Meditation')
        self.assertEqual(outliers_count, 0)

        outliers_count_political = detect_outliers(self.df, 'Political_Orientation')
        self.assertEqual(outliers_count_political, 0)

    def test_generate_qa_report(self):
        """Test generation of QA report for missing values and outliers."""
        qa_report = generate_qa_report(self.df.copy())

        # Check for correct missing values report
        self.assertIn('Age', qa_report['missing_values'])
        self.assertIn('Often_Feel_Awe', qa_report['missing_values'])

        # Check outliers report
        self.assertNotIn('Experience_Meditation', qa_report['outliers'])
        self.assertNotIn('Political_Orientation', qa_report['outliers'])

        # Check for rows with 3 or more missing values
        self.assertIn('rows_with_3_or_more_missing_values', qa_report)
        self.assertEqual(qa_report['rows_with_3_or_more_missing_values']['count'], 0)

    def test_preprocess_for_output(self):
        """Test the preprocessing of dataframe for statistical output."""
        processed_df = preprocess_for_output(self.df.copy())

        # Check that ordinal and nominal columns are correctly converted
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['Gender']))  # Nominal column
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df['Frequency_Exercise']))  # Ordinal column

        # Check that free text columns are not modified
        self.assertEqual(processed_df['Commitment'].iloc[0], 'I will provide my best answers')

    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_process_data_pipeline(self, mock_to_csv, mock_read_csv, mock_file):
        """Test the full pipeline by mocking file I/O."""
        mock_read_csv.return_value = self.df  # Mock the read_csv function to return the sample DataFrame

        # Set up the mock command-line arguments
        test_args = ['pipeline.py', 'input.csv']
        with patch.object(sys, 'argv', test_args):
            processed_df, qa_report = process_data_pipeline(self.df)

            # Check the processed DataFrame is correct
            self.assertIsNotNone(processed_df)
            self.assertIn('Gender', processed_df.columns)
            self.assertIn('Frequency_Exercise', processed_df.columns)

            # Check that the QA report is not empty
            self.assertTrue(len(qa_report) > 0)


# Run the tests if this script is executed directly
if __name__ == '__main__':
    unittest.main()

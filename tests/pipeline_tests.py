import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import sys
from scripts.pipeline import handle_missing_values, detect_outliers, generate_qa_report, preprocess_for_output, \
    simplify_gender, process_data_pipeline


class TestDataPipeline(unittest.TestCase):

    # setUp method is run before each test to set up a sample DataFrame
    def setUp(self):
        """Set up a sample dataframe similar to the structure provided."""
        self.df = pd.DataFrame({
            'Age': [40, 45, np.nan, 30],  # Numerical with a missing value
            'Gender': ['Female', 'Male', 'Female', 'Female'],  # Categorical
            'Commitment': ['I will provide my best answers'] * 4,  # Categorical
            'Experience_Meditation': [2, 1, 3, 2],  # Numerical
            'Vivid_Image_Face': ['Realistically vivid', 'Moderately vivid', 'Perfectly realistic',
                                 'Perfectly realistic'],  # Categorical
            'Moved_By_Language': ['Sometimes', 'Never', 'Sometimes', 'Rarely'],  # Categorical
            'Political_Orientation': [3, 2, 2, 5],  # Numerical (Likert scale)
            'Often_Feel_Awe': [2, 4, 5, np.nan]  # Numerical with a missing value
        })

    # Test the handle_missing_values function
    def test_handle_missing_values(self):
        """Test that missing values are correctly handled."""
        df_result = handle_missing_values(self.df.copy())

        # Check numerical columns are filled with the mean
        self.assertEqual(df_result['Age'].iloc[2], (40 + 45 + 30) / 3)  # Mean of column 'Age'
        self.assertEqual(df_result['Often_Feel_Awe'].iloc[3], (2 + 4 + 5) / 3)  # Mean of column 'Often_Feel_Awe'

        # Check categorical columns are filled with 'Missing'
        self.assertEqual(df_result['Gender'].iloc[2], 'Female')  # No missing value in Gender here, so no change
        # No categorical columns have missing values in this sample, so no further check here

    # Test the detect_outliers function
    def test_detect_outliers(self):
        """Test outlier detection in numerical columns."""
        # Test with column 'Experience_Meditation', where there should be no outliers
        outliers_count = detect_outliers(self.df, 'Experience_Meditation')
        self.assertEqual(outliers_count, 0)

        # Test with column 'Political_Orientation', no obvious outliers
        outliers_count_political = detect_outliers(self.df, 'Political_Orientation')
        self.assertEqual(outliers_count_political, 0)

    # Test the generate_qa_report function
    def test_generate_qa_report(self):
        """Test generation of QA report for missing values and outliers."""
        qa_report = generate_qa_report(self.df.copy())

        # Check for correct missing values report
        self.assertIn('Age', qa_report['missing_values'])
        self.assertIn('Often_Feel_Awe', qa_report['missing_values'])

        # Check outliers report
        self.assertNotIn('Experience_Meditation', qa_report['outliers'])  # No outliers in this column
        self.assertNotIn('Political_Orientation', qa_report['outliers'])  # No outliers in this column

    # Test the preprocess_for_output function
    def test_preprocess_for_output(self):
        """Test the preprocessing of dataframe for statistical output."""
        # Process the dataframe, but check gender categories before encoding
        processed_df = self.df.copy()

        # Simplify gender categories (before encoding)
        processed_df['Gender'] = processed_df['Gender'].apply(simplify_gender)

        # Check the actual gender categories after simplification
        gender_mapping = processed_df['Gender'].unique()  # Get unique gender categories
        print(
            f"Actual gender categories after simplification: {gender_mapping}")  # This should print the simplified categories

        # Define possible gender categories
        possible_categories = ['Female', 'Male', 'Non-Binary', 'Other']

        # Assert that the actual categories are a subset of the possible categories
        self.assertTrue(set(gender_mapping).issubset(possible_categories))

        # Now run the full preprocessing
        processed_df = preprocess_for_output(processed_df)

        # Check that 'Gender' column is now numeric after encoding
        self.assertTrue(pd.api.types.is_numeric_dtype(processed_df['Gender']))

    # Test the full data pipeline by mocking file I/O
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv')
    def test_process_data_pipeline(self, mock_to_csv, mock_read_csv, mock_file):
        """Test the full pipeline by mocking file I/O."""
        mock_read_csv.return_value = self.df  # Mock the read_csv function to return the sample DataFrame

        # Set up the mock command-line arguments
        test_args = ['data_pipeline.py', 'input.csv', 'output.csv', 'qa_report.txt']
        with patch.object(sys, 'argv', test_args):
            process_data_pipeline('input.csv', 'output.csv', 'qa_report.txt')

            # Check that files were opened and written to
            mock_read_csv.assert_called_once_with('input.csv')
            mock_to_csv.assert_called_once_with('output.csv', index=False)

            # Check that the QA report was written to the correct file
            mock_file.assert_called_with('qa_report.txt', 'w')


# Run the tests if this script is executed directly
if __name__ == '__main__':
    unittest.main()

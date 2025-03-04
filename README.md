# README.md

## Data Pipeline Web Application

## Overview
This project provides a streamlined solution for processing survey datasets using a data pipeline script (`pipeline.py`) integrated with a user-friendly GUI built using Streamlit (`app.py`). The application allows users to upload their CSV survey data, perform data quality assurance, preprocess the data, and download the processed results. It also includes a dedicated feature for handling survey-specific nuances like identifying and aggregating questionnaire scales and conducting sanity checks for certain response patterns.

## Project Structure

- **`pipeline.py`**: Core data pipeline script that handles:
  - Data cleaning, normalization, and preprocessing.
  - Outlier detection and QA reporting.
  - Scoring survey scales with customizable mappings.
  - Data transformation using models (e.g., BERT-based embeddings for text analysis).

- **`app.py`**: Streamlit app that provides an interactive GUI, allowing users to:
  - Upload datasets for analysis.
  - Interact with preprocessing steps, including text responses review and scoring.
  - Download processed data or reports.

- **`scoring_functions.py`**: Contains class and functions for scoring specific survey scales. The main `ScaleScorer` class calculates scores based on user-defined mappings, enabling flexible scale aggregation.

- **`helpers.py`**: Utility functions for data normalization and processing, such as:
  - `normalize_column_name`: Standardizes column names across different datasets.
  - `get_score_from_mapping`: Calculates scale scores based on mapping definitions provided by the user.

### Pipeline Design & Logic

The pipeline is designed to handle various data quality and preprocessing steps in a sequential manner. The main functions of the pipeline are:

1. **`handle_missing_values(df)`**:
    - Handles missing values by imputing numerical columns with the mean and categorical columns with the placeholder 'Missing'.
    - Flags columns with more than 50% missing data and warns the user.

2. **`detect_outliers(df, column_name, threshold=3)`**:
    - Detects outliers in numerical columns using Z-score calculations based on a given threshold.
    - Skips columns that have zero variance to avoid false detection.

3. **`generate_qa_report(df)`**:
    - Generates a comprehensive QA report summarizing:
      - Missing values in each column.
      - Count of outliers in numerical columns.
      - Rows with 3 or more missing values.

4. **`detect_column_types(df)`**:
    - Automatically detects column types (`nominal`, `ordinal`, `free text`, `timestamp`) based on heuristic rules.
    - This function is essential for dynamically determining how to preprocess various column types.
    - Utilizes

5. **`preprocess_for_output(df)`**:
    - Encodes ordinal columns with `OrdinalEncoder` to preserve order.
    - Converts nominal columns to numeric codes.
    - Skips transformation of free text and timestamp columns to retain their original format.

6. **`sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag')`**:
    - Identifies inconsistencies between a binary chills response column and a corresponding intensity column.
    - Provides options to either flag or drop these inconsistent rows.

7. **`process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag')`**:
    - The main pipeline function that integrates all individual processing steps, allowing users to:
      - Handle missing values.
      - Generate a QA report.
      - Perform sanity checks for chills responses.
      - Preprocess the data for statistical analysis.

### Streamlit Application Design

The Streamlit app (`app.py`) provides a step-by-step interface for users to interact with the pipeline. The application is designed with the following key features:

1. **File Upload**:
   - Users can upload their survey dataset as a CSV file.
   
2. **Column Selection and Dropping**:
   - Users can select and drop columns they do not want to include in the analysis using a multi-select dropdown.

3. **Sanity Check Configuration**:
   - Users can specify which columns represent `Chills Response` and `Chills Intensity`.
   - Choose an intensity threshold to flag or drop rows with inconsistencies.
   - Preview and confirm flagged rows before making any changes.

4. **Review and Flagging of Text Responses**:
   - Users can select specific free text columns (e.g., open-ended survey responses) for detailed review.
   - Flag problematic responses along with an option to drop flagged rows before finalizing the processed dataset.

5. **Aggregation of Questionnaire Scales (Future Feature)**:
   - The pipeline is designed to identify and aggregate columns belonging to known questionnaire scales (e.g., MODTAS, VVIQ).
   - Data is ready for use in statistical analysis software such as JAMOVI

6. **Processed Data Preview and Download**:
   - The app displays a preview of the processed data and provides a download option for the encoded dataset and the final cleaned dataset, which includes aggregated behavioral measure scores.

7. **QA Report Download**:
   - Users can view and download a detailed QA report that includes information on missing values, outliers, and any flagged entries.

### How to Run the Application on the Cloud
1. Navigate to the link [here](https://behavioral-survey-processing-app.streamlit.app/)
2. Follow the steps in the app

### How to Run the Application Locally
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/akcone2003/chills_study.git
   cd chills_study
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL (`http://localhost:8501`) in your web browser to interact with the application.

### Key Design Choices and Trade-offs
1. **Handling Missing Values**:
   - Missing values in numerical columns are replaced with the mean to preserve data distribution.
   - Categorical columns are filled with a placeholder (`'Missing'`), ensuring that no rows are dropped during processing.
   
2. **Outlier Detection**:
   - The choice to use Z-score calculations was made to provide a standardized method of outlier detection.
   - This method may not work well with small datasets or datasets with zero variance, so checks were added to handle such cases.

3. **Automated Column Type Detection**:
   - The `detect_column_types` function uses heuristic rules to classify columns, providing flexibility for different datasets.
   - However, this method may require fine-tuning for datasets with unconventional column names or formats.

4. **Sanity Check Implementation**:
   - The sanity check logic focuses on detecting inconsistencies between binary response columns and associated intensity columns.
   - Users can choose to flag or drop rows, providing flexibility in data cleaning.

5. **User-Driven Data Review and Flagging**:
   - The app allows users to manually flag and review rows, ensuring that critical data points are not discarded without user consent.

### Future Enhancements
1. **Automated Scale Aggregation**:
   - Expand the scale detection feature to automatically recognize new questionnaires and allow users to define custom aggregation rules.

2. **Enhanced QA Reporting**:
   - Include visual summaries of the QA report, such as bar charts for missing values or heatmaps for outlier detection.

3. **Customizable Encoding Options**:
   - Provide more options for encoding nominal and ordinal variables based on user-defined rules.

4. **Visualization of Data**
   - Implement dynamic charts and tables, allowing researchers to explore data interactively and visualize trends without needing external tools.

### Adding New Functionalities
If you'd like to add your own functionalities to this web app, you can easily do so by following these steps:

1. **Fork the Repository**: Click the "Fork" button on the GitHub repository to create a copy of the project in your own GitHub account.

2. **Clone Your Fork**:
      ```bash
      git clone <your_forked_repository_url>
      cd chills_study
      ```
3. **Modify the code**: 
* New functionalities can be added by creating additional functions or modules. 
* For instance:
  * Add new data processing utilities in helpers.py or similar modules.
  * Integrate new models or scoring functions in scoring_functions.py.
  
4. **Test Locally**: Run the app locally to make sure it works as expected
   ```bash
   streamlit run app.py                  
   ```                      

5. **Submit a Pull Request**:
* Once you've tested your modifications, commit and push your changes.
* Go to the original repository and submit a pull request. Describe your new functionality in detail and request a review.

By following these steps, you can add new functionalities and improve the app’s features. Contributions are highly valued!
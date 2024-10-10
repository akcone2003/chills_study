import streamlit as st
import pandas as pd
from scripts.pipeline import process_data_pipeline


def save_dataframe_to_csv(df):
    """Convert a DataFrame to CSV format in-memory and return as a string."""
    return df.to_csv(index=False)


def save_text_to_file(text, filename="report.txt"):
    """Save text to a downloadable in-memory file and return it."""
    return text.encode('utf-8')


# Streamlit App Interface
st.title("Data Pipeline Web Application")

# Step 1: File Upload for Input CSV
input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

# Global variable to store flagged rows
flagged_rows = {}

# Step 2: Run the Pipeline if a file is uploaded
if input_file is not None:
    try:
        # Read the uploaded file into a temporary dataframe
        input_df = pd.read_csv(input_file)

        # Step 3: Let the user select columns they want to drop
        st.write("### Select Columns to Drop Before Downloading")
        drop_columns = st.multiselect(
            "Select columns you want to exclude from the analysis:",
            options=input_df.columns.tolist(),
            help="These columns will be removed before you download the processed dataset."
        )

        # If there are columns to drop, apply it to the DataFrame
        if drop_columns:
            input_df = input_df.drop(columns=drop_columns)
            st.success(f"Dropped columns: {drop_columns}")

        # Step 4: Let the user select columns for the sanity check
        st.write("### Sanity Check Configuration")

        chills_column = st.selectbox(
            "Select the column representing Chills Response (0 or 1):",
            options=input_df.columns.tolist(),
            help="This should be a binary column where 0 means no chills and 1 means chills were experienced."
        )

        chills_intensity_column = st.selectbox(
            "Select the column representing Chills Intensity:",
            options=input_df.columns.tolist(),
            help="This column should represent the intensity of chills, with higher values indicating stronger chills."
        )

        intensity_threshold = st.number_input(
            "Enter the intensity threshold to flag or drop inconsistent rows:",
            min_value=0,
            max_value=10,
            value=0,
            help="Rows where the Chills Response is 0 but the intensity is above this value will be flagged or dropped."
        )

        mode = st.radio(
            "Select how to handle inconsistent rows:",
            options=['flag', 'drop'],
            help="'flag' will add a column indicating the inconsistent rows, while 'drop' will remove these rows from the dataset."
        )

        # Step 5: Run the pipeline with the selected configuration and capture the outputs
        if st.button("Run Pipeline"):
            processed_df, intermediate_encoded_df, flagged_rows_df, qa_report = process_data_pipeline(
                input_df,
                chills_column=chills_column if chills_column else None,
                chills_intensity_column=chills_intensity_column if chills_intensity_column else None,
                intensity_threshold=intensity_threshold,
                mode=mode

            )

            st.success("Data pipeline completed successfully!")

            # Step 6: Display the processed DataFrame preview
            st.write("### Processed Data Preview:")
            st.dataframe(processed_df.head())

            # Step 7: Display the intermediate encoded DataFrame preview
            st.write("### Intermediate Encoded Data Preview:")
            st.dataframe(intermediate_encoded_df.head())

            # Step 8: Display flagged rows, if any
            if not flagged_rows_df.empty:
                st.write("### Flagged Rows:")
                st.dataframe(flagged_rows_df)

            # Step 9: Download options
            # 9.1 Download Final Processed CSV
            csv_data = save_dataframe_to_csv(processed_df)
            st.download_button(
                label="Download Final Processed CSV",
                data=csv_data,
                file_name="final_processed_data.csv",
                mime='text/csv'
            )

            # 9.2 Download Intermediate Encoded CSV
            encoded_csv_data = save_dataframe_to_csv(intermediate_encoded_df)
            st.download_button(
                label="Download Intermediate Encoded Data CSV",
                data=encoded_csv_data,
                file_name="intermediate_encoded_data.csv",
                mime='text/csv'
            )

            # 9.3 Display and Download QA Report
            st.write("### Quality Assurance (QA) Report:")
            st.text(qa_report)

            qa_report_file = save_text_to_file(qa_report, filename="QA_Report.txt")
            st.download_button(
                label="Download QA Report",
                data=qa_report_file,
                file_name="QA_Report.txt",
                mime='text/plain'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

import streamlit as st
import pandas as pd
from io import StringIO
from scripts.pipeline import process_data_pipeline


def save_dataframe_to_csv(df):
    """Convert a DataFrame to CSV format in-memory and return as a string."""
    return df.to_csv(index=False)


# Streamlit App Interface
st.title("Data Pipeline Web Application")

# Step 1: File Upload for Input CSV
input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

# Step 2: Run the Pipeline if a file is uploaded
if input_file is not None:
    try:
        # Read the uploaded file into a temporary dataframe
        input_df = pd.read_csv(input_file)

        # Run the pipeline and capture the outputs
        processed_df, qa_report = process_data_pipeline(input_df)

        st.success("Data pipeline completed successfully!")

        # Display the processed DataFrame preview
        st.write("Processed Data Preview:")
        st.dataframe(processed_df.head())

        # Convert processed DataFrame to CSV format for download
        csv_data = save_dataframe_to_csv(processed_df)

        # Download button for the processed CSV
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime='text/csv'
        )

        # Display and Download QA report
        st.write("Quality Assurance Report:")
        st.text(qa_report)

        # Download button for the QA report
        st.download_button(
            label="Download QA Report",
            data=qa_report,
            file_name="qa_report.txt",
            mime='text/plain'
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

import streamlit as st
import pandas as pd
from scripts.pipeline import process_data_pipeline  # Updated import

def save_dataframe_to_csv(df, file_name):
    """Save a DataFrame to a CSV and provide a download link."""
    return df.to_csv(file_name, index=False).encode('utf-8')


# Streamlit App Interface
st.title("Data Pipeline Web Application")

# Step 1: File Upload for Input CSV
input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

# Step 2: Run the Pipeline
if input_file is not None:
    # Save the uploaded file temporarily
    with open("temp_input.csv", "wb") as f:
        f.write(input_file.getbuffer())

    # Run the pipeline and capture the outputs
    try:
        processed_df, qa_report = process_data_pipeline("temp_input.csv")

        st.success("Data pipeline completed successfully!")

        # Display the processed DataFrame
        if processed_df is not None:
            st.write("Processed Data Preview:")
            st.dataframe(processed_df.head())
        else:
            st.warning("Processed data is empty or not generated.")

        # Download processed data if available
        if processed_df is not None:
            st.download_button(
                label="Download Processed CSV",
                data=save_dataframe_to_csv(processed_df, "processed_data.csv"),
                file_name="processed_data.csv",
                mime='text/csv'
            )

        # Display and Download QA report if available
        if qa_report:
            st.write("Quality Assurance Report:")
            st.text(qa_report)

            st.download_button(
                label="Download QA Report",
                data=qa_report.encode('utf-8'),
                file_name="qa_report.txt",
                mime='text/plain'
            )
        else:
            st.warning("QA Report is empty or not generated.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

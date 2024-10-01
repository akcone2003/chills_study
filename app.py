import streamlit as st
from scripts.pipeline import process_data_pipeline  # Use your existing data pipeline functions

# Streamlit App Interface
st.title("Data Pipeline Web Application")

# Step 1: File Upload for Input CSV
input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

# Step 2: File Input for Output CSV and QA Report
output_file = st.text_input("Output CSV File Path (e.g., output.csv)")
qa_report_file = st.text_input("QA Report File Path (e.g., qa_report.txt)")

# Step 3: Run the Pipeline
if st.button("Run Pipeline"):
    if not input_file or not output_file or not qa_report_file:
        st.error("Please provide all file paths!")
    else:
        # Save the uploaded file temporarily
        with open("temp_input.csv", "wb") as f:
            f.write(input_file.getbuffer())

        # Call the data pipeline function
        try:
            process_data_pipeline("temp_input.csv", output_file, qa_report_file)
            st.success("Data pipeline completed successfully!")
            st.write(f"Output saved to: {output_file}")
            st.write(f"QA Report saved to: {qa_report_file}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

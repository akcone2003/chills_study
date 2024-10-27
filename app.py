import streamlit as st
import pandas as pd
from scripts.pipeline import process_data_pipeline


def save_dataframe_to_csv(df):
    """Convert a DataFrame to CSV format in-memory and return as a string."""
    return df.to_csv(index=False)


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

        # Step 3: Let the user select columns they want to drop before processing
        st.write("### Select Columns to Drop Before Processing")
        drop_columns = st.multiselect(
            "Select columns you want to exclude from the analysis:",
            options=input_df.columns.tolist(),
            help="These columns will be removed before running the pipeline."
        )

        # If there are columns to drop, apply it to the DataFrame
        if drop_columns:
            input_df = input_df.drop(columns=drop_columns)
            st.success(f"Dropped columns: {drop_columns}")

        # Step 4: Run the pipeline and capture the outputs
        processed_df, qa_report = process_data_pipeline(input_df)

        st.success("Data pipeline completed successfully!")

        # Step 5: Display the processed DataFrame preview
        st.write("Processed Data Preview:")
        st.dataframe(processed_df.head())

        # Step 6: Let the user select columns for review and flagging
        st.write("### Select Columns for Review and Flagging")

        # Identify all categorical (text) columns
        text_columns = processed_df.select_dtypes(include='object').columns.tolist()

        if text_columns:
            # Display a multi-select widget for the user to choose columns
            selected_columns = st.multiselect(
                "Select text columns to review (choose one or more):",
                options=text_columns,
                default=[]
            )

            # Step 7: If the user selects any columns, display rows for review
            if selected_columns:
                st.write("### Review and Flag Text Responses")
                for col in selected_columns:
                    st.write(f"**Column**: `{col}`")

                    # Create a list to capture flags for each row in the selected column
                    flag_list = []

                    for idx, value in processed_df[col].items():
                        # Create an expander for each row of text response
                        with st.expander(f"Row {idx + 1}: {value[:50]}..."):
                            st.write(f"Full Response: {value}")
                            # Checkbox to flag this row
                            flag = st.checkbox(f"Flag this row in '{col}'", key=f"{col}_{idx}")
                            if flag:
                                reason = st.text_input(f"Reason for flagging row {idx + 1}:", key=f"reason_{col}_{idx}")
                                flag_list.append((idx, reason))

                    # Store the flags for this column if any rows are flagged
                    if flag_list:
                        flagged_rows[col] = flag_list

        # Step 8: Download processed data
        csv_data = save_dataframe_to_csv(processed_df)
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime='text/csv'
        )

        # Step 9: Modify QA report to include flagged information
        if flagged_rows:
            flagged_info = "Flagged Rows Information:\n\n"
            for col, flags in flagged_rows.items():
                flagged_info += f"Column: {col}\n"
                for idx, reason in flags:
                    flagged_info += f" - Row {idx + 1}: {reason}\n"
            qa_report += "\n\n" + flagged_info

        # Display the QA report
        st.write("Quality Assurance Report:")
        st.text(qa_report)

        # Step 10: Download button for the QA report
        st.download_button(
            label="Download QA Report",
            data=qa_report,
            file_name="qa_report.txt",
            mime='text/plain'
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

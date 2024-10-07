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

        # Step 5: Preview Inconsistent Rows Before Making Changes
        preview_flag = False
        if st.button("Preview Inconsistent Rows"):
            # Find rows where the Chills Response is 0 but Intensity > threshold
            inconsistent_rows = input_df[
                (input_df[chills_column] == 0) & (input_df[chills_intensity_column] > intensity_threshold)
            ]
            st.write("### Inconsistent Rows Preview")
            st.write(inconsistent_rows)
            preview_flag = True

        # Confirm row drops if 'drop' mode is selected
        if mode == 'drop' and preview_flag:
            confirm_drop = st.checkbox("I confirm I want to drop these rows before finalizing.")
        else:
            confirm_drop = True  # No confirmation needed if just flagging

        if preview_flag and not confirm_drop:
            st.warning("Please confirm you want to drop the rows or switch to 'flag' mode to proceed.")

        # Step 6: Run the pipeline with the selected configuration and capture the outputs
        if confirm_drop:
            processed_df, qa_report = process_data_pipeline(
                input_df,
                chills_column=chills_column,
                chills_intensity_column=chills_intensity_column,
                intensity_threshold=intensity_threshold,
                mode=mode
            )

            st.success("Data pipeline completed successfully!")

            # Step 7: Display the processed DataFrame preview
            st.write("Processed Data Preview:")
            st.dataframe(processed_df.head())

            # Step 8: Let the user select columns for review and flagging
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

                # Step 9: If the user selects any columns, display rows for review
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

            # Step 10: Add an option to remove flagged rows
            if flagged_rows:
                if st.checkbox("Drop all flagged rows before download?"):
                    flagged_indices = [idx for col_flags in flagged_rows.values() for idx, _ in col_flags]
                    processed_df = processed_df.drop(flagged_indices)
                    st.success("Flagged rows have been dropped from the final dataset.")

            # Step 11: Download processed data
            csv_data = save_dataframe_to_csv(processed_df)
            st.download_button(
                label="Download Processed CSV",
                data=csv_data,
                file_name="processed_data.csv",
                mime='text/csv'
            )

            # Step 12: Modify QA report to include flagged information
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

            # Step 13: Download button for the QA report
            st.download_button(
                label="Download QA Report",
                data=qa_report,
                file_name="qa_report.txt",
                mime='text/plain'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

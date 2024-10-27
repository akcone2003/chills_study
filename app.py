import streamlit as st
import pandas as pd

from scripts.helpers import normalize_column_input
from scripts.pipeline import process_data_pipeline


def save_dataframe_to_csv(df):
    """Convert a DataFrame to CSV format in-memory and return as a string."""
    return df.to_csv(index=False)

def rebuild_qa_report():
    """Rebuild the QA report with the current flagged rows."""
    qa_report = "Quality Assurance Report\n\n"
    qa_report += f"Missing Values: {st.session_state.get('missing_values', {})}\n\n"
    qa_report += f"Outliers: {st.session_state.get('outliers', {})}\n\n"

    flagged_info = "Flagged Rows Information:\n\n"
    for col, flags in st.session_state.flagged_rows.items():
        flagged_info += f"Column: {col}\n"
        for idx, reason in flags:
            flagged_info += f" - Row {idx + 1}: {reason if reason else 'No reason provided'}\n"

    qa_report += flagged_info
    st.session_state.qa_report = qa_report  # Rebuild the QA report from scratch


# Streamlit App Interface
st.title("Data Pipeline Web Application")

# Initialize session state variables if they don't exist
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'intermediate_df' not in st.session_state:
    st.session_state.intermediate_df = None
if 'qa_report' not in st.session_state:
    st.session_state.qa_report = "Quality Assurance Report\n\n"
if 'qa_report_flags' not in st.session_state:
    st.session_state.qa_report_flags = {}  # Track rows already added to the QA report
if 'flagged_rows' not in st.session_state:
    st.session_state.flagged_rows = {}
if 'user_column_mappings' not in st.session_state:
    st.session_state.user_column_mappings = {}
# Initialize the `sanity_check_drops` set if it doesn't exist in session state
if 'sanity_check_drops' not in st.session_state:
    st.session_state.sanity_check_drops = set()


# Step 1: File Upload for Input CSV
input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

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

        # Step 3.1: Let the user select columns for each scale using a multiselect
        st.write("### Map Columns to Scale Questions")

        available_scales = ["MODTAS", "TIPI", "VVIQ", "KAMF", "DPES-Awe", "MAIA",
                            "Ego-Dissolution", "SMES", "Emotional Breakthrough"]  # TODO - Extend this list as more scales are added

        # User selects the scales they want to include in the analysis
        selected_scales = st.multiselect(
            "Select scales to include in the analysis:",
            options=available_scales,
            default=[]
        )

        # Initialize the user_column_mappings dictionary
        user_column_mappings = {}

        # Loop over each selected scale
        for scale in selected_scales:
            st.write(f"### Select Columns for {scale}")

            # Option 1: Text area for pasting column names (newline-separated)
            pasted_columns = st.text_area(
                f"Paste the columns for {scale}:",
                placeholder="Paste column names here, "
                            "separated by new lines (pressing 'enter' at the beginning of each new question)...",
                key=f"{scale}_paste"
            )

            # Option 2: Optional multiselect for additional manual selection (if needed)
            selected_columns = st.multiselect(
                f"Select additional columns for {scale} (optional):",
                options=input_df.columns.tolist(),
                key=f"{scale}_select"
            )

            # Normalize and clean the pasted input
            pasted_list = normalize_column_input(pasted_columns) if pasted_columns.strip() else []

            # Show the detected column count to the user
            st.write(f"**Detected {len(pasted_list)} columns** from pasted input.")

            # Combine pasted columns and manually selected columns, ensuring uniqueness
            all_selected_columns = list(set(pasted_list + selected_columns))

            if all_selected_columns:
                user_column_mappings[scale] = {
                    f"Question {i + 1}": col for i, col in enumerate(all_selected_columns)
                }

        # Store the column mappings in session state
        st.session_state.user_column_mappings = user_column_mappings

        # Step 4: Let the user select columns for the sanity check
        st.write("### Sanity Check Configuration")

        chills_column = st.selectbox(
            "Select the column representing Chills Response (0 or 1):",
            options=[None] + input_df.columns.tolist(),
            format_func=lambda x: "" if x is None else x,
            help="This should be a binary column where 0 means no chills and 1 means chills were experienced."
        )

        chills_intensity_column = st.selectbox(
            "Select the column representing Chills Intensity:",
            options=[None] + input_df.columns.tolist(),
            format_func=lambda x: "" if x is None else x,
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

        # Step 6: Run the pipeline with the selected configuration and capture the outputs
        if st.button("Run Pipeline"):
            st.session_state.processed_df, st.session_state.intermediate_df, st.session_state.qa_report = process_data_pipeline(
                input_df,
                chills_column=chills_column if chills_column else None,
                chills_intensity_column=chills_intensity_column if chills_intensity_column else None,
                intensity_threshold=intensity_threshold,
                mode=mode,
                user_column_mappings=st.session_state.user_column_mappings
            )

            st.success("Data pipeline completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# If the pipeline has been run, display the outputs
if st.session_state.processed_df is not None:
    processed_df = st.session_state.processed_df
    intermediate_df = st.session_state.intermediate_df
    qa_report = st.session_state.qa_report

    # Step 7: Display the processed DataFrame preview
    st.write("Processed Data Preview:")
    st.dataframe(processed_df.head())

    # Step 8: Mid-processing Download - Encoded Values
    st.write("### Download Encoded Dataset (Mid-Processing)")
    encoded_csv = save_dataframe_to_csv(intermediate_df)
    st.download_button(
        label="Download Encoded CSV",
        data=encoded_csv,
        file_name="encoded_dataset.csv",
        mime='text/csv'
    )

    # Step 9: Display flagged rows for individual review and dropping
    if "Sanity_Flag" in processed_df.columns:
        flagged_rows = processed_df[processed_df['Sanity_Flag'] == True]
        st.write("### Sanity Check - Review Flagged Rows")
        st.write("Below are the rows flagged for sanity check inconsistencies:")

        # Display each flagged row with a checkbox for user to drop it
        for idx, row in flagged_rows.iterrows():
            with st.expander(f"Row {idx}"):
                st.write(row)
                if st.checkbox(f"Drop row {idx}?", key=f"sanity_drop_{idx}"):
                    st.session_state.sanity_check_drops.add(idx)

        # Step 10: Apply individual row drops if any
    if st.session_state.sanity_check_drops:
        st.write(f"Rows marked for removal: {st.session_state.sanity_check_drops}")
        if st.button("Remove Selected Rows"):
            processed_df = processed_df.drop(st.session_state.sanity_check_drops, errors='ignore')
            st.success(f"Dropped rows: {st.session_state.sanity_check_drops}")
            st.session_state.sanity_check_drops.clear()  # Reset the drop list
            st.session_state.processed_df = processed_df  # Update the session state

    # Step 11: Let the user select columns for review and flagging
    st.write("### Select Text Columns for Review and Flagging")
    text_columns = processed_df.select_dtypes(include='object').columns.tolist()

    if text_columns:
        selected_columns = st.multiselect(
            "Select text columns to review (choose one or more):",
            options=text_columns,
            default=[]
        )

        if selected_columns:
            st.write("### Review and Flag Text Responses")
            for col in selected_columns:
                st.write(f"**Column**: `{col}`")

                flag_list = []
                for idx, value in processed_df[col].items():
                    with st.expander(f"Row {idx + 1}: {value[:50]}..."):
                        st.write(f"Full Response: {value}")
                        flag = st.checkbox(f"Flag this row in '{col}'", key=f"{col}_{idx}")
                        if flag:
                            reason = st.text_input(f"Reason for flagging row {idx + 1}:", key=f"reason_{col}_{idx}")
                            flag_list.append((idx, reason))

                if flag_list:
                    st.session_state.flagged_rows[col] = flag_list

    # Step 12: Add an option to remove flagged rows
    if st.session_state.flagged_rows:
        if st.checkbox("Drop all flagged rows before download?"):
            flagged_indices = [idx for col_flags in st.session_state.flagged_rows.values() for idx, _ in col_flags]
            processed_df = processed_df.drop(flagged_indices)
            st.success("Flagged rows have been dropped from the final dataset.")

    # Step 14: Update the QA report with flagged rows only if new flags are detected
    rebuild_qa_report()

    # Step 15: Final CSV download - With Only Behavioral Scores
    st.write("### Download Final Processed Dataset (With Scores Only)")
    csv_data = save_dataframe_to_csv(processed_df)
    st.download_button(
        label="Download Final Processed CSV",
        data=csv_data,
        file_name="final_processed_data.csv",
        mime='text/csv'
    )

    # Step 16: Display the QA report
    st.write("Quality Assurance Report:")
    st.text(st.session_state.qa_report)

    # Step 17: Download button for the QA report
    st.download_button(
        label="Download QA Report",
        data=st.session_state.qa_report,
        file_name="qa_report.txt",
        mime='text/plain'
    )


# TODO - add a GenAI way of seeing the column names and then organizing those into the scales and call the corresponding functions
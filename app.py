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

        # Step 3.1: Let the user select columns for each scale using a multiselect
        st.write("### Map Columns to Scale Questions")

        # Initialize the mappings dictionary
        user_column_mappings = {}

        # Get the list of scales to be used
        available_scales = ["MODTAS"]  # Extend this list as more scales are added

        # User selects the scales they want to include in the analysis
        selected_scales = st.multiselect(
            "Select scales to include in the analysis:",
            options=available_scales,
            default=[]
        )

        # For each selected scale, let the user select the columns for that scale
        for scale in selected_scales:
            st.write(f"### Select Columns for {scale}")

            # Multiselect to let the user select all the relevant columns for this scale at once
            selected_columns = st.multiselect(
                f"Select the columns that correspond to the questions for {scale}:",
                options=input_df.columns.tolist(),
                help=f"Select columns from your dataset that match the {scale} questions.",
                key=f"{scale}_columns"
            )

            # Store the mapping only if valid columns are selected
            if selected_columns:
                # Simply store the scale and the selected columns
                user_column_mappings[scale] = {f"Question {i + 1}": col for i, col in enumerate(selected_columns)}

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
            processed_df, qa_report = process_data_pipeline(
                input_df,
                chills_column=chills_column if chills_column else None,
                chills_intensity_column=chills_intensity_column if chills_intensity_column else None,
                intensity_threshold=intensity_threshold,
                mode=mode,
                user_column_mappings=user_column_mappings  # Pass user-selected column mappings here
            )

            st.success("Data pipeline completed successfully!")

            # Step 7: Display the processed DataFrame preview
            st.write("Processed Data Preview:")
            st.dataframe(processed_df.head())

            # Step 11: Download processed data
            csv_data = save_dataframe_to_csv(processed_df)
            st.download_button(
                label="Download Processed CSV",
                data=csv_data,
                file_name="processed_data.csv",
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

import streamlit as st
import pandas as pd
from collections import OrderedDict
from scripts.utils import normalize_column_input, handle_checkbox_change
from scripts.pipeline import process_data_pipeline
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def save_dataframe_to_csv(df):
    return df.to_csv(index=False)


def rebuild_qa_report():
    """
    Rebuilds the QA report with enhanced information about dropped rows and data quality.
    Now includes detailed information about rows dropped due to chills response inconsistencies.
    """
    report_parts = ["Quality Assurance Report\n\n"]

    # Add missing values section
    report_parts.append(f"Missing Values: {st.session_state.get('missing_values', {})}\n\n")

    # Add dropped rows section
    if st.session_state.sanity_check_drops:
        report_parts.append("Dropped Rows from Chills Sanity Check:\n")
        report_parts.append(f"Total rows dropped: {len(st.session_state.sanity_check_drops)}\n")
        report_parts.append("Row indices dropped: " +
                            ", ".join(str(idx) for idx in sorted(st.session_state.sanity_check_drops)) + "\n\n")

    # Add flagged rows section
    report_parts.append("Flagged Rows Information:\n")
    if 'flagged_rows' in st.session_state and st.session_state.flagged_rows:
        for col, flags in st.session_state.flagged_rows.items():
            report_parts.append(f"Column: {col}\n")
            for idx, reason in flags:
                report_parts.append(f" - Row {idx + 1}: {reason if reason else 'No reason provided'}\n")

    st.session_state.qa_report = "".join(report_parts)


def initialize_session_state():
    session_vars = {
        'processed_df': None,
        'intermediate_df': None,
        'qa_report': "Quality Assurance Report\n\n",
        'qa_report_flags': {},
        'flagged_rows': {},
        'user_column_mappings': {},
        'sanity_check_drops': set()
    }

    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default


initialize_session_state()

st.title("Data Pipeline Web Application")

input_file = st.file_uploader("Upload your Input CSV File", type=["csv"])

if input_file is not None:
    try:
        with st.spinner("Reading uploaded file..."):
            input_df = pd.read_csv(input_file)

        st.write("### Select Columns to Drop Before Downloading")
        drop_columns = st.multiselect(
            "Select columns to exclude:",
            options=input_df.columns.tolist(),
            help="These columns will be removed before processing."
        )

        if drop_columns:
            input_df = input_df.drop(columns=drop_columns)
            st.success(f"Dropped {len(drop_columns)} columns")

        st.write("### Map Columns to Scale Questions")

        available_scales = ["MODTAS", "TIPI", "VVIQ", "KAMF", "MAIA", "MAIA-S",
                            "Ego-Dissolution", "SMES",
                            "Emotional_Breakthrough", "WCS", "Religiosity", "NEO-FFI-3_Five_Factor_Inventory",
                            "Psychological_Insight", "DPES-Joy", "DPES-Love", "DPES-Pride", "DPES-Awe",
                            "DPES-Amusement", "DPES-Compassion",
                            "MAAS", "Five_Facet_Mindfulness_Questionnaire_(FFMQ)",
                            "Positive_Negative_Affect_Schedule_(PANAS)",
                            "PANAS_X", "Self-Transcendence_Scale",
                            "Early_Maladaptive_Schema_(EMS)_Young_Schema_Questionnaire_Short_Form_3_(YSQ-S3)",
                            "Multidimensional_Iowa_Suggestibility_Scale_(MISS)", "Short_Suggestibility_Scale_(SSS)",
                            "Cloninger_Self_Transcendence_Subscale",
                            "Warwick-Edinburgh_Mental_Wellbeing_Scale_(WEMWBS)",
                            "Cognitive_and_Affective_Mindfulness_Scale_Revised_(CAMS-R)", "Toronto_Mindfulness_Scale",
                            "Copenhagen_Burnout_Inventory_(CBI)", "NEO-PI-3_(Openness_to_Experience)",
                            "Dispositional_Resilience_\'Hardiness\'_Scale_(HARDY)",
                            "Montgomery-Åsberg_Depression_Rating_Scale_(MADRS)",
                            "Hamilton_Anxiety_Rating_Scale_(HAM-A)", "State-Trait_Anxiety_Inventory_(STAI-State_Form)",
                            "5-Dimensional_Altered States_of_Consciousness_Questionnaire_(5DASC)",
                            "Anxiety_Sensitivity_Index-3_(ASI-3_ASI-R)",
                            "Karolinska_Sleepiness_Scale_(KSS)", "Wong-Baker_Pain_Scale",
                            "Overall_Anxiety_Severity_and_Impairment_Scale_(OASIS)",
                            "PHQ-9", "Sheehan_Disability_Scale_(SDS)", "Brief_Symptom_Inventory-18_(BSI-18)", 
                            "Dispositional_Hope_Scale_(DHS)", "General_Self-Efficacy_Scale_(GSES)", "Subjective_Vitality_Scale_(SVS)",
                            "Flow_State_Scale_(FFS)_(short-version)", "Purpose_In_Life_(PSS)", "Perceived_Stress_Scale_(PSS)",
                            "Multidimensional_Health_Locus_of_Control_(MHLC)", "Connor-Davidson_Resilience_Scale_(CD-RISC-10)",
                            "Profile_of_Mood_States_(POMS)"]

        selected_scales = st.multiselect(
            "Select scales to analyze:",
            options=available_scales,
            default=[]
        )

        user_column_mappings = {}
        if selected_scales:
            progress_bar = st.progress(0)
            for i, scale in enumerate(selected_scales):
                with st.expander(f"Configure {scale}"):
                    pasted_columns = st.text_area(
                        f"Paste columns for {scale}:",
                        placeholder="Paste column names here",
                        key=f"{scale}_paste"
                    )

                    selected_columns = st.multiselect(
                        "Additional columns (optional):",
                        options=input_df.columns.tolist(),
                        key=f"{scale}_select"
                    )

                    pasted_list = normalize_column_input(pasted_columns) if pasted_columns.strip() else []
                    st.write(f"**Found {len(pasted_list)} columns** from input.")

                    all_columns = list(dict.fromkeys(pasted_list + selected_columns))

                    if all_columns:
                        user_column_mappings[scale] = OrderedDict({
                            f"Question {i + 1}": col for i, col in enumerate(all_columns)
                        })

                progress_bar.progress((i + 1) / len(selected_scales))

        st.session_state.user_column_mappings = user_column_mappings

        include_chills = st.radio(
            "Did your study include chills?",
            options=["No", "Yes"],
            index=0,
            help="Select 'Yes' if your study recorded data about chills."
        )

        chills_column = None
        chills_intensity_column = None
        intensity_threshold = 0
        mode = 'flag'

        if include_chills == "Yes":
            st.write("### Chills Sanity Check Configuration")

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
                help="This column should represent the intensity of chills."
            )

            intensity_threshold = st.number_input(
                "Enter the intensity threshold:",
                min_value=0,
                max_value=10,
                value=0,
                help="Threshold for flagging inconsistent rows. This value is inclusive, so anything greater than or equal will be flagged."
            )

            mode = st.radio(
                "Select how to handle inconsistent rows:",
                options=['flag', 'drop'],
                help="'flag' will add a column indicating the inconsistent rows, while 'drop' will remove these rows."
            )

        if st.button("Run Pipeline"):
            try:
                start_time = time.time()
                with st.spinner("Processing data..."):
                    st.session_state.processed_df, st.session_state.intermediate_df, st.session_state.qa_report = (
                        process_data_pipeline(
                            input_df,
                            chills_column=chills_column,
                            chills_intensity_column=chills_intensity_column,
                            intensity_threshold=intensity_threshold,
                            mode=mode,
                            user_column_mappings=st.session_state.user_column_mappings
                        )
                    )

                processing_time = time.time() - start_time
                st.success(f"Processing completed in {processing_time:.2f} seconds!")

            except Exception as e:
                logger.error(f"Pipeline error: {str(e)}")
                st.error(f"Processing error: {str(e)}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if st.session_state.processed_df is not None:
    processed_df = st.session_state.processed_df
    intermediate_df = st.session_state.intermediate_df
    qa_report = st.session_state.qa_report

    st.write("Processed Data Preview:")
    st.dataframe(processed_df.head())

    st.write("### Download Encoded Dataset (Mid-Processing)")
    encoded_csv = save_dataframe_to_csv(intermediate_df)
    st.download_button(
        label="Download Encoded CSV",
        data=encoded_csv,
        file_name="encoded_dataset.csv",
        mime='text/csv'
    )

    if "Sanity_Flag" in processed_df.columns:
        # Filter columns that contain the word "chills" (case-insensitive)
        chills_columns = [col for col in processed_df.columns if 'chills' in col.lower()]

        if chills_columns:
            flagged_rows = processed_df[processed_df['Sanity_Flag'] == True]
            st.write("### Chills Response Sanity Check")
            st.write("Below are the rows where chills responses may be inconsistent. "
                     "Double click on the cell containing the response to expand it for viewing.")

            # Initialize checkbox states in session state if they don't exist
            if 'checkbox_states' not in st.session_state:
                st.session_state.checkbox_states = {}

            for idx, row in flagged_rows.iterrows():
                with st.expander(f"Row {idx}"):
                    # Create a subset of the row with only chills-related columns
                    chills_data = row[chills_columns + ['Sanity_Flag']]
                    st.write(chills_data)

                    # Initialize checkbox state for this row if not already present
                    if f"sanity_drop_{idx}" not in st.session_state.checkbox_states:
                        st.session_state.checkbox_states[
                            f"sanity_drop_{idx}"] = idx in st.session_state.sanity_check_drops

                    # Create the checkbox with a callback function
                    current_state = st.session_state.checkbox_states[f"sanity_drop_{idx}"]
                    if st.checkbox(
                            f"Drop row {idx}?",
                            value=current_state,
                            key=f"sanity_drop_{idx}",
                            on_change=lambda: handle_checkbox_change(idx)
                    ):
                        if idx not in st.session_state.sanity_check_drops:
                            st.session_state.sanity_check_drops.add(idx)
                    else:
                        st.session_state.sanity_check_drops.discard(idx)

                    # Update the state tracker
                    st.session_state.checkbox_states[f"sanity_drop_{idx}"] = idx in st.session_state.sanity_check_drops

                    # Rebuild QA report whenever the checkbox state changes
                    rebuild_qa_report()

        # Display summary of marked rows
        if st.session_state.sanity_check_drops:
            st.write(f"Rows marked for removal: {sorted(st.session_state.sanity_check_drops)}")
            if st.button("Remove Selected Rows"):
                processed_df = processed_df.drop(st.session_state.sanity_check_drops, errors='ignore')
                st.success(
                    f"Dropped {len(st.session_state.sanity_check_drops)} rows with inconsistent chills responses")
                st.session_state.checkbox_states.clear()  # Clear checkbox states when rows are dropped
                st.session_state.sanity_check_drops.clear()
                st.session_state.processed_df = processed_df

    st.write("### Select Text Columns for Review and Flagging")
    text_columns = processed_df.select_dtypes(include='object').columns.tolist()

    if text_columns:
        selected_columns = st.multiselect(
            "Select text columns to review:",
            options=text_columns,
            default=[]
        )

        if selected_columns:
            st.write("### Review and Flag Text Responses")
            if 'flagged_rows' not in st.session_state:
                st.session_state.flagged_rows = {}

            for col in selected_columns:
                st.write(f"**Column**: `{col}`")
                flag_list = []

                column_data = processed_df[col].fillna("No Value")
                safe_col = ''.join(e for e in col if e.isalnum())

                for idx in range(len(column_data)):
                    value = column_data.iloc[idx]
                    display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)

                    with st.expander(f"Row {idx + 1}: {display_value}"):
                        st.write(f"Full Response: {value}")
                        if st.checkbox("Flag this row", key=f"flag_{safe_col}_{idx}"):
                            reason = st.text_input("Reason for flagging:", key=f"reason_{safe_col}_{idx}")
                            if reason:
                                flag_list.append((idx, reason))

                if flag_list:
                    st.session_state.flagged_rows[col] = flag_list

    if st.session_state.flagged_rows:
        if st.checkbox("Drop all flagged rows before download?"):
            flagged_indices = [idx for col_flags in st.session_state.flagged_rows.values() for idx, _ in col_flags]
            processed_df = processed_df.drop(flagged_indices)
            st.success("Flagged rows have been dropped from the final dataset.")

    rebuild_qa_report()

    st.write("### Download Final Processed Dataset (With Scores Only)")
    csv_data = save_dataframe_to_csv(processed_df)
    st.download_button(
        label="Download Final Processed CSV",
        data=csv_data,
        file_name="final_processed_data.csv",
        mime='text/csv'
    )

    st.write("Quality Assurance Report:")
    st.text(st.session_state.qa_report)

    st.download_button(
        label="Download QA Report",
        data=st.session_state.qa_report,
        file_name="qa_report.txt",
        mime='text/plain'
    )

    st.write("## Suggest Improvements")
    st.write(
        "We'd love to hear your feedback to help us improve this application. "
        "Please click the link below to submit your suggestions via Google Form."
    )

    google_form_url = "https://forms.gle/LeXMZkkhJ5D3gdLw8"
    st.markdown(f"[Submit Your Suggestion]({google_form_url})", unsafe_allow_html=True)
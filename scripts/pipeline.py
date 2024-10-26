import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from scripts.scoring_functions import ScaleScorer
from scripts.helpers import normalize_column_name
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def get_embedding(text):
    """Get the embedding of a given text using BERT."""
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**tokens)
    # Use the [CLS] token as the sentence embedding
    embedding = output.last_hidden_state[:, 0, :].numpy().flatten()
    return embedding


# Define the ordered categories for known ordinal scales
PREDEFINED_ORDINAL_CATEGORIES = {
    'HARDY': ['Not true at all', 'A little true', 'Quite true', 'Completely true'],
    'CD-RISC': ['Not True at All', 'True Nearly All the Time'],
    'BRS': ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],
    'ARS-30': ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    'STAI': ['Not at all', 'Somewhat', 'Moderately so', 'Very much so'],
    'ASI': ['Very little', 'A little', 'Some', 'Much', 'Very much'],
    'Cloninger': ['Definitely False', 'Definitely True']
}


def handle_missing_values(df):
    """
    Handle missing values by filling numerical columns with the mean
    and categorical columns with 'Missing'.
    """
    num_cols = df.select_dtypes(include=np.number)
    df[num_cols.columns] = df[num_cols.columns].fillna(num_cols.mean())

    cat_cols = df.select_dtypes(include='object')
    df[cat_cols.columns] = df[cat_cols.columns].fillna('Missing')
    return df


def detect_column_types(df):
    """
    Detect column types as nominal, ordinal, or free text.
    """
    column_types = {
        'nominal': [],
        'ordinal': [],
        'free_text': [],
    }

    cat_cols = df.select_dtypes(include='object')

    # Heuristic to classify free text or categorical columns
    for col in cat_cols.columns:
        unique_values = df[col].nunique()
        total_rows = len(df)

        if unique_values / total_rows > 0.3:
            column_types['free_text'].append(col)
        else:
            # Determine if a column is ordinal using ordered keywords or known categories
            if col in PREDEFINED_ORDINAL_CATEGORIES:
                column_types['ordinal'].append(col)
            else:
                ordered_keywords = ['never', 'rarely', 'sometimes', 'often', 'always',
                                    'strongly disagree', 'disagree', 'disagree somewhat',
                                    'neither agree nor disagree', 'agree somewhat',
                                    'agree', 'strongly agree', 'not at all', 'a little',
                                    'moderately', 'quite a bit', 'extremely', 'somewhat', 'neutral']

                if any(keyword in [str(val).lower() for val in df[col].unique()] for keyword in ordered_keywords):
                    column_types['ordinal'].append(col)
                else:
                    column_types['nominal'].append(col)

    return column_types


def determine_category_order(col_values):
    """
    Determine the correct order of categories for a given column dynamically.
    This ensures that even for new scales, the order is consistent and logical.
    """

    # Define multiple ordered keyword lists for different types of scales
    ordered_keywords_sets = {
        'frequency': ['never', 'rarely', 'sometimes', 'often', 'always'],
        'recency': ['within the last year', 'within the last month', 'within the last 24 hours'],
        'frequency_01': ['never', 'less than once a month', 'once a month',
                         '2-3 times a month', 'once a week', '2-3 times a week',
                         'about once a day', 'two or more times per day'],
        'frequency_02': ['never', 'rarely', 'occasionally', 'often', 'very often'],
        'agreement': ['strongly disagree', 'disagree', 'neither agree nor disagree', 'agree', 'strongly agree'],
        'agreement_1': ['strongly disagree', 'disagree', 'somewhat disagree', 'neutral', 'somewhat agree', 'agree',
                        'strongly agree'],
        'intensity': ['not at all', 'a little', 'moderately', 'quite a bit', 'extremely'],
        'positivity': ['poor', 'fair', 'good', 'very good', 'excellent']
    }

    # Convert column values to lowercase for easier comparison
    lower_col_values = [str(value).lower() for value in col_values]

    # Attempt to match column values to one of the ordered keyword sets
    best_match = None
    best_match_count = 0

    for scale_name, keywords in ordered_keywords_sets.items():
        # Count how many values from the column match the current keyword list
        match_count = sum(1 for value in lower_col_values if value in keywords)

        # Keep track of the best match (most matches with the keywords)
        if match_count > best_match_count:
            best_match = keywords
            best_match_count = match_count

    # If a matching scale is found, use it to sort the categories
    if best_match:
        sorted_categories = sorted(col_values,
                                   key=lambda x: best_match.index(x.lower()) if x.lower() in best_match else float(
                                       'inf'))
    else:
        # Fallback to semantic similarity-based sorting
        # Generate embeddings for each category
        embeddings = {value: get_embedding(value) for value in col_values}

        # Define a reference point for ordering, like "least" or "most"
        reference_min = get_embedding("least")
        reference_max = get_embedding("most")

        # Calculate similarity to reference points
        scores = {}
        for value, embedding in embeddings.items():
            # Similarity to "least" reference
            similarity_to_min = cosine_similarity([embedding], [reference_min])[0][0]
            # Similarity to "most" reference
            similarity_to_max = cosine_similarity([embedding], [reference_max])[0][0]

            # Calculate a combined score (the difference helps capture ordering)
            scores[value] = similarity_to_max - similarity_to_min

        # Sort categories based on their calculated score
        sorted_categories = sorted(col_values, key=lambda x: scores[x])

    return sorted_categories


def encode_columns_for_jamovi(df, column_types):
    """
    Encode ordinal columns with OrdinalEncoder and nominal columns with codes,
    keeping the data interpretable for Jamovi.
    """
    # Step 1: Handle ordinal columns (using predefined or dynamically detected categories)
    for col in column_types['ordinal']:
        print(f"[DEBUG] Encoding ordinal column: {col}")
        try:
            if col in PREDEFINED_ORDINAL_CATEGORIES:
                # Use predefined categories for known ordinal columns
                categories = [PREDEFINED_ORDINAL_CATEGORIES[col]]
            else:
                # Dynamically determine categories for unseen ordinal columns
                unique_values = df[col].dropna().unique()
                categories = [determine_category_order(unique_values)]

            encoder = OrdinalEncoder(categories=categories)
            df[col] = encoder.fit_transform(df[[col]]) + 1  # +1 to avoid 0-based indexing
        except Exception as e:
            print(f"[ERROR] Ordinal encoding failed for '{col}': {e}")

    # Step 2: Handle nominal columns (label encoding to keep single column structure)
    for col in column_types['nominal']:
        if df[col].nunique() == 2:  # Handle binary columns explicitly
            df[col] = df[col].map({'No': 0, 'Yes': 1})  # Adjust based on actual values
        else:
            # Use LabelEncoder for other nominal columns
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convert to string to handle non-string categories
        print(f"[DEBUG] Encoding nominal column: {col}")

    return df


def sanity_check_chills(df, chills_column, chills_intensity_column, threshold=0):
    """
    Sanity check to flag inconsistencies between chills columns.
    """
    inconsistent_rows = (df[chills_column] == 0) & (df[chills_intensity_column] > threshold)
    df['Sanity_Flag'] = inconsistent_rows.astype(int)
    return df


def preprocess_data(df):
    """
    Preprocess the DataFrame for Jamovi analysis with proper encodings.
    """
    # Normalize column names right at the beginning
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Detect column types
    column_types = detect_column_types(df)

    # Encode columns
    df = encode_columns_for_jamovi(df, column_types)

    # Ensure numeric columns are consistent in type (float64)
    df = df.astype({col: 'float64' for col in df.select_dtypes(include=[np.int64, np.float64]).columns})
    return df


def generate_qa_report(df):
    """
    Generate a QA report identifying missing values and outliers.
    """
    report = {'missing_values': df.isnull().sum().to_dict()}

    outliers_report = {}
    num_cols = df.select_dtypes(include=np.number)
    for col in num_cols:
        outliers_count = detect_outliers(df, col)
        if outliers_count > 0:
            outliers_report[col] = outliers_count
    report['outliers'] = outliers_report

    rows_with_many_missing = df[df.isnull().sum(axis=1) >= 3]
    report['rows_with_3_or_more_missing_values'] = {
        'count': len(rows_with_many_missing),
        'row_indices': rows_with_many_missing.index.tolist()
    }

    return report


def detect_outliers(df, column_name, threshold=3):
    """
    Detect outliers in a numerical column using Z-scores.
    """
    col_data = df[[column_name]].dropna()
    if col_data.std().iloc[0] == 0:
        return 0

    scaler = StandardScaler()
    z_scores = scaler.fit_transform(col_data)
    return (np.abs(z_scores) > threshold).sum()


# Full pipeline
def process_data_pipeline(input_df, chills_column, chills_intensity_column, intensity_threshold=0, mode='flag',
                          user_column_mappings=None):
    """
    Main pipeline function that handles QA, sanity checks, encoding, and scoring.
    """
    # Step 1: Handle missing values
    df = handle_missing_values(input_df)

    # Step 2: Run automated QA and generate QA report
    qa_report = generate_qa_report(df)

    # Step 3: Perform sanity check for chills response using dynamic columns
    df = sanity_check_chills(df, chills_column, chills_intensity_column, intensity_threshold)

    # Step 4: Preprocess data
    intermediate_df = preprocess_data(df.copy())

    # Step 5: Calculate the scores
    scorer = ScaleScorer(intermediate_df, user_column_mappings)
    final_df = scorer.calculate_all_scales()

    return final_df, intermediate_df, str(qa_report)

# Testing Semantic Analysis
if __name__ == "__main__":
    from scipy.stats import kendalltau
    # Step 1: Create Test Data
    test_data = {
        "Intensity": ["not at all", "extremely", "quite a bit", "moderately", "a little"],
        "Frequency": ["never", "sometimes", "often", "always", "rarely"],
        "Custom Scale": ["minimal", "extreme", "moderate", "low", "high"]
    }

    df = pd.DataFrame(test_data)

    # Step 2: Run Semantic Ranking
    for column in df.columns:
        col_values = df[column].unique()
        ordered_categories = determine_category_order(col_values)
        print(f"Column: {column}")
        print(f"Original Categories: {col_values}")
        print(f"Determined Order: {ordered_categories}\n")

    # Step 3: Qualitative Analysis
    # Manually verify the determined order by looking at the printed output

    # Step 4: Quantitative Evaluation using Kendall's Tau
    for column in df.columns:
        if column == "Intensity":
            expected_order = ["not at all", "a little", "moderately", "quite a bit", "extremely"]
        elif column == "Frequency":
            expected_order = ["never", "rarely", "sometimes", "often", "always"]
        else:
            continue  # Skip Custom Scale since it has no defined order for comparison

        ranked_order = determine_category_order(df[column].unique())

        # Convert categories to numeric rank based on expected and actual order
        expected_ranks = [expected_order.index(x) for x in expected_order]
        actual_ranks = [ranked_order.index(x) for x in expected_order]

        # Calculate Kendall's Tau
        tau, p_value = kendalltau(expected_ranks, actual_ranks)
        print(f"Kendall's Tau for '{column}': {tau}, P-value: {p_value}")

    # Step 5: Iterate based on findings (manual iteration step)
    # If Kendall's Tau value is low or p-value indicates poor correlation,
    # consider modifying the semantic ranking fallback or adding more predefined sets.

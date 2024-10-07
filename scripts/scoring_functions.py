from scales.scale_mappings import get_scale_questions

# Function to score the MODTAS scale
def score_modtas(df):
    """
    Calculate the MODTAS (Modified Tellegen Absorption Scale) score for each row in the DataFrame.

    Parameters:
    df : pd.DataFrame
        Input DataFrame with columns corresponding to the MODTAS questions.

    Returns:
    pd.Series
        A Series containing the MODTAS scores for each row in the DataFrame.
    """
    # Get the list of questions for the MODTAS scale
    modtas_questions = get_scale_questions('MODTAS')

    # Check if the necessary questions are in the dataframe
    missing_columns = [q for q in modtas_questions if q not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for MODTAS scoring: {missing_columns}")

    # Sum the values for all MODTAS questions (assuming Likert scale 0 to 4)
    return df[modtas_questions].sum(axis=1)

# TODO - make functions for other scales in the questionnaire



def calculate_all_scales(df):
    """
    Calculate all available scale scores for the input DataFrame and drop question columns after scoring.

    Parameters:
    df : pd.DataFrame
        Input DataFrame containing columns corresponding to multiple scales.

    Returns:
    pd.DataFrame
        A DataFrame containing the original columns with additional columns for each calculated scale.
        All question columns are removed after scoring.
    """
    df_scored = df.copy()
    scoring_functions = {
        'MODTAS': score_modtas,  # Add other scale functions here as needed
    }

    # Track all question columns to be dropped
    question_columns_to_drop = []

    # Calculate each scale score and add as a new column
    for scale_name, scoring_fn in scoring_functions.items():
        try:
            # Calculate the score
            df_scored[scale_name + '_Score'] = scoring_fn(df)
            print(f"Successfully scored {scale_name}")

            # Add the columns used in this scale to the drop list
            question_columns_to_drop.extend(get_scale_questions(scale_name))
        except ValueError as e:
            print(f"Skipping {scale_name} due to missing columns: {e}")

    # Remove the columns used for scoring
    df_scored = df_scored.drop(columns=question_columns_to_drop, errors='ignore')
    return df_scored


# Testing MODTAS
if __name__ == "__main__":
    import pandas as pd

    # Create a sample DataFrame for MODTAS with some random data
    sample_data = {
        "Sometimes I feel and experience things as I did when I was a child.": [1, 2, 0, 3],
        "I can be greatly moved by eloquent or poetic language.": [4, 3, 2, 1],
        "While watching a movie, a T.V. show, or a play, I may become so involved that I forgot about myself and my surroundings, and experience the story as if it were real and as if I were taking part in it.": [
            2, 3, 1, 4],
        "If I stare at a picture and then look away from it, I can sometimes “see” an image of the picture, almost as if I were still looking at it.": [
            1, 4, 3, 0],
        "Sometimes I feel as if my mind could envelop the whole world.": [3, 2, 1, 4],
        "I like to watch cloud shapes change in the sky.": [0, 2, 3, 4],
        "If I wish I can imagine (or daydream) some things so vividly that it’s like watching a good movie or hearing a good story.": [
            3, 1, 4, 2],
        "I think I really know what some people mean when they talk about mystical experiences.": [4, 2, 3, 1],
        "I sometimes “step outside” my usual self and experience a completely different state of being.": [2, 3, 4, 1],
        "Textures—such as wool, sand, wood—sometimes remind me of colors or music.": [1, 0, 2, 3],
        "Sometimes I experience things as if they were doubly real.": [3, 4, 1, 2],
        "When I listen to music I can get so caught up in it that I don’t notice anything else.": [2, 1, 3, 4],
        "If I wish I can imagine that my body is so heavy that I cannot move it.": [4, 3, 2, 1],
        "I can often somehow sense the presence of another person before I actually see or hear her/him.": [3, 2, 4, 0],
        "The crackle and flames of a wood fire stimulate my imagination.": [1, 4, 2, 3],
        "Sometimes I am so immersed in nature or in art that I feel as if my whole state of consciousness has somehow been temporarily changed.": [
            0, 2, 3, 4],
        "Different colors have distinctive and special meanings for me.": [3, 1, 4, 2],
        "I can so completely wander off into my own thoughts while doing a routine task that I actually forget that I am doing the task and find a few minutes later that I have finished it.": [
            4, 3, 2, 1],
        "I can sometimes recall certain past experiences in my life so clearly and vividly that it is like living them again, or almost so.": [
            2, 3, 1, 4],
        "Things that might seem meaningless to others often make sense to me.": [3, 4, 2, 0],
        "If I acted in a play I think I would really feel the emotions of the character and “become” that person for the time being, forgetting both myself and the audience.": [
            1, 2, 3, 4],
        "My thoughts often occur as visual images rather than as words.": [4, 3, 1, 2],
        "I am often delighted by small things (like the colors in soap bubbles and the five-pointed star shape that appears when you cut an apple across the core).": [
            2, 1, 4, 3],
        "When listening to organ music or other powerful music, I sometimes feel as if I am being lifted into the air.": [
            3, 4, 0, 1],
        "Sometimes I can change noise into music by the way I listen to it.": [1, 3, 4, 2],
        "Some of my most vivid memories are called up by scents and smells.": [4, 2, 3, 1],
        "Some music reminds me of pictures or changing patterns of color.": [0, 1, 2, 3],
        "I often know what someone is going to say before he or she says it.": [2, 3, 4, 1],
        "I often have “physical memories”; for example, after I’ve been swimming I may feel as if I’m still in the water.": [
            3, 4, 1, 2],
        "The sound of a voice can be so fascinating to me that I can just go on listening to it.": [1, 2, 3, 4],
        "At times I somehow feel the presence of someone who is not physically there.": [4, 3, 2, 1],
        "Sometimes thoughts and images come to me without any effort on my part.": [2, 1, 4, 3],
        "I find that different smells have different colors.": [3, 4, 2, 1],
        "I can be deeply moved by a sunset.": [1, 2, 3, 4]
    }

    # Create the sample DataFrame
    df_test = pd.DataFrame(sample_data)

    # Calculate MODTAS scores using the function
    scored_df = calculate_all_scales(df_test)

    # Print the resulting DataFrame with calculated scores
    print(scored_df)

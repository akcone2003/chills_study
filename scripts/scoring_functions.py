# TODO - adding scoring the behavioral measures into pipeline for users to dynamically aggregate the survey data
# idea - maybe make a function for each measure that has the algorithm for aggregating the data?
# outline - create the scoring functions and create dictionaries/maps to call the functions on
# aggregates the data and creates a new column in the dataframe
# TODO - look into developing a scraper that scrapes all the scoring measures

import pandas as pd

#-----------------------------------------------------------FUNCTIONS------------------------------------------------

# MODTAS Scoring Function
def score_modtas(df, modtas_columns):
    """
    Score the Modified Tellegen Absorption Scale (MODTAS).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the survey responses.
    modtas_columns : list
        List of column names corresponding to the MODTAS items.

    Returns
    -------
    pd.Series
        A pandas Series with the calculated MODTAS sum for each row.
    """
    # Calculate the MODTAS sum by summing over the specified columns
    modtas_scores = df[modtas_columns].sum(axis=1)
    return modtas_scores

#--------------------------------------------------MAPS---------------------------------------------------------------
# Define a mapping for scales and their corresponding columns
# Define the question-to-category mappings
scale_mapping = {
    'Religiosity': [
        'Please indicate the extent to which the following statements apply to you. [How often do you think about religious issues?]',
        'Please indicate the extent to which the following statements apply to you. [To what extent do you believe that God(s) or something divine exists?]',
        'Please indicate the extent to which the following statements apply to you. [How often do you take part in religious services?]',
        'Please indicate the extent to which the following statements apply to you. [How often do you pray?]',
        'Please indicate the extent to which the following statements apply to you. [How often do you meditate?]',
        'Please indicate the extent to which the following statements apply to you. [How often do you experience situations in which you have the feeling that God(s) or something(s) divine intervenes in your life?]',
        'Please indicate the extent to which the following statements apply to you. [How often do you experience situations in which you have the feeling that you are one with all?]'
    ],
    'Meditation Experience': [
        'Do you have any experience with meditation?'
    ],
    'VVIQ': [
        'For each scenario try to form a mental picture of the people, objects, or setting. Rate how vivid the image is using the 5-point scale. If you do not have a visual image, rate vividness as ‘1’. [The exact contours of face, head, shoulders, and body.]',
        'For each scenario try to form a mental picture of the people, objects, or setting. [Characteristic poses of head, attitudes of body, etc.]',
        'For each scenario try to form a mental picture of the people, objects, or setting. [The precise carriage, length of step, etc., in walking.]',
        'For each scenario try to form a mental picture of the people, objects, or setting. [The different colors worn in some familiar clothes.]',
        'Visualize a rising sun. [The sun rising above the horizon into a hazy sky.]',
        'Visualize a rising sun. [The sky clears and surrounds the sun with blueness.]',
        'Visualize a rising sun. [Clouds. A storm blows up with flashes of lightning.]',
        'Visualize a rising sun. [A rainbow appears.]'
    ],
    'MODTAS': [
        'Please rate how frequent the following experiences are for you. [Sometimes I feel and experience things as I did when I was a child.]',
        'Please rate how frequent the following experiences are for you. [I can be greatly moved by eloquent or poetic language.]',
        'Please rate how frequent the following experiences are for you. [While watching a movie, a T.V. show, or a play, I may become so involved that I forgot about myself and my surroundings, and experience the story as if it were real and as if I were taking part in it.]',
        'Please rate how frequent the following experiences are for you. [If I stare at a picture and then look away from it, I can sometimes “see” an image of the picture, almost as if I were still looking at it.]',
        'Please rate how frequent the following experiences are for you. [Sometimes I feel as if my mind could envelop the whole world.]',
        'Please rate how frequent the following experiences are for you. [I like to watch cloud shapes change in the sky.]',
        'Please rate how frequent the following experiences are for you. [If I wish I can imagine (or daydream) some things so vividly that it’s like watching a good movie or hearing a good story.]',
        'Please rate how frequent the following experiences are for you. [I think I really know what some people mean when they talk about mystical experiences.]',
        'Please rate how frequent the following experiences are for you. [I sometimes “step outside” my usual self and experience a completely different state of being.]',
        'Please rate how frequent the following experiences are for you. [Textures—such as wool, sand, wood—sometimes remind me of colors or music.]',
        'Please rate how frequent the following experiences are for you. [Sometimes I experience things as if they were doubly real.]',
        'Please rate how frequent the following experiences are for you. [When I listen to music I can get so caught up in it that I don’t notice anything else.]',
        'Please rate how frequent the following experiences are for you. [If I wish I can imagine that my body is so heavy that I cannot move it.]',
        'Please rate how frequent the following experiences are for you. [I can often somehow sense the presence of another person before I actually see or hear her/him.]',
        'Please rate how frequent the following experiences are for you. [The crackle and flames of a wood fire stimulate my imagination.]',
        'Please rate how frequent the following experiences are for you. [Sometimes I am so immersed in nature or in art that I feel as if my whole state of consciousness has somehow been temporarily changed.]',
        'Please rate how frequent the following experiences are for you. [Different colors have distinctive and special meanings for me.]',
        'Please rate how frequent the following experiences are for you. [I can so completely wander off into my own thoughts while doing a routine task that I actually forget that I am doing the task and find a few minutes later that I have finished it.]',
        'Please rate how frequent the following experiences are for you. [I can sometimes recall certain past experiences in my life so clearly and vividly that it is like living them again, or almost so.]',
        'Please rate how frequent the following experiences are for you. [Things that might seem meaningless to others often make sense to me.]',
        'Please rate how frequent the following experiences are for you. [If I acted in a play I think I would really feel the emotions of the character and “become” that person for the time being, forgetting both myself and the audience.]',
        'Please rate how frequent the following experiences are for you. [My thoughts often occur as visual images rather than as words.]',
        'Please rate how frequent the following experiences are for you. [I am often delighted by small things (like the colors in soap bubbles and the five-pointed star shape that appears when you cut an apple across the core).]',
        'Please rate how frequent the following experiences are for you. [When listening to organ music or other powerful music, I sometimes feel as if I am being lifted into the air.]',
        'Please rate how frequent the following experiences are for you. [Sometimes I can change noise into music by the way I listen to it.]',
        'Please rate how frequent the following experiences are for you. [Some of my most vivid memories are called up by scents and smells.]',
        'Please rate how frequent the following experiences are for you. [Some music reminds me of pictures or changing patterns of color.]',
        'Please rate how frequent the following experiences are for you. [I often know what someone is going to say before he or she says it.]',
        'Please rate how frequent the following experiences are for you. [I often have “physical memories”; for example, after I’ve been swimming I may feel as if I’m still in the water.]',
        'Please rate how frequent the following experiences are for you. [The sound of a voice can be so fascinating to me that I can just go on listening to it.]',
        'Please rate how frequent the following experiences are for you. [At times I somehow feel the presence of someone who is not physically there.]',
        'Please rate how frequent the following experiences are for you. [Sometimes thoughts and images come to me without any effort on my part.]',
        'Please rate how frequent the following experiences are for you. [I find that different smells have different colors.]',
        'Please rate how frequent the following experiences are for you. [I can be deeply moved by a sunset.]'
    ],
    'KAMF': [
        'KAMF_1', 'KAMF_1r', 'KAMF_2', 'KAMF_3_1', 'KAMF_4', 'KAMF_4r'
    ], # TODO - figure out how to score KAMF
    'DPES-Awe': [
        'How much do you agree with the following statements? [I often feel awe]',
        'How much do you agree with the following statements? [I see beauty all around me]',
        'How much do you agree with the following statements? [I often look for patterns in the objects around me.]',
        'How much do you agree with the following statements? [I have many opportunities to see the beauty of nature.]',
        'How much do you agree with the following statements? [I seek out experiences that challenge my understanding of the world.]'
    ],
    'MAIA': [
        'I can pay attention to my breath without being distracted.',
        'I can maintain awareness of my inner bodily sensations even when there is a lot going on around me.',
        'When I am in conversation with someone, I can pay attention to my posture.',
        'I can return awareness to my body if I am distracted.',
        'I can refocus my attention from thinking to sensing my body.',
        'I can maintain awareness of my whole body even when a part of me is in pain or discomfort.',
        'I am able to consciously focus on my body as a whole.'
    ],
    # TODO - Add other mappings here like 'Emotional Breakthrough', 'Psychological Insight', etc.
}


# Map scale names to their respective scoring functions
scale_scoring_functions = {
    'MODTAS': lambda df, columns: score_modtas(df, columns),
    # TODO - add more scoring functions
}


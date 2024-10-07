# Import the dictionary from `scale_questions.py`
from scales.scale_questions import ALL_SCALES

# Use this dictionary to map the scales
def get_scale_questions(scale_name):
    """
    Retrieve the list of questions for a given scale.

    Parameters:
    scale_name (str): Name of the scale to look up.

    Returns:
    list: List of questions for the scale if found, else an empty list.
    """
    return ALL_SCALES.get(scale_name, [])


# Example usage (testing purpose)
if __name__ == "__main__":
    print("MODTAS Questions: ", get_scale_questions('MODTAS'))



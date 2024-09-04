"""
The function for converting the original multi-valued target variable
into binary (healthy-seizure).

This function is part of data preparation
and called from all notebooks.
"""


def is_seizure(original_y):
    """
    convert the target variable from multiclass to binary.

    Args:
        original_y (int from 1 to 5): the category of the input vector

    Returns:
        new target (int): 0 (healthy) or 1 (seizure)
    """
    if original_y > 1:
        return 0
    else:
        return 1
      

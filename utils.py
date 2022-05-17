
import os


def label_decoder(labels: dict, x: int):
    """
    label_decoder Performs a decoding from int to label

    :param labels: A dictionary of labels and its respective integer representation
    :type labels: dict
    :param x: The integer we wish to convert
    :type x: int
    :return: Returns the label associated with the integer representation
    :rtype: Any
    """
    return list(labels.keys())[list(labels.values()).index(x)]


def plurality_vote(region_classifications: dict, classes: tuple):
    """
    plurality_vote Performs a plurality vote to determine overall accuracy based on class distribution

    :param region_classifications: Key value pairs of each region and its respective classification
    :type region_classifications: dict
    :param classes: A tuple of labels or classes that were used for training
    :type classes: tuple
    :return: Returns the label or class with the highest vote
    :rtype: Any
    """
    votes = {c: 0 for c in classes}
    for c in region_classifications.values():
        votes[c] += 1

    return votes[max(votes, key=votes.get)]


def open_file(filepath, mode='r', binary_mode=False):
    if mode in ['w', 'x']:
        mode = 'w' if os.path.exists(filepath) else 'x'
    if binary_mode:
        mode += 'b'
    return open(filepath, mode)

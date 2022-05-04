def label_decoder(labels: dict, x: int):
    """
    label_decoder _summary_

    :param labels: _description_
    :type labels: dict
    :param x: _description_
    :type x: int
    :return: _description_
    :rtype: _type_
    """
    return list(labels.keys())[list(labels.values()).index(x)]


def plurality_vote(region_classifications: dict, classes: tuple):
    """
    plurality_vote _summary_

    :param region_classifications: _description_
    :type region_classifications: dict
    :param classes: _description_
    :type classes: tuple
    :return: _description_
    :rtype: _type_
    """
    votes = {c: 0 for c in classes}
    for c in region_classifications.values():
        votes[c] += 1

    return votes[max(votes, key=votes.get)]

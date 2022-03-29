def label_decoder(labels: dict, x: int):
    return list(labels.keys())[list(labels.values()).index(x)]


def plurality_vote(region_classifications: dict, classes: tuple):
    votes = {c: 0 for c in classes}
    for c in region_classifications.values():
        votes[c] += 1

    return votes[max(votes, key=votes.get)]
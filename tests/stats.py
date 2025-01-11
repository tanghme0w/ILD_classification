import os

def get_data(filepath):
    """ read csv file and return a dictionary with the number of lines, words, and characters

    Args:
        filepath (str): the path to the csv file

    Returns:
        three lists, each is a column in the csv file
    """

    lines = []
    with open(filepath, 'r') as file:
        content = file.readlines()
    
    for i, line in enumerate(content):
        # ignore first two lines
        if i < 2:
            continue
        entry = line.strip().split(',')
        lines.append(entry)

    # split the lines into three lists  
    centers = [line[0] for line in lines]
    ids = [line[1] for line in lines]
    labels = [line[2] for line in lines]

    return lines, centers, ids, labels


def get_center_label_distribution(centers, labels):
    """ get the distribution of the labels for each center

    Args:
        centers (list): the list of centers
        labels (list): the list of labels

    Returns:
        a dictionary with the distribution of the labels for each center
    """
    distribution = {}
    for center, label in zip(centers, labels):
        if center not in distribution:
            distribution[center] = {}
        distribution[center][label] = distribution[center].get(label, 0) + 1
    return distribution


if __name__ == "__main__":
    filepath = "/tanghaomiao/medai/label4Tang.csv"
    lines, centers, ids, labels = get_data(filepath)
    distribution = get_center_label_distribution(centers, labels)
    print(distribution)

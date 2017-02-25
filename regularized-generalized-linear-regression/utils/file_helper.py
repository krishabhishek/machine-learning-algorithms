import json


def read_data_file(data_file_path):

    data_vectors = list()
    with open(data_file_path) as data_file:
        for line in data_file:
            vector = line.split(",")
            result = [1.0] + list(map(float, vector))
            data_vectors.append(result)

    return data_vectors


def read_label_file(label_file_path):

    labels = list()
    with open(label_file_path) as label_file:
        for line in label_file:
            label = float(line.strip())
            labels.append(label)

    return labels


def dump_dict_to_file(results, metrics_file_path):
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(results, metrics_file)

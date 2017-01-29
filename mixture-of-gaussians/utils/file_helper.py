data_file_prefix = "/data"
data_file_suffix = ".csv"
label_file_prefix = "/labels"
label_file_suffix = ".csv"


def read_data_file(data_file_path):

    data_vectors = list()
    with open(data_file_path) as data_file:
        for line in data_file:
            vector = list(map(int, line.split(",")))
            data_vectors.append(vector)

    return data_vectors


def read_label_file(label_file_path):

    labels = list()
    with open(label_file_path) as label_file:
        for line in label_file:
            label = str(line.strip())
            labels.append(label)

    return labels


def get_datasets(input_data_folder, test_set_identifier, file_range):
    train_set_vectors = list()
    train_set_labels = list()
    test_set_vectors = list()
    test_set_labels = list()

    for current_segment in file_range:
        if current_segment == test_set_identifier:
            test_set_vectors.extend(
                read_data_file(input_data_folder + data_file_prefix + str(test_set_identifier) + data_file_suffix)
            )
            test_set_labels.extend(
                read_label_file(input_data_folder + label_file_prefix + str(test_set_identifier) + label_file_suffix)
            )
        else:
            train_set_vectors.extend(
                read_data_file(input_data_folder + data_file_prefix + str(test_set_identifier) + data_file_suffix)
            )
            train_set_labels.extend(
                read_label_file(input_data_folder + label_file_prefix + str(test_set_identifier) + label_file_suffix)
            )

    return train_set_vectors, train_set_labels, test_set_vectors, test_set_labels

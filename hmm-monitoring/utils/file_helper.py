train_data_file_prefix = "/trainDataSeq"
train_label_file_prefix = "/trainLabelSeq"
test_data_file_prefix = "/testDataSeq"
test_label_file_prefix = "/testLabelSeq"

file_suffix = ".csv"



def read_data_file(data_file_path):

    data_vectors = list()
    with open(data_file_path) as data_file:
        for line in data_file:
            vector = list(map(float, line.split(",")))
            data_vectors.append(vector)

    return data_vectors


def read_label_file(label_file_path):

    labels = list()
    with open(label_file_path) as label_file:
        for line in label_file:
            label = str(line.strip())
            labels.append(label)

    return labels


def get_datasets(input_data_folder, file_range):
    train_set_vectors = list()
    train_set_labels = list()
    test_set_vectors = list()
    test_set_labels = list()
    vector_sequences = list()
    label_sequences = list()

    for i in file_range:
        # Building train set
        tmp_vectors = read_data_file(input_data_folder + train_data_file_prefix + str(i) + file_suffix)
        train_set_vectors.extend(tmp_vectors)
        vector_sequences.append(tmp_vectors)

        tmp_labels = read_label_file(input_data_folder + train_label_file_prefix + str(i) + file_suffix)
        train_set_labels.extend(tmp_labels)
        label_sequences.append(tmp_labels)

        # Building test set
        tmp_vectors = read_data_file(input_data_folder + test_data_file_prefix + str(i) + file_suffix)
        test_set_vectors.extend(tmp_vectors)
        vector_sequences.append(tmp_vectors)

        tmp_labels = read_label_file(input_data_folder + test_label_file_prefix + str(i) + file_suffix)
        test_set_labels.extend(tmp_labels)
        # label_sequences.append(tmp_labels)

    return train_set_vectors, train_set_labels, test_set_vectors, test_set_labels, vector_sequences, label_sequences

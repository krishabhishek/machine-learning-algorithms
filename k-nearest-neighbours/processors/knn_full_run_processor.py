import numpy as np

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.knn import Knn

log = log_helper.get_logger("KnnFullRunProcessor")


class KnnFullRunProcessor(Processor):

    def process(self):
        log.info("KnnFullRunProcessor begun")

        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("min_k: " + str(self.options.min_k))
        log.info("max_k: " + str(self.options.max_k))

        data_vectors = list()
        labels = list()
        for i in range(1, 11):
            data_vectors.extend(file_helper.read_data_file(self.options.input_data_folder + "data" + str(i) + ".csv"))
            labels.extend(file_helper.read_label_file(self.options.input_data_folder + "labels" + str(i) + ".csv"))

        log.info("Running full training set")
        knn = Knn()
        X_train = np.array(data_vectors)
        y_train = np.array(labels)

        knn.fit(X_train, y_train)
        results = dict()
        for i in range(self.options.min_k, self.options.max_k + 1):
            predicted_y = knn.predict(X_train, i)
            total = 0
            correct = 0
            for j in range(len(predicted_y)):
                if predicted_y[j] == y_train[j]:
                    correct += 1
                total += 1

            log.info("full accuracy: " + str(round((correct * 1.0 / total), 2)) + " for k = " + str(i))
            results[i] = round((correct * 1.0 / total), 2)

        file_helper.dump_dict_to_file(results, self.options.metrics_file)

        log.info("KnnFullRunProcessor concluded")

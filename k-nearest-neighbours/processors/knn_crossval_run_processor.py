import numpy as np

from sklearn.model_selection import KFold

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.knn import Knn

log = log_helper.get_logger("KnnCrossvalRunProcessor")


class KnnCrossvalRunProcessor(Processor):

    def process(self):
        log.info("KnnCrossvalRunProcessor begun")

        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("min_k: " + str(self.options.min_k))
        log.info("max_k: " + str(self.options.max_k))

        data_vectors = list()
        labels = list()
        for i in range(1, 11):
            data_vectors.extend(file_helper.read_data_file(self.options.input_data_folder + "data" + str(i) + ".csv"))
            labels.extend(file_helper.read_data_file(self.options.input_data_folder + "labels" + str(i) + ".csv"))

        log.info("Running 10-fold cross validation segment")
        kf = KFold(n_splits=10)
        results = dict()
        for i in range(self.options.min_k, self.options.max_k + 1):
            total = 0
            correct = 0
            for train_index, test_index in kf.split(data_vectors):
                X_train, X_test = np.array(data_vectors)[train_index], np.array(data_vectors)[test_index]
                y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

                knn = Knn()
                knn.fit(X_train, y_train)
                predicted_y = knn.predict(X_test, i)

                for j in range(len(predicted_y)):
                    if predicted_y[j] == y_test[j]:
                        correct += 1
                    total += 1

            log.info("accuracy: " + str(round((correct * 1.0 / total), 2)) + " for k = " + str(i))
            results[i] = round((correct * 1.0 / total), 2)

        file_helper.dump_dict_to_file(results, self.options.metrics_file)

        log.info("KnnCrossvalRunProcessor concluded")

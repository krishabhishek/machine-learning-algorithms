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
            labels.extend(file_helper.read_label_file(self.options.input_data_folder + "labels" + str(i) + ".csv"))

        log.info("Running 10-fold cross validation segment")

        results = dict()
        for i in range(self.options.min_k, self.options.max_k + 1):
            total = 0
            correct = 0

            for j in range(1, 11):
                x_train = list()
                y_train = list()
                x_test = list()
                y_test = list()

                for k in range(1, 11):
                    if j == k:
                        x_test = file_helper.read_data_file(self.options.input_data_folder + "data" + str(k) + ".csv")
                        y_test = file_helper.read_label_file(self.options.input_data_folder + "labels" + str(k) + ".csv")
                        continue
                    x_train.extend(file_helper.read_data_file(self.options.input_data_folder + "data" + str(k) + ".csv"))
                    y_train.extend(file_helper.read_label_file(self.options.input_data_folder + "labels" + str(k) + ".csv"))

                knn = Knn()
                knn.fit(x_train, y_train)
                predicted_y = knn.predict(x_test, i)

                for k in range(len(predicted_y)):
                    if predicted_y[k] == y_test[k]:
                        correct += 1
                    total += 1

            accuracy = correct / total
            log.info("accuracy: " + str(accuracy) + " for k = " + str(i))
            results[i] = accuracy

        file_helper.dump_dict_to_file(results, self.options.metrics_file)

        log.info("KnnCrossvalRunProcessor concluded")

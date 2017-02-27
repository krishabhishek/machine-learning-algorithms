import time

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.bayesian_linear_regressor import BayesianLinearRegression
from utils.metrics_helper import calculate_euclidean_loss

log = log_helper.get_logger("BayesianLinearRegressionProcessor")


class BayesianLinearRegressionProcessor(Processor):

    def process(self):
        log.info("BayesianLinearRegressionProcessor begun")
        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("max_degree: " + str(self.options.max_degree))

        log.info("Running 10-fold cross validation segment")
        results = list()
        j = 1
        regularization_parameter = 0.1
        degree = 1
        while degree <= self.options.max_degree:
            start_time = time.time()
            error_list = list()
            for j in range(1, 11):

                X_train = list()
                y_train = list()
                X_test = list()
                y_test = list()

                for i in range(1, 11):
                    if j == i:
                        X_test = file_helper.read_data_file(self.options.input_data_folder + "fData" + str(i) + ".csv")
                        y_test = file_helper.read_label_file(self.options.input_data_folder + "fLabels" + str(i) + ".csv")
                        continue

                    X_train.extend(file_helper.read_data_file(self.options.input_data_folder + "fData" + str(i) + ".csv"))
                    y_train.extend(file_helper.read_label_file(self.options.input_data_folder + "fLabels" + str(i) + ".csv"))

                lr = BayesianLinearRegression(X_train, y_train, regularization_parameter, degree)
                predicted_y = lr.predict(X_test)

                error = calculate_euclidean_loss(predictions=predicted_y, target=y_test)
                error_list.append(error)

            avg_error = sum(error_list) / float(len(error_list))
            elapsed_time = time.time() - start_time
            log.info("For degree = " + str(degree) + ", euclidean loss = " + str(avg_error) + ", elapsed time: " +
                     str(elapsed_time) + " seconds")

            result = dict()
            result["degree"] = degree
            result["error"] = avg_error
            result["elapsed_time"] = elapsed_time

            results.append(result)

            degree += 1

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("BayesianLinearRegressionProcessor concluded")

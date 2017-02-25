from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.generalized_linear_regressor import GeneralizedLinearRegression
from utils.metrics_helper import calculate_euclidean_loss

log = log_helper.get_logger("GeneralizedLinearRegressionProcessor")


class GeneralizedLinearRegressionProcessor(Processor):

    def process(self):
        log.info("GeneralizedLinearRegressionProcessor begun")
        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("max_degree: " + str(self.options.max_degree))

        log.info("Running 10-fold cross validation segment")
        results = dict()
        j = 1
        regularization_parameter = 0.1
        degree = 1
        while degree <= self.options.max_degree:
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

                lr = GeneralizedLinearRegression(X_train, y_train, regularization_parameter)
                predicted_y = lr.predict(X_test)

                error = calculate_euclidean_loss(predictions=predicted_y, target=y_test)
                error_list.append(error)

            avg_error = sum(error_list) / float(len(error_list))
            log.info("For degree = " + str(degree) + ", euclidean loss = " + str(avg_error))
            results[degree] = avg_error

            degree += 1

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("GeneralizedLinearRegressionProcessor concluded")

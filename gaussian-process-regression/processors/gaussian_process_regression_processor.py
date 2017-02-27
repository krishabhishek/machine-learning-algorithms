from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.gaussian_process_regressor import GaussianProcessRegression
from utils.metrics_helper import calculate_euclidean_loss

log = log_helper.get_logger("GaussianProcessRegressionProcessor")


class GaussianProcessRegressionProcessor(Processor):

    def process(self):
        log.info("GaussianProcessRegressionProcessor begun")
        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("min_lambda: " + str(self.options.min_lambda))
        log.info("max_lambda: " + str(self.options.max_lambda))
        log.info("lambda_increment: " + str(self.options.lambda_increment))

        log.info("Running 10-fold cross validation segment")
        results = dict()
        l = self.options.min_lambda
        j = 1
        while l <= self.options.max_lambda + self.options.lambda_increment:
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

                lr = GaussianProcessRegression(X_train, y_train, l)
                predicted_y = lr.predict(X_test)

                error = calculate_euclidean_loss(predictions=predicted_y, target=y_test)
                error_list.append(error)

            avg_error = sum(error_list) / float(len(error_list))
            log.info("For lambda = " + str(round(l, 1)) + ", euclidean loss = " + str(avg_error))
            results[round(l, 1)] = avg_error

            l += self.options.lambda_increment

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("GaussianProcessRegressionProcessor concluded")

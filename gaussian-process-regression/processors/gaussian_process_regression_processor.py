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
        log.info("kernel_type: " + self.options.kernel_type)
        log.info("metrics_file: " + self.options.metrics_file)

        log.info("Running 10-fold cross validation segment")
        results = dict()
        error_list = list()
        for j in range(1, 2):

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

            lr = GaussianProcessRegression(X_train, y_train, self.options.kernel)
            predicted_y = lr.predict(X_test)

            error = calculate_euclidean_loss(predictions=predicted_y, target=y_test)
            error_list.append(error)

        avg_error = sum(error_list) / float(len(error_list))
        log.info("Avg error: " + str(avg_error))

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("GaussianProcessRegressionProcessor concluded")

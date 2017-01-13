from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy

from processors.processor import Processor
from utils import file_helper
from utils import log_helper
from utils.linear_regressor import LinearRegression

log = log_helper.get_logger("LinearRegressionProcessor")


class LinearRegressionProcessor(Processor):

    def process(self):
        log.info("LinearRegressionProcessor begun")
        log.info("input_data_folder: " + self.options.input_data_folder)
        log.info("min_lambda: " + str(self.options.min_lambda))
        log.info("max_lambda: " + str(self.options.max_lambda))
        log.info("lambda_increment: " + str(self.options.lambda_increment))

        data_vectors = list()
        labels = list()
        for i in range(1, 11):
            data_vectors.extend(file_helper.read_data_file(self.options.input_data_folder + "fData" + str(i) + ".csv"))
            labels.extend(file_helper.read_label_file(self.options.input_data_folder + "fLabels" + str(i) + ".csv"))

        log.info("Running 10-fold cross validation segment")
        kf = KFold(n_splits=10)
        results = dict()
        l = self.options.min_lambda
        while l <= self.options.max_lambda + self.options.lambda_increment:
            error_list = list()
            for train_index, test_index in kf.split(data_vectors):
                X_train, X_test = numpy.array(data_vectors)[train_index], numpy.array(data_vectors)[test_index]
                y_train, y_test = numpy.array(labels)[train_index], numpy.array(labels)[test_index]

                lr = LinearRegression(X_train, y_train, l)
                predicted_y = lr.predict(X_test)

                error = r2_score(y_test.tolist(), predicted_y)
                error_list.append(error)

            avg_error = sum(error_list) / float(len(error_list))
            log.info("For lambda = " + str(round(l, 1)) + ", r2_score = " + str(avg_error))
            results[round(l, 1)] = avg_error

            l += self.options.lambda_increment

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("LinearRegressionProcessor concluded")

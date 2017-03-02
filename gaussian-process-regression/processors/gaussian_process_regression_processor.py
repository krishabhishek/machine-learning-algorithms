from time import time

from kernels.gaussian_kernel import GaussianKernel
from kernels.identity_kernel import IdentityKernel
from kernels.polynomial_kernel import PolynomialKernel
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
        results = dict()

        if self.options.kernel_type == "IdentityKernel":
            self.options.kernel = IdentityKernel()
            start_time = time()
            error = self.run_cross_validation_phase()
            elapsed_time = time() - start_time
            results['error'] = error
            results['elapsed_time'] = elapsed_time
        elif self.options.kernel_type == "GaussianKernel":
            if self.options.gaussian_stddev:
                for i in range(self.options.gaussian_stddev):
                    self.options.kernel = GaussianKernel(i+1)
                    start_time = time()
                    error = self.run_cross_validation_phase()
                    elapsed_time = time() - start_time
                    result = dict()
                    result['error'] = error
                    result['elapsed_time'] = elapsed_time
                    results[i+1] = result
            else:
                msg = "'gaussian_stddev' argument required for GaussianKernel"
                log.error(msg)
                raise RuntimeError(msg)
        elif self.options.kernel_type == "PolynomialKernel":
            if self.options.polynomial_degree:
                for i in range(self.options.polynomial_degree):
                    self.options.kernel = PolynomialKernel(i+1)
                    start_time = time()
                    error = self.run_cross_validation_phase()
                    elapsed_time = time() - start_time
                    result = dict()
                    result['error'] = error
                    result['elapsed_time'] = elapsed_time
                    results[i+1] = result
            else:
                msg = "'polynomial_degree' argument required for PolynomialKernel"
                log.error(msg)
                raise RuntimeError(msg)
        else:
            msg = "Invalid kernel chosen. The choices are IdentityKernel, GaussianKernel & PolynomialKernel. Exiting"
            log.error(msg)
            raise RuntimeError(msg)

        file_helper.dump_dict_to_file(results, self.options.metrics_file)
        log.info("GaussianProcessRegressionProcessor concluded")

    def run_cross_validation_phase(self):

        log.info("Running 10-fold cross validation segment")
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

            lr = GaussianProcessRegression(X_train, y_train, self.options.kernel)
            predicted_y = lr.predict(X_test)

            error = calculate_euclidean_loss(predictions=predicted_y, target=y_test)
            error_list.append(error)

        avg_error = sum(error_list) / float(len(error_list))
        log.info("Avg error: " + str(avg_error))

        return avg_error

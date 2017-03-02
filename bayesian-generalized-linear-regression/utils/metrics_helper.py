def calculate_euclidean_loss(predictions, target):

    sum_of_squared_errors = 0
    for i in range(len(predictions)):
        diff = predictions[i] - target[i]
        squared_error = pow(diff, 2)
        sum_of_squared_errors += squared_error

    return sum_of_squared_errors / len(predictions)

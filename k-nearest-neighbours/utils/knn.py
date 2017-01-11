from scipy.spatial import distance

import sys


class Knn(object):

    def __init__(self, train_set_x, test_set_x, train_set_y, test_set_y, k):
        self.train_set_x = train_set_x
        self.test_set_x = test_set_x
        self.train_set_y = train_set_y
        self.test_set_y = test_set_y
        self.k = k

    def get_test_set_predictions(self):

        for i in xrange(len(self.test_set_x)):
            dist_dict = dict()

            for j in xrange(len(self.train_set_x)):
                dist = distance.euclidean(self.test_set_x[i], self.train_set_x[j])
                prev_list = dist_dict[dist]
                if prev_list:
                    prev_list.append(j)
                else:
                    prev_list = [j]
                dist_dict[dist] = prev_list

            final_point_indices = []
            sorted_distances = sorted(dist_dict.keys())

            i = 0
            while len(final_point_indices) < self.k:
                list_indices = sorted_distances[i]
                final_point_indices.extend(list_indices)
                i += 1

            final_point_indices = final_point_indices[:self.k]

            label_count_dict = dict()
            for index in final_point_indices:
                value = self.train_set_y[index]

                count = label_count_dict[value]
                if count:
                    label_count_dict[value] += 1
                else:
                    label_count_dict[value] = 1

            common_label = None
            highest_frequency = -sys.maxsize
            for value in label_count_dict.keys():
                if label_count_dict[value] > highest_frequency:
                    highest_frequency = label_count_dict[value]
                    common_label = value

            test_set_predicted_y = list()
            test_set_predicted_y[i] = common_label


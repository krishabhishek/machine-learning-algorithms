from scipy.spatial import distance

import sys


class Knn(object):

    def __init__(self):
        self.train_set_x = None
        self.train_set_y = None

    def fit(self, train_set_x, train_set_y):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

    def predict(self, test_set_x, k):
        test_set_x = test_set_x.tolist()
        test_set_predicted_y = list()
        for i in xrange(len(test_set_x)):

            # dictionary containing mapping of distances of test_set[i] to a list of indices in the train_set
            dist_dict = dict()

            # calculate distance vs each point in the train set, and add it to the dictionary
            for j in xrange(len(self.train_set_x)):
                dist = distance.euclidean(test_set_x[i], self.train_set_x[j])
                prev_list = None
                if dist in dist_dict:
                    prev_list = dist_dict[dist]
                    prev_list.append(j)
                else:
                    prev_list = [j]
                dist_dict[dist] = prev_list

            # print dist_dict

            # getting the nearest k options by finding the minimum distances in dist_dict
            final_point_indices = list()
            sorted_distances = sorted(dist_dict.keys())
            i = 0
            while len(final_point_indices) < k:
                list_indices = dist_dict[sorted_distances[i]]
                final_point_indices.extend(list_indices)
                i += 1
            final_point_indices = final_point_indices[:k]

            # counting frequency of each label
            label_count_dict = dict()
            for index in final_point_indices:
                value = self.train_set_y[index][0]  # nd-array
                if value in label_count_dict.keys():
                    label_count_dict[value] += 1
                else:
                    label_count_dict[value] = 1

            # identifying the common label
            common_label = None
            highest_frequency = -sys.maxsize
            for value in label_count_dict.keys():
                if label_count_dict[value] > highest_frequency:
                    highest_frequency = label_count_dict[value]
                    common_label = value

            test_set_predicted_y.append(common_label)

        return test_set_predicted_y

import numpy as np
from sklearn import datasets

def pmf_multivariate(data_matrix):

    if len(data_matrix.shape) == 1:
        num_rows , num_columns = (data_matrix.shape[0], 1)
    else:
        num_rows, num_columns = data_matrix.shape
    unique_rows_array, num_repetitions = np.unique(data_matrix, axis=0, return_counts=True)  #the parameter axis=0 allows to count the unique rows
    return unique_rows_array, num_repetitions/num_rows  #To obtain the probability, the count must be normalized to the total count of samples

##

iris = datasets.load_iris()
data_matrix , class_vector = iris.data , iris.target

integer_data_matrix = (10 * data_matrix).astype(int) #convert to integer
_ , iris_features_pmf = pmf_multivariate(integer_data_matrix)
print("Iris features pmf: ", iris_features_pmf)


##

def entropy(pmf):
    entropy = -np.sum(pmf * np.log2(pmf))
    return entropy


##

def mutual_info(data, f1_index, f2_index):

    f1 = data[:, f1_index]
    f2 = data[:, f2_index]

    f1_unique, pmf1 = pmf_multivariate(f1)
    f2_unique, pmf2 = pmf_multivariate(f2)

    pairs = np.column_stack((f1,f2))
    unique_pairs , join_pmf = pmf_multivariate(pairs)

    m_info = 0
    for i in range(len(unique_pairs)):
                                                    #first column
        px_index = np.where(f1_unique == unique_pairs[i][0])
                                                    #second column
        py_index = np.where(f2_unique == unique_pairs[i][1])

        px = pmf1[px_index]
        py = pmf2[py_index]
        pxy = join_pmf[i]

        m_info += (pxy * np.log2(pxy/(px*py)))

    return m_info


print('Mutual information between feature 0 and 1: ',mutual_info(data_matrix,0,1))
# print(mutual_info(data_matrix,0,1))



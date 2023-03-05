#KNN классификатор для задачи многоклассовой классификации
import numpy as np

class KNN_classifier:
  def __init__(self, n_neighbors: int, **kwargs):
    self.K = n_neighbors

  def fit(self, x: np.array, y: np.array):

    self.data = x
    self.labels = y
  pass

  def predict(self, x: np.array):
    
    predictions = []
    labeled_distances = []
    
    for i in range(x.shape[0]):
      distances_row = []
      for j in range(self.data.shape[0]):
        distances_row.append((np.linalg.norm(x[i] - self.data[j]),self.labels[j]))
      labeled_distances.append(distances_row)

    sorted_lst = []
    for i in range(len(labeled_distances)):
        sorted_lst.append(list(sorted(labeled_distances[i])))
    for i in range(len(sorted_lst)):
       labels_count = {}
       for el in sorted_lst[i][0:self.K]:
         if el[1] in labels_count:
           labels_count[el[1]] += 1
         else:
           labels_count[el[1]] = 1 
       predictions.append(max(labels_count, key=labels_count.get))
    predictions = np.array(predictions)

    return predictions
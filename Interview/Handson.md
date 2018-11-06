#### Main Steps for machine learning:
1.  Look at the big picture
2.  Get the data
3.  Discover and visualize the data to gain insights
4.  Prepare the data for Machine learning algorithm
5.  Select a model and train it
6.  Fine-tune your model
7.  Present your solution
8.  Launch, monitor and maintain your system.

#### Visualize the data
1.  housing.head(): get the top five rows's details
2.  housing.info(): get a quick description of the data, in particular the total number of rows
and each attribute's type and number of non-null values
3.  housing['a'].value_counts(): how many districts belongs to each category
4.  housing.describe(): a summary of the numerical attributes

#### Prepare the data for machine learning
1.  Create a Test set:
```
import numpy as np
def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) & test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```
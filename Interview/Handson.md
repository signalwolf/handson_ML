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
```
# sklearn solution:
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
```
2.  Data cleaning:
    1.  Deal with missing features: drop data one with missing feature; remove the feature completely;
    set the value to zero/mean/median
    ```
    housing.dropna(subset = ['total_bedrooms'])
    housing.drop('total_bedrooms', axis = 1)
    median = housing['total_bedrooms'].median()
    housing['total_bedrooms'].fillna(median, inplace = True)
    ```
    2.  Handling text and categorical attributes: convert those to numbers
    one hot encoding (consider the original distance between two entities) or linear transfer (each one is independent)
    
    3.  Feature scaling:
        1.  min_max scaling/ normalization; MinMaxScale
        2.  standardization scaling (new data have zero mean and unit variance): StandardScaler
    
    4.  Transformation Pipelines: custom the process steps and for future use.
    
#### Select and train a Model:

    1.  Better Evaluation using cross-validation:
        1.  Split the training set into a smaller training set and validation set.
        2.  Train your model base on training set and evaluate against the validation set
        3.  Easy way is to use sklearn's cross-validation feature.
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
        
#### Fine Tune Your Model:
    1.  Grid Search: automatic try different combination of hyperparameter values using cross-validation.
    2.  Randomized Search: if the search space is very large, then you should use this model.

#### Performance Measurement:
    1.  

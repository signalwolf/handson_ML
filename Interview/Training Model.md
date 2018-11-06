# Machine learning models

#### Classification:
1.  binary classification: precision and recall
2.  Multiclass Classification: 
3.  Multilabel classification:
4.  Multioutput classification: 

#### Linear Regression:
1.  Norm equation: too much computation power since it have matrix multiply, take O(n3) but it can give the global minimum.
    1.  More specified: O(m*n2 + n3), very sensitive to feature numbers. 
2.  Gradient Descent: 
    1.  Batch Gradient Descent: calculate the derivative based on whole data set. O(kmn)
        1. k is the number of iterations.
        2. m is the number of data
        3. n is the number of features
        4. Stable but slow
    2.  Stochastic Gradient Descent: calculate the derivative base on one data in data set. O(kn)
        1.  Faster but not stable
    3.  Mini Batch: calculate the derivative based on part of data set
        1.  middle performance. Most people will choice this one.
        
#### Polynomial Regression:
1.  
    
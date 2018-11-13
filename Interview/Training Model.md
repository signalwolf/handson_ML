# Machine learning models

#### Classification:
1.  binary classification: precision and recall
2.  Multiclass Classification: 
3.  Multilabel classification:
4.  Multioutput classification: 

#### Linear Regression:
1.  Norm equation: too much computation power since it have matrix multiply, take O(n3) but it can give the global minimum.
    1.  More specified: O(m*n2 + n3), very sensitive to feature numbers. 
    2.  好处是一次到位，直接得到最小点的位置并进行处理
2.  Gradient Descent: 三种方法，batch, stochastic, mini batch
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
1.  Add powers of each feature as new features, then train a linear model on this extended set of features.
2.  简单来说就是将feature 给 ** 2/3/4... 之后再做linear regression
 
#### Regularlization:
1.  Ridge Regression: 加入weight的平方，L2 regularization.将入这个 regularization 后你的weight变得尽可能的小了。
2.  Lasso Regression: 加入weight的绝对值，L1 regularization. 加入后，会使得结果倾向于清除掉不重要的feature，它类似于
直接进行了feature selection并且倾向与得到一个sparse model
3.  Early Stopping: 提早停止

#### Logisitc Regression:
1.  用于计算input 属于某个特别的class的概率（两个class）

#### Softmax/Multinomial Logisitc Regression:
1.  类似于logisitc regression, 只是它计算的是多个class的概率

#### SVM:
1.  Support Vector Machines/ Large Margin Classification
2.  SVM 非常sensitive于feature 的scale
3.  通过增加 Kernal能够形成 NonLinear SVM classification.

#### SVM Regression:


#### Decision Trees:


#### Ensemble learning and random forests:
# Machine learning models
[Classification](#Classification)  
[Linear Regression](#Linear-Regression)  
[Polynomial Regression](#Polynomial-Regression)  
[Logistic Regression](#Logistic-Regression)   
[Softmax/Multinomial Logisitc Regression](Softmax/Multinomial-Logisitc-Regression)  
[SVM](#SVM)  
[SVM Regression](#SVM-Regression)  
[Decision Trees](#Decision-Trees)  
[Ensemble learning and random forests](#Ensemble-learning-and-random-forests) 
[Regularization](#Regularization)   
[降维](#降维)
 

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
 
#### Logistic Regression:
1.  用于计算input 属于某个特别的class的概率（两个class）
2.  Logistic regression 与 linear regression 的区别与联系：
    1.  区别：
        1.  本质上logistic regression是处理分类问题，linear regression处理的是回归问题。
        2.  逻辑回归的变量是离散的，而线性回归中的变量是连续的
    2.  联系：
        1.  两者都用了ML来对样本进行建模。
        2.  两者都采用了梯度下降的方法
3.  logistic regression也可以用于多类的分别，例如 K 类就有 K个分类器，每个分类器就只判断是否是当前类，只是在训练的时候预处理就非常麻烦
所有的类别在每个node处都要分成是该类和不是该类两项而不能是之前的K 项
        
#### Softmax/Multinomial Logisitc Regression:
1.  类似于logisitc regression, 只是它计算的是多个class的概率
2.  不同于logistic regression的是他的output function 使用的是 softmax 而不是 logisitc function.

#### SVM:
1.  Support Vector Machines/ Large Margin Classification
2.  SVM 非常sensitive于feature 的scale
3.  通过增加 Kernal能够形成 NonLinear SVM classification.
4.  **在空间上线性可分的两类点在SVM的超平面上投影一定是不可分的** 见下图：
![image](https://github.com/signalwolf/handson_ML/blob/master/Interview/Image/Screen%20Shot%202018-11-13%20at%2011.55.54%20AM.png)


#### SVM Regression:


#### Decision Trees:
1.  叶子节点就是类别。
2.  决策树的形成有三个重要部分：特征的选择，树的构造，树的剪枝
3.  常用的决策树的算法有 ID3, C4.5, CRAT:
    1.  ID3: 最大信息增益：
        1.  通过计算每个feature对熵的影响来判断。需要详细看
    2.  C4.5: 最大信息增益比：
    3.  CRAT:最大基尼指数：
        1.  Gini: 表示的是数据的纯度
        2.  每次产生两个分支，因此最后会形成一个二叉树
4.  剪枝：两种方法：
    1.  Pre-pruning: 在生成树的过程中遇到某个条件便停止生长。
        1.  方法：
            1.  通过计算如果进行当前的剪枝活动，那么能够得到泛化能力的提升，如果不能就停止。
            2.  当树达到一定深度的时候停止
            3.  当到达当前node的样本数量低于某个阈值的时候就停止
            4.  当进行了这次的分裂对精确度的影响小于某个阈值的时候就停止
        2.  优点：
            1.  简单高效，适合大规模问题
        3.  缺点：
            1.  容易欠拟合，例如当前的划分导致了准确率的降低但是在之后的分析中却可能导致准确率的大幅度提高
            2.  不同的问题需要不同的深度以及阈值
            
    2.  Post-pruning: 在已经生成的过拟合的树种向上进行剪枝，从而得到简化的树。
        1.  方法：
            1.  REP (Reduced Error Pruning): 
            2.  PEP (Pessimistic Error Pruning):
            3.  CCP (Cost Complexity Pruning): 最常用
            4.  MEP (Minimum Error Pruning):
            5.  CVP (Critical Value Pruning):
            6.  OPP (Optimal Pruning):

#### Ensemble learning and random forests:

#### Regularization:
1.  Ridge Regression: 加入weight的平方，L2 regularization.将入这个 regularization 后你的weight变得尽可能的小了。
2.  Lasso Regression: 加入weight的绝对值，L1 regularization. 加入后，会使得结果倾向于清除掉不重要的feature，它类似于
直接进行了feature selection并且倾向与得到一个sparse model
3.  Early Stopping: 提早停止

#### 降维：
1.  PCA: Principle Component Analysis: 无监督的降维算法
    1.  PCA的本质其实是将原始数据映射到一些方差较大的方向上。
    1.  PCA 最大方差理论
    2.  PCA 最小方差理论
    3.  
2.  LDA: Linear Discriminant Analysis: 有监督的降维算法
    1.  它服务于分类问题，尝试寻找一个投影方向使得投影狗的样本尽可能的按照原始的类别分开。
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

#### Prepare the data for machine learning 数据处理
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
2.  数据的处理
    1.  Deal with missing features: drop data one with missing feature; remove the feature completely;
    set the value to zero/mean/median
    ```
    housing.dropna(subset = ['total_bedrooms'])
    housing.drop('total_bedrooms', axis = 1)
    median = housing['total_bedrooms'].median()
    housing['total_bedrooms'].fillna(median, inplace = True)
    ```
    2.  Handling text and categorical attributes/feature: convert those to numbers
        1.  Most of machine learning algorithm needed to translate categorical feature to numbers. Only decision tree can accept string as an input.  
        2.  Type of convert:
            1.  One hot encoding (each entity is independent and no order):
                1.  For example: Blood Type A, B, AB, O:
                    1.  A --> [1, 0 ,0 ,0]
                    2.  B --> [0, 1, 0, 0]
                    3.  AB -> [0, 0, 1, 0]
                    4.  O --> [0, 0, 0, 1]
                2.  With one hot encoding, all input get an independent presentation. But cost too many space
                3.  使用稀疏向量能够节省空间，并且大多数的算法都接受稀疏向量作为input
                4.  或者使用降维的手段
            2.  Ordinal Encoding (each one related and have order, like score A is better than B, B is better than C)
            then 
                1.  A --> 3
                2.  B --> 2
                3.  C --> 1
            3.  Binary Encoding (使用二进制数来表示)：
                1.  A --> [0, 0, 0, 1]
                2.  B --> [0, 0, 1, 0]
                3. AB --> [0, 1, 1, 0]
                4.  O --> [1, 0, 0, 0]
 
    3.  Feature scaling/Normalization: help to get to the best answer faster. Almost all dx based machine learning algorithm
     need to use normalization. Others don't need, for example: decision tree/ naive bayes.
        1.  min_max scaling/ normalization; MinMaxScale
        2.  standardization scaling/Zero Score normalization (new data have zero mean and unit variance): StandardScaler
    
    4.  Transformation Pipelines: custom the process steps and for future use.
    
    5.  高维组合特征的处理：feature combine.
        1.  当feature的取值有恩多的时候，两个feature 的组合便有 m * n 个组合，如果使用on hot encoding的话那么就会造成严重的问题。
        所以需要降维来处理。例如购买物品是，用户ID 有千万个，物品ID也有千万个，而y就是用户是否点击，如果这样的话就造成了 m * n 的组合
        而这个组合的结果是非常大的。
        2.  故而我们将 feature combine, 我们的 X 中只记录： x1 = 用户ID = a 并点击了物品 ID == b 的情况。这样就删除了那些
        data，例如用户A 没有点击物品 B的情况
    
    6. 文本表示模型：
        1.  Bag of word: TF-IDF
            1.  TF-IDF(t,d) = TF(t,d) * IDF(t):
                1.  单词 t 在文档 d中出现的频率
                2.  单词 t 在所有的文档中出现的次数的反比 = log(文章总数/(包含单词t的文章总数 + 1))
        2.  N-gram: 考虑单词的前后连贯性，将 N 个单词作为一个 word t 来处理，这样也能得到一个结果。
        2.  Topic model: 找到具有代表性的文章然后做词频统计
        3.  World Embedding: Word2Vec: 最常用的词嵌入模型。
    
    7.  Word2Vec 是一种浅层神经网络，有两种网络结构：
        1.  CBOW: continuous bog of words: 通过上下文来预测当前词的生成概率。即输入是上下文，输出是当前词
        2.  Skip-gram: 根据当前的词来预测上下文中各词的生成概率。即输入是当前词，输出是当前词
    
    8.  图像处理中数据量不足的问题及解决方法：
        1.  问题：过拟合
        2.  解决方案：
            1.  在模型上：简化模型，Regularization, 集成学习，dropout
            2.  在数据上：Data Augmentation (图像的随机旋转，平移，缩放，裁剪)；对图像增加噪音扰动（高斯白噪声）
            颜色变换，改变亮度、清晰度、对比度等。
            3.  迁移学习 transfer learning: 用一个在大规模数据集上预训练好的通用模型做基础然后再针对现在的小的data进行微调。
    9.  
        
        
#### Select and train a Model:
Better Evaluation using cross-validation:
1.  Split the training set into a smaller training set and validation set.
2.  Train your model base on training set and evaluate against the validation set
3.  Easy way is to use sklearn's cross-validation feature.
        
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)
        
#### Fine Tune Your Model:
1.  Grid Search: automatic try different combination of hyperparameter values using cross-validation.
2.  Randomized Search: if the search space is very large, then you should use this model.

#### Performance Measurement for classification
1.  T/F: True or False 代表的是在预测的对不对，如果对就是True
2.  P/N: Positive or Negative: 代表的是真正的分布情况中，它是否符合某条件。
例如医院检查是否患病，如果来人是病人的话，那就是 positive, 不是病了就是negative；医院可能检查出来了，然后检查出来有病的人
只有一部分是真的有病，这部分人就是 True Positive, 相应的本来没病查出病了的就是 False Negative; 有病没查出来的就是
False Positive; 没病也没查出病的就是 True Negative
3.  Confusion Matrix: 
    1.  [True Negative, False Positive] 
    2.  [False Negative, True Positive]
4.  Precision and recall: 
    1.  Precision: TP/(TP + FP): 有病的人中有多少被检查出来了
    2.  Recall: TP/(TP + PN)：查出来有病的人有多少是真的有病
5.  F-score: F1 = 2 * (1/precision + 1/recall)
6.  ROC Curve: used with binary classifiers. plot TP/FP

#### Performance Measurement for linear regression:
1.  learning curve, plot 在training set上和validation set上的RMSE. 
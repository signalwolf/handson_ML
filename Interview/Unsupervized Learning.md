#非监督学习  
[K均值聚类](#K均值聚类)  
[高斯混合模型](#高斯混合模型)  
[聚类算法的评估](#聚类算法的评估)  

## K均值聚类: K mean clustering
1.  基本思想：通过迭代的方法寻找K个cluster, 使得聚类的结果对应的cost最小。最简单的cost就是各个样本距离其簇中心的误差平方和。
2.  基本步骤：
    1.  数据预处理：归一化，离群点处理
    2.  随机选取k个簇中心，不一定是data的点
    3.  计算cost =
    4.  对每个样本计算其到 K个簇中心的距离并由此判断其属于哪一个簇
    5.  对每个簇，重新计算当前的簇的中心
    6.  重复步骤3 - 5 直至cost收敛
3.  优缺点：
    1.  缺点：
        1.  最开始的K值需要手工选择，而K个分簇可能不符合分布
        1.  分簇的结果收到初始的预设的簇中心影响较大，
        2.  簇中心收到离群点的影响较大：因为可能一个离群点会导致中心偏离
        3.  由于1、2，经常导致每次的结果都不稳定从而出现局部最优解的情况
        4.  无法解决簇样本数量相差较大的情况，例如一个是另一个的100被的话就是导致cost的计算出现问题
    2.  优点：
        1.  计算复杂度是O(NKt): N:数据量，K:簇数量，t:iteration次数
        2.  虽然经常以局部最优结束，但是一般情况下能达到局部最优就可以
4.  优化：
    1.  数据预处理：归一化以及离群点的处理
    2.  合理选择K的值：plot K 与 最后的cost之间的关系。
        1.  一般认为拐点处的K值就是最好的选择。但是需要人工干预不能自动化
        2.  Gap Statics: 自动化的选择K值
    3.  加入核函数：
        1.  因为欧式距离往往不是最好的选择。它最后只会形成一个球型，而这往往不是最好的情况。
    2.  K mean 的优化算法：
        1.  K mean ++: 优化的是最最最开始随机选点；由于最开始的随机选点可能非常近，这样就导致了问题；K mean ++ 的优化就在于此，它
所尝试在做的就是在产生第 i 个点的时候考虑前面的点，离前面所有的聚类中心点越远的点越是有可能被选择到
        2.  ISODATA算法：优化的是K的值得大小，如果我们发现大量的data聚集在一起，那么就尝试split簇，当我们发现某个簇周围非常少的点
被划入的话，就尝试去除掉当前的簇。
            1.  参数一：预计的聚类数量 K；算法最后产生的簇数将会是 K/2 -- 2K
            2.  参数二：每个簇所要求的最少样本数：Nmin
            3.  参数三：最大方差 sigma: 当某个簇的方差超过这个值得时候就进行分裂
            4.  参数四：两个聚类中心之间的最小的距离，当两个聚类中心非常近的时候，就尝试合并两个簇
## 高斯混合模型
## 聚类算法的评估
1.  估计聚类趋势：霍普金斯统计量, Hopkins statics
    1.  在所有的样本中随机取N点，然后计算其与最近的样本之间的距离，记为 Xn
    2.  在样本的取值空间中随机取N点，然后计算与其最近的样本之间的距离，记为 Yn
    3.  如果样本的随机性比较大的话，那么Xn之和会非常的接近Yn; 反之，Yn 应当远大于 Xn.
    4.  计算方式，当值接近于 0.5时认为随机性比较大；反之认为数据有趋势
![image](https://github.com/signalwolf/handson_ML/blob/master/Interview/Image/Screen%20Shot%202018-11-13%20at%208.33.16%20PM.png)  
2.  判断数据簇数：
    1.  利用手肘法找到最合适的K值
3.  判断聚类的质量：
    1.  轮廓系数：对于给定的某点，其轮廓系数为
![image](https://github.com/signalwolf/handson_ML/blob/master/Interview/Image/Screen%20Shot%202018-11-13%20at%208.33.33%20PM.png)  
        1.  a(p)是p到本簇其他点之间的平均距离（本簇的紧凑程度）；b(p)是p到不同簇的最小平均距离（到邻近簇的分离程度）。
        2.  因此这样计算了聚类的质量是不是好（本簇紧凑而且离邻近簇比较远），average所有的点后就可以计算出聚类的效果了
    2.  RMSSTD: 均方根标准偏差
![image](https://github.com/signalwolf/handson_ML/blob/master/Interview/Image/Screen%20Shot%202018-11-13%20at%208.33.41%20PM.png)  
        1.  下面的方程相当于 P * (N - K). P 代表数据的维度。其实就是归一化的簇距离。它只考虑了到本簇中心的距离
    3.  R-Square: 聚类的差异度：
![image](https://github.com/signalwolf/handson_ML/blob/master/Interview/Image/Screen%20Shot%202018-11-13%20at%208.33.57%20PM.png)    
        |x-c|**2 代表的是如果只分一簇（也就是不分）的话的loss，另一个就是当前分法的loss，故而这个式子计算的是相比于不分所带来的gain

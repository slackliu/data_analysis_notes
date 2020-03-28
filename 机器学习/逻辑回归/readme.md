## sklearn中的逻辑回归
from  sklearn.liner_model  import  LogisticRegression 

| 逻辑回归相关的类 | 说明 | 
| --------------- | --- |
| linear_model.LogisticRegression | 逻辑回归分类器(最大熵分类器) |
| linear_model.LogisticRegressionCV | 带交叉验证的逻辑回归分类器 | 
| linear_model.logistic_regression_path | 计算Logistic回归模型以获得正则化参数的列表 | 
| linear_model.SGDClassifier | 利用梯度下降求解的线性分类器(SVM, 逻辑回归等) |
| linear_model.SGDRegressor | 利用梯度下降最小化正则化后的损失函数的线性回归模型 | 
| metrics.log_loss |对数损失, 又称为逻辑损失或交叉熵损失 |
| metrics.confusion_matrix | 混淆矩阵, 模型评估指标之一 |
| metrics.roc_auc_score | ROC曲线, 模型评估指标之一 | 
| metrics.acuracy_score |精确性, 模型评估指标之一 | 


#### 主要参数介绍:
__penalty:__ 惩罚项,取值为l1或者l2,默认为l2,当模型满足高斯分布时(正太分布),使用l2,当模型参数满足拉普拉斯分布时,使用l1。

__solver:__ 代表的是逻辑回归损失函数的优化方法。有五个参数可选，分别是liblinear、lbfgs、newton-cg、sag和saga。
默认为liblinear，使用与数量小的数据集，当数据量很大时可以选用sag或者saga方法。  

__max_iter:__ 算法收敛的最大迭代次数,默认为10.

__n_jobs:__ 拟合和预测的时候cpu的核数, 默认是1,也可以使整数,如果是-1表示使用机器所有的cpu核数。

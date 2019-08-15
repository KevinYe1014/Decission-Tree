import  matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets.california_housing import fetch_california_housing
housing = fetch_california_housing()
# print(housing.DESCR)
# 查看数据
# print(housing.data.shape) # (20640, 8)
# print(housing.data[0])
# print(housing.target) # [4.526 3.585 3.521 ... 0.923 0.847 0.894]
# print(housing.feature_names)


# 决策树，但是特征只考虑经纬度
from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth= 2 )
dtr.fit(housing.data[:, [6, 7]], housing.target)
'''
参数说明：
DecisionTreeRegressor(criterion='mse', max_depth=2, max_features=None,
           max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')
'''
# 要可视化显示，首先需要安装 graphviz
dot_data = tree.export_graphviz(dtr,
                                out_file=None, feature_names=housing.feature_names[6:8],
                                filled=True, impurity=False, rounded=True
                                )
# pip install pydotplus
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#FFF2DD')
from IPython.display import Image
Image(graph.create_png())
graph.write_png('dtr_white_background.png')


# 决策树，但是特征考虑所有的。
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    housing.data, housing.target, test_size=0.1, random_state=4
)
dtr = tree.DecisionTreeRegressor(random_state=42)
dtr.fit(data_train, target_train)

dtr.score(data_test, target_test)


# 随机森林 特征考虑所有的。
from  sklearn.ensemble import RandomForestRegressor
rft = RandomForestRegressor(random_state= 42)
rft.fit(data_train, target_train)
rft.score(data_test, target_test)

from sklearn.model_selection import GridSearchCV
tree_param_grid = {'min_samples_split': list((3, 6, 9)), 'n_estimators':list((10, 50, 100))}
grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print('Best Params: ', grid.best_params_) # 6, 100
print('Best Score: ', grid.best_score_) # 0.8..
'''
作用：返回该次预测的系数R2    
  其中R2 =（1-u/v）。
  u=((y_true - y_pred) ** 2).sum()     v=((y_true - y_true.mean()) ** 2).sum()
其中可能得到的最好的分数是1，并且可能是负值（因为模型可能会变得更加糟糕）。当一个模型不论输入何种特征值，其总是输出期望的y的时候，此时返回0。
'''

rfr = RandomForestRegressor(min_samples_split=6, n_estimators=100, random_state=4)
rfr.fit(data_train, target_train)
rfr.score(data_test, target_test)

print(pd.Series(rfr.feature_importances_, index=housing.feature_names).sort_values(ascending=False))


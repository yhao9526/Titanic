# 引入pandas库
import pandas as pd
import random
import statsmodels.api as sm


# 读取本地 CSV 文件
df = pd.read_csv('D:\_Kaggle/1titanic/rawdata/train.csv')
# 显示前几行数据
print(df.head(5))

# 删除列，保留所需变量列表，并另存为新的表格
df_selected = df.drop(columns=['Name','Ticket'])
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train1.csv', index=False)
# 替换 "male" 为 1，"female" 为 0
df_selected['Sex'] = df_selected['Sex'].replace({'male': 1, 'female': 0})
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train2.csv', index=False)
# 船舱部分进行编号
df['First_Letter'] = df_selected['Cabin'].astype(str).str[0]  # 取第一个字母
unique_letters = sorted(df['First_Letter'].dropna().unique())  # 按字母顺序,去除空值并排序
letter_mapping = {letter: idx + 1 for idx, letter in enumerate(unique_letters)}  # 生成映射表
df_selected['Cabin'] = df['First_Letter'].map(letter_mapping).fillna(0).astype(int) # 映射编号
df.drop(columns=['First_Letter'], inplace=True) # 删除辅助列
df_selected['Cabin'] = df_selected['Cabin'].replace(9, 0)  # 将里面填充空值的9，替换为0
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train3.csv', index=False)
# 上船地点进行编号，C=1，Q=2，S=3，else=0
df_selected['Embarked'] = df_selected['Embarked'].map({'C': 1, 'Q': 2, 'S': 3}).fillna(0).astype(int)
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train4.csv', index=False)
# 针对行李价格所在区间进行分类
df_selected['Fareplus'] = (df_selected['Fare'] // 10).rank(method='dense').astype(int) - 1 # 按照 10 为一组，划分类别
df_selected.drop(columns=['Fare'], inplace=True) # 删除原列
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train5.csv', index=False)
# 填充年龄空缺值，并排序
df_selected['Nuage'] = (df_selected['Parch'] * df_selected['Embarked']) # 检查发现，空值绝大多数没有带父母孩子，可认为年龄分布在30-60
df_selected.drop(columns=['Nuage'], inplace=True) # 删除拓展列
def fill_with_random(x):
    if pd.isna(x):
        return random.randint(30, 60)
    return x # 定义填充函数
df_selected['Age'] = df_selected['Age'].apply(fill_with_random) # 填充指定列的空值
df_selected['Ageplus'] = (df_selected['Age'] // 10).rank(method='dense').astype(int) - 1 # 按照 10 为一组，划分类别
df_selected.drop(columns=['Age'], inplace=True) # 删除原列
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train6.csv', index=False)

# 普通线性拟合
X = df_selected.drop(['Survived', 'PassengerId'], axis=1) # 去除因变量列和序号列，剩下为自变量
y = df_selected['Survived'] # 选择因变量
X = sm.add_constant(X) # 添加常数项
model = sm.OLS(y, X) # 创建线性回归模型
results = model.fit() # 拟合模型
output_path = 'D:\_Kaggle/1titanic/tempdata/output1.txt'  # 指定输出文件路径
with open(output_path, 'w') as f:
    f.write(results.summary().as_text()) # 文件操作完成后自动关闭该文件
# 回归显示，parch和Fareplus不显著，去除这两个变量重新回归
X = df_selected.drop(['Survived', 'PassengerId','Parch','Fareplus'], axis=1) # 去除因变量列和序号列，剩下为自变量
y = df_selected['Survived'] # 选择因变量
X = sm.add_constant(X) # 添加常数项
model = sm.OLS(y, X) # 创建线性回归模型
results = model.fit() # 拟合模型
output_path = 'D:\_Kaggle/1titanic/tempdata/output2.txt'  # 指定输出文件路径
with open(output_path, 'w') as f:
    f.write(results.summary().as_text()) # 文件操作完成后自动关闭该文件

# Survived = -0.1327 * Pclass - 0.4994 * Sex - 0.0387 * SibSp + 0.0266 * Cabin - 0.0469 * Embarked - 0.0373 * Ageplus + 1.2376

# 查看预测概率情况
df_selected['Survivedplus'] = (-0.1327*df_selected['Pclass'] - 0.4994*df_selected['Sex'] - 0.0387*df_selected['SibSp'] + 0.0266*df_selected['Cabin'] - 0.0469*df_selected['Embarked'] - 0.0373*df_selected['Ageplus'] + 1.2376)
df_selected = df_selected[df_selected['Survived'] != 0] # 删除列中值为 0 的行
df_selected['p0'] = (df_selected['Survivedplus'] - (df_selected['Survivedplus']*(1-df_selected['Survivedplus'])/891)**0.5)
df_selected.to_csv('D:\_Kaggle/1titanic/tempdata/train7.csv', index=False)
# 计算 col1 列的平均数
average = df_selected['p0'].mean()
print(f"p0 列的平均数是: {average}")

# p0_hat = 0.5854717319617528

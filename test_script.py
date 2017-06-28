import pandas as pd
import numpy as py
import tensorflow as tf

train = "train.csv"
test = "test.csv"

# read csv files
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)

# separate columns by dtype
columns = list(df_train)
categorical_columns = columns[2:10]
binary_columns = list(filter(lambda x: x not in categorical_columns 
	and x not in ["y","row"],columns))

# replace categorical columns with dummy variables
df_train_one_hot = df_train[binary_columns+["y"]]

for col in categorical_columns:
	dummy_temp = pd.get_dummies(df_train[col])
	dummy_colnames = list(dummy_temp)
	dummy_temp.columns = [col + "_" + x for x in dummy_colnames]
	df_train_one_hot = pd.concat([df_train_one_hot,dummy_temp], 
		axis = 1 ,join_axes = [df_train_one_hot.index])

# fully connected NN
features = df_train_one_hot.drop("y",1)
features = features.astype(np.float32)

labels = df_train_one_hot["y"]
labels = labels.astype(np.float32)

sample_size = len(features)
feature_size = len(features.columns)

W1 = tf.Variable(tf.truncated_normal([feature_size,10],stddev = 0.1))
W2 = tf.Variable(tf.truncated_normal([10,1],stddev = 0.1))
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([1]))

x = tf.constant(features)
h1 = tf.matmul(x,W1)+b1
y = tf.matmul(h1,W2)+b2
y_ = tf.constant(labels)

target_fun = tf.reduce_mean(tf.squared_difference(y,y_))

train_step = tf.train_GradientDescentOptimizer(0.5).minimize(target_fun)

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for _ in range(1000):
		sess.run(train_step)

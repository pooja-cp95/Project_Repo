import pandas as pd
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from pandas_profiling import ProfileReport

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape and convert to DataFrame
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
columns = [f"pixel_{i}" for i in range(28*28)]
df_train = pd.DataFrame(x_train, columns=columns)
df_train['label'] = y_train
df_test = pd.DataFrame(x_test, columns=columns)
df_test['label'] = y_test
df = pd.concat([df_train, df_test], axis=0)

# Generate EDA report
profile = ProfileReport(df, title="Fashion MNIST Dataset EDA Report", explorative=True)
profile.to_file("fashion_mnist_eda_report.html")
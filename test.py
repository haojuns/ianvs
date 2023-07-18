import pickle

# 指定字典文件的路径
file_path = './workspace/benchmarkingjob/rfnet_lifelong_learning/c50b5ca4-222b-11ee-ba17-a906087290a8/output/train/1/index.pkl'



# 读取字典文件
dictionary = pickle.loads(file_path)

# 打印读取到的字典
print(dictionary)

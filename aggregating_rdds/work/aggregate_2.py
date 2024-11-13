from pyspark import SparkContext
import os
import json
import sys
import time

# set environment variables since im working in a conda environment
# from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

data_path = os.path.join('..', 'resource', 'asnlib', 'publicdata')
file_path = os.path.join(data_path, 'review.json')
out_path = 'output2.json'

sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")
# A read JSON file with output as RDD
def spark_read_json(f_path):
    
    rdd = sc.textFile(f_path).map(lambda x: json.loads(x))
    return rdd

rdd = spark_read_json(file_path)
# create a decorator to calculate time for any function called
def exec_time_decorator(func):

    def wrapper(*args):
        start = time.time()
        out = func(*args)
        end = time.time()
        return end - start
    
    return wrapper

@exec_time_decorator
# create a function to call faster
def T1QF(rdd):
    top10_business = rdd.map(lambda x: (x['business_id'], 1)
                            ).reduceByKey(lambda x, y: x+y
                                        ).sortBy(lambda x: (-x[1], x[0])
                                                    ).take(10)
    return top10_business


def evaluate_task(func, rdd):
    n_partitions = rdd.getNumPartitions()
    n_items = rdd.mapPartitions(lambda x: [len(list(x))]).collect()
    exec_time = func(rdd)

    return n_partitions, n_items, exec_time

def_n_part, def_n_items, def_exe_time = evaluate_task(T1QF, rdd)

default_dict = {'n_partition': def_n_part, 
                'n_items': def_n_items,
                'exe_time': def_exe_time}

'''
custom partitioning function:
logic is:
since it just hashes the key of the textFile
I want to hash by business_id key to do the shuffling first
This will cause similar business_id to be at same partitions
'''
def customPartitionFunc(rdd, n_partitions):
    return rdd.keyBy(lambda x: x.get('business_id', '')).partitionBy(numPartitions=n_partitions)

n_partitions = 10
new_rdd = spark_read_json(file_path)
custom_rdd = customPartitionFunc(new_rdd, n_partitions)

# modify code to avoid shuffling
@exec_time_decorator
def better_shuffle_T1QF(rdd):
    top10_business = rdd.map(lambda x: (x[0], 1)
                            ).reduceByKey(lambda x, y: x + y
                                         ).sortBy(lambda x: (-x[1], x[0])
                                                  ).take(10)
    return top10_business
cust_n_part, cust_n_items, cust_exe_time = evaluate_task(better_shuffle_T1QF, custom_rdd)

custom_dict = {'n_partition': cust_n_part, 
                'n_items': cust_n_items,
                'exe_time': cust_exe_time}

output_dict = {'default': default_dict, 'customized': custom_dict}

print(output_dict)

'''
Code to check if output/result is same!
def_res = rdd.map(lambda x: (x['business_id'], 1)
                            ).reduceByKey(lambda x, y: x+y
                                        ).sortBy(lambda x: (-x[1], x[0])
                                                    ).take(10)
cust_res = custom_rdd.map(lambda x: (x[0], 1)
                            ).reduceByKey(lambda x, y: x + y
                                         ).sortBy(lambda x: (-x[1], x[0])
                                                  ).take(10)

print('\n')
print(def_res)
print('\n')
print(cust_res)
print(def_res == cust_res)
IT IS THE SAME RESULT! WITH SIGNIFICANTLY FASTER EXEC TIME!
'''

with open(out_path, 'w') as f_out:
    json.dump(output_dict, f_out)
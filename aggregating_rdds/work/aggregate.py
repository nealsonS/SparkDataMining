from pyspark import SparkContext
import os
import json
import sys
from datetime import datetime

# set environment variables since im working in a conda environment
# from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

data_path = os.path.join('..', 'resource', 'asnlib', 'publicdata')
file_path = os.path.join(data_path, 'test_review.json')
out_path = 'output1.json'

sc = SparkContext('local[*]', 'task1')

def spark_read_json(f_path):
    
    rdd = sc.textFile(f_path).map(lambda x: json.loads(x))
    return rdd

rdd = spark_read_json(file_path)

def add(x, y):
    return x+y

n_review = rdd.count()

# B set datetime format to parse
dt_format = '%Y-%m-%d %H:%M:%S'
n_review_2018 = rdd.map(lambda x: datetime.strptime(x['date'], dt_format).year).filter(lambda year: year == 2018).count()

# C get distinct user ids
n_user = rdd.map(lambda x: x['user_id']).distinct().count()

# D top 10 users review count
top10_user = rdd.map(lambda x: (x['user_id'], 1)
                      ).reduceByKey(add
                                    ).sortBy(lambda x: (x[1], x[0]), ascending=[False, True]
                                             ).take(10)

# E business count
n_business = rdd.map(lambda x: x['business_id']).distinct().count()

# F top 10 businesses
top10_business = rdd.map(lambda x: (x['business_id'], 1)
                         ).reduceByKey(add
                                       ).sortBy(lambda x: (x[1], x[0]), ascending=[False, True]
                                                ).take(10)

print((n_review, n_review_2018, n_user, top10_user, n_business, top10_business))
json_output = {'n_review': n_review, 
               'n_review_2018': n_review_2018, 
               'top10_user': top10_user, 
               'n_business': n_business, 
               'top10_business': top10_business}

with open(out_path, 'w', encoding = 'utf-8') as f_out:
    json.dump(json_output, f_out)

sc.stop()


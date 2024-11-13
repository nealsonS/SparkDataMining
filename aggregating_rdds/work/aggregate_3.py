from pyspark import SparkContext
import os
import json
import sys
import time

# set environment variables since im working in a conda environment
# from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext('local[*]', 'task3')
sc.setLogLevel("ERROR")
# data folder path
data_path = os.path.join('..', 'resource', 'asnlib', 'publicdata')

# file paths
rev_path = os.path.join(data_path, 'test_review.json')
bun_path = os.path.join(data_path, 'business.json')
outA_path = 'output3A.txt'
outB_path = 'output3B.json'

def spark_read_json(f_path):
    
    rdd = sc.textFile(f_path).map(lambda x: json.loads(x))
    return rdd

b_rdd = spark_read_json(bun_path)
r_rdd = spark_read_json(rev_path)

# A. 
# get k-v pairs of business id and stars
bus_id_stars = r_rdd.map(lambda x: (x['business_id'], x['stars']))

# get business_id and their cities
bus_id_city = b_rdd.map(lambda x: (x['business_id'], x.get('city', None)))

# join the rdd's to get k-v pair of city, stars then get the average
stars_avg = bus_id_city.join(bus_id_stars).map(lambda x: (x[1][0], x[1][1])).groupByKey().mapValues(lambda x: sum(x)/ len(x))

# sort the stars_avg
sorted_stars_avg = stars_avg.sortBy(lambda x: (-x[1], x[0])).collect()

with open(outA_path, 'w', encoding = 'utf-8') as f_out:
    f_out.write('city,stars\n')

    for i in sorted_stars_avg:
        f_out.write(','.join(str(x) for x in i) + '\n')

# B
# create a decorator to calculate time for any function called
def exec_time_decorator(func):

    def wrapper(*args):
        start = time.time()
        out = func(*args)
        end = time.time()
        return end - start
    
    return wrapper

@exec_time_decorator
# make task A as a function to apply decorator
def task_M1(bun_path, rev_path):

    b_rdd = spark_read_json(bun_path)
    r_rdd = spark_read_json(rev_path)

    # A. 
    # get k-v pairs of business id and stars
    bus_id_stars = r_rdd.map(lambda x: (x['business_id'], x['stars']))

    # get business_id and their cities
    bus_id_city = b_rdd.map(lambda x: (x['business_id'], x.get('city', None)))

    # join the rdd's to get k-v pair of city, stars then get the average
    stars_avg = bus_id_city.join(bus_id_stars).map(lambda x: (x[1][0], x[1][1])).groupByKey().mapValues(lambda x: sum(x)/ len(x))

    # sort and print
    s_avg_list = sorted(stars_avg.collect(), key = lambda x: -x[1])

    # take top 10 and print it
    print([x[0] for x in s_avg_list][:10])

@exec_time_decorator
def task_M2(bun_path, rev_path):
    b_rdd = spark_read_json(bun_path)
    r_rdd = spark_read_json(rev_path)

    # A. 
    # get k-v pairs of business id and stars
    bus_id_stars = r_rdd.map(lambda x: (x['business_id'], x['stars']))

    # get business_id and their cities
    bus_id_city = b_rdd.map(lambda x: (x['business_id'], x.get('city', None)))

    # join the rdd's to get k-v pair of city, stars then get the average
    stars_avg = bus_id_city.join(bus_id_stars).map(lambda x: (x[1][0], x[1][1])).groupByKey().mapValues(lambda x: sum(x)/ len(x))

    # take top 10 stars rom stars_avg
    print([i[0] for i in stars_avg.takeOrdered(10, key=lambda x: -x[1])])

M1_res = task_M1(bun_path, rev_path)
M2_res = task_M2(bun_path, rev_path)
reason = """This is because Python doesn't distribute the sorting task to many different clusters 
but perform on one machine only. As a result, it's not optimized to deal with long lists of values.
However for task 2: Spark allows distributed parallel computing and it will split the tasks to different clusters.
Thus it's optimized for large datasets as you can perform tasks simultaneously and perform computations of a whole dataset really quickly because of that."""

out_dict = {'m1': M1_res, 'm2': M2_res, 'reason': reason}
with open(outB_path, 'w') as f_out:
    json.dump(out_dict, f_out)
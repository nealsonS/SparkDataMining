from pyspark import SparkContext
import os
import sys
import time

# functions
def read_csv(f_path, header_row = False):

    rdd = sc.textFile(f_path).map(lambda row: row.strip().split(','))

    # convert to string number to int
    #rdd = rdd.map(lambda row_list: [int(x) if x.replace('.', '').isnumeric() else x for x in row_list])
    

    if header_row:
        # get header
        header = rdd.first()
        rdd = rdd.filter(lambda r_list: r_list!=header)

    return rdd


def calc_weights(rdd):

    bus_user_ratings = rdd.map(lambda x: 
                           (x[0][1], (x[0][0], x[1]))
                           )
    
    '''avg_coratings_dict = sc.broadcast(dict(bus_user_ratings.map(lambda x: (x[0], x[1][1])).aggregateByKey(
        (0,0),
        lambda a,b: (a[0] + b, a[1] + 1),
        lambda a,b: (a[0] + b[0], a[1] + b[1])
    ).mapValues(lambda x: x[0]/x[1]).collect()))'''
    
    bus_pairs = bus_user_ratings.join(bus_user_ratings)
    item_pair_ratings = bus_pairs.map(lambda x: ())

    

    
    #bus_pairs = bus_user_ratings.join(bus_user_ratings)


    
    return item


    


if __name__ == '__main__':
    start = time.time()

    # set environment variables since im working in a conda environment
    # from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.path.dirname(sys.executable)

    sc = SparkContext('local[*]', 'hw3_task21')
    sc.setLogLevel('ERROR')

    TRAIN_PATH = 'yelp_train.csv'
    TEST_PATH = 'yelp_val.csv'
    OUT_PATH = 'out_2_1.csv'

    train_rdd = read_csv(TRAIN_PATH, header_row=False).map(lambda x: ((x[0], x[1]), float(x[2])))

    weights_rdd = calc_weights(train_rdd)

    print(weights_rdd.take(5))

    end = time.time()
    print(f'Duration: {end-start}')
    sc.stop()









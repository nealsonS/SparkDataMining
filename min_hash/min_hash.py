from pyspark import SparkContext
import os
import sys
import time
import random
from itertools import combinations

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

def convert_to_characteristic_matrix(rdd):

    return rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)

def generate_hash_functions(num_functions, m):

    '''Generate n lambda functions of type:
    ((a*x + b) % 31) % m
        where 31 is a large prime number'''
    
    hash_list = []
    p = 31

    for _ in range(num_functions):
        a = random.randint(3, m-1)
        b = random.randint(2, m-1)
        hash_list.append(lambda x, a=a, b=b, p=p, m=m: ((a*x +b) % p) % m)

    return hash_list

def min_hash(item_list, hash_function):
    hash_result = []

    for h in hash_function:

        min_hash_value = float('inf')

        for i in item_list:
            user_id_row_number = broad_user_id_dict.value.get(i, None)
            hash_res = h(user_id_row_number)

            if hash_res < min_hash_value:
                min_hash_value = hash_res

        hash_result.append(min_hash_value)

    return hash_result

def create_signature_matrix(rdd, hash_functions):
    return rdd.mapValues(lambda x: min_hash(x, hash_functions))

def divide_into_bands(row, b, r):

    b_id = row[0]
    sig_vec = list(row[1])

    hash_band = []
    for i in range(b):
        band = sig_vec[(i*r) : (i*r+r)]
        hash_band.append((i, hash(tuple(band)), b_id))


    return hash_band

def create_candidate_pairs(rdd, b, r):
    band_rdd = rdd.flatMap(lambda x: divide_into_bands(x, b, r)).map(lambda x: ((x[0], x[1]), x[2]))
    group_band_rdd = band_rdd.groupByKey().filter(lambda x: len(x[1]) > 1) # to make sure that there is more than one similar pairs

    cand_rdd = group_band_rdd.mapValues(list).flatMap(lambda x: [sorted(pair) for pair in combinations(x[1], 2)]).map(tuple).distinct()
    return cand_rdd

def calc_cand_jacc_sim(cand_rdd, char_rdd):
    def calc_jacc_sim(pair):
        b1 = pair[0]
        b2 = pair[1]

        users1 = broad_char_dict.value[b1]
        users2 = broad_char_dict.value[b2]
        jacc_sim = len(users1 & users2) / len(users1 | users2)
        return jacc_sim
    
    jacc_sim_rdd = cand_rdd.map(lambda x: (x,calc_jacc_sim(x)))

    return jacc_sim_rdd

if __name__ == '__main__':
    start = time.time()
    # set environment variables since im working in a conda environment
    # from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.path.dirname(sys.executable)

    random.seed(42)
    IN_PATH = 'yelp_train.csv'
    OUT_PATH = 'out_task1.csv'

    sc = SparkContext('local[*]', 'hw3_task1')
    sc.setLogLevel('ERROR')

    rdd = read_csv(IN_PATH, header_row=False)

    unique_user_id = rdd.map(lambda x: x[0]).distinct().collect()
    user_id_dict = {user: i for i, user in enumerate(unique_user_id)}
    broad_user_id_dict = sc.broadcast(user_id_dict)

    char_rdd = convert_to_characteristic_matrix(rdd)
    broad_char_dict = sc.broadcast(dict(char_rdd.collect()))

    m = 100 # number of columns
    num_hash_functions = 100
    b = 10
    r = num_hash_functions // b
    s = 0.5 # sim threshold

    hash_function_list = generate_hash_functions(num_hash_functions, m)
    sig_rdd = create_signature_matrix(char_rdd, hash_function_list)
    candidate_rdd = create_candidate_pairs(sig_rdd, b, r)
    jacc_sim_rdd = calc_cand_jacc_sim(candidate_rdd, char_rdd)
    filtered_jacc_sim = jacc_sim_rdd.filter(lambda x: x[1] > s).sortBy(lambda x: str(x[0]))

    result = filtered_jacc_sim.collect()
    
    with open(OUT_PATH, 'w') as f_out:
        
        f_out.write('business_id_1, business_id_2, similarity\n')
        
        for row in result:
            f_out.write(f'{row[0][0]},{row[0][1]},{row[1]}\n')

    end = time.time()
    print(f'Duration: {end-start}')

    sc.stop()
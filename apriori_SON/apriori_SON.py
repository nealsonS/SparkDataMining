from pyspark import SparkContext
from itertools import combinations
import time
import sys

def read_csv(f_path, header_row = False):

    rdd = sc.textFile(f_path).map(lambda row: row.strip().replace('"', '').split(','))

    if header_row:

        # get header
        header = rdd.first()
        rdd = rdd.filter(lambda r_list: r_list!=header)

    return rdd

def process_data(rdd):

    new_rdd = rdd.map(lambda row: (str(row[0][:5] + row[0][-2:] + '-' + row[1]), str(row[5])))
    return new_rdd


def out_csv(rdd, out_path):
    
    with open(out_path, 'w') as f_out:
        f_out.write("DATE-CUSTOMER_ID, PRODUCT_ID\n")
        
        for row in rdd.collect():
            str_row = [str(x) for x in row]
            f_out.write(','.join(str_row) + '\n')
        
def filter_rdd(rdd, filt_thresh):
    
    g_rdd = rdd.groupByKey()
    
    count_rdd = g_rdd.mapValues(lambda x: len(list(x)))
    
    filt_keys = count_rdd.filter(lambda x: x[1] > filt_thresh).keys().collect()
    
    filt_rdd = g_rdd.filter(lambda x: x[0] in filt_keys).mapValues(list)
    
    return filt_rdd
def get_out_str(tuple_list, header):

    size_val_dict = {}

    for t in sorted(tuple_list):

        if not isinstance(t, (tuple, list)):
            t = (t,)
        
        size = len(t)
        sorted_t = sorted(t)
        if size not in size_val_dict:
            size_val_dict[size] = [sorted_t]
        else:
            size_val_dict[size].append(sorted_t)
    
    sorted_dict = {k: sorted(v) for k, v in size_val_dict.items()}

    str_list = []
    str_list.append(header + '\n')
    for i, x in enumerate(sorted_dict):
        if x == 1:
            x_str = ','.join("('" + str(t[0]) + "')" for t in sorted_dict[x])
        else:
            x_str = ','.join(str(tuple(t)) for t in sorted_dict[x])
        
        #if i != len(size_val_dict) - 1:
        x_str = x_str + '\n\n'

        str_list.append(x_str)

    return ''.join(str_list)
    
# generate candidates by union every two sets
# then keeping only ones that have k distinct values
def generate_candidates(freq_isets, k):
    cand_set = set()
    for i in range(len(freq_isets)):
        for j in range(i + 1, len(freq_isets)):
            comb_set = set(freq_isets[i]).union(freq_isets[j])
            if len(comb_set) == k:
                cand_set.add(tuple(sorted(comb_set)))
    return cand_set


def apriori(iterator, s, N):
    part_list = [x for x in iterator]
    part_size = len(part_list)
    p = part_size/N
    sp = s*p

    # isets with count: key: count
    iset_count = {}

    # container frequent itemsets list
    freq_isets = []

    # add count for every value in dict
    for _, basket in part_list:
        for item in basket:

            iset_count[(item,)] = iset_count.get((item,), 0) + 1

    filt_isetcount = [item for item, count in iset_count.items() if count >= sp]

    freq_isets.extend(filt_isetcount)

    gen_cand_freqisets = freq_isets
    k = 2
    while True:

        if not gen_cand_freqisets:
            break
        cand_set = generate_candidates(gen_cand_freqisets, k)

        # container itemset
        iset_count = {}

        # add count for each itemset
        for _, b in part_list:
            for cand in cand_set:
                if set(cand).issubset(b):
                    iset_count[cand] = iset_count.get(cand, 0) + 1

        new_filt_isetcount = [iset for iset, count in iset_count.items() if count >= sp]

        gen_cand_freqisets = new_filt_isetcount
        if not new_filt_isetcount:
            break

        freq_isets.extend(new_filt_isetcount)
        k = k + 1

    return iter(freq_isets)

def check_freq_part(iterator, broad_fiset, s):
    c_dict = {}
    
    for _, b in iterator:
        for iset in broad_fiset.value:
            if set(iset).issubset(set(b)):
                c_dict[iset] = c_dict.get(iset, 0) + 1
                
                
    out_list = [(iset, count) for iset, count in c_dict.items()]
    
    return out_list

def son_algorithm(rdd, s, out_path):
    N = rdd.count()

    # phase 1
    # partition and run apriori on each partition
    local_fiset = rdd.mapPartitions(lambda partition: apriori(partition, s, N))
    
    local_list = local_fiset.distinct().collect()

    # broadcast local i_sets to send to every map worker node
    broad_fiset = sc.broadcast(local_list)

    # phase 2
    global_fisets = (
        rdd.mapPartitions(lambda x: check_freq_part(x, broad_fiset, s))
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda x: x[1] >= s)
            .map(lambda x: x[0])
            .collect()
    )

    return local_list, global_fisets


if __name__ == '__main__':
    start = time.time()
    sc = SparkContext('local', 'HW2_task2')
    sc.setLogLevel('ERROR')

    # parameters
    f_path = sys.argv[3]
    out_csv_path = 'out2.csv'
    out_txt_path = sys.argv[4]
    filt_thresh = int(sys.argv[1])
    s = int(sys.argv[2])
    
    new_rdd =read_csv(f_path, header_row=True)
    proc_rdd = process_data(new_rdd)
    
    # filter threshold
    filt_rdd = filter_rdd(proc_rdd, filt_thresh)
    local_isets, global_isets = son_algorithm(filt_rdd, s)

    str1 = get_out_str(local_isets, 'Candidates:')

    str2 = get_out_str(global_isets, 'Frequent Itemsets:')
    
    out_str = str1 + str2

    with open(out_txt_path, 'w') as f_out:
        f_out.write(out_str)
    
    #out_csv(proc_rdd, out_csv_path)

        
        
    
    end = time.time()
    print(f'Duration: {end-start}')
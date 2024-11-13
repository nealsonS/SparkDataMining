import time
import numpy as np
import xgboost as xgb
import math
import os
import sys
import json
from pyspark import SparkContext

def read_csv(f_path, train=True):

    Xy = np.genfromtxt(f_path, delimiter=',',skip_header=True, dtype=str)
    
    if train:
        X = np.delete(Xy, 2, axis = 1)
        y = Xy[:, 2].astype(float)
        return X, y
    else:
        return Xy

def rdd_read_json(f_path):

    rdd = sc.textFile(f_path).map(lambda row: json.loads(row.strip()))
    return rdd

def get_label_encode_dict(list_arr):

    user_set = set()
    bus_set = set()
    
    for arr in list_arr:
        user_unique_set = set(arr[:, 0])
        bus_unique_set = set(arr[:, 1])

        user_set = user_set.union(user_unique_set)
        bus_set = bus_set.union(bus_unique_set)

    user_set = sorted(user_set)
    bus_set = sorted(bus_set)

    user_dict = {val: i for i, val in enumerate(user_set)}
    bus_dict = {val: i for i, val in enumerate(bus_set)}

    return user_dict, bus_dict

def encode_label(arr, u_dict, b_dict, train=True):
    def encode_helper(user, bus):
        u = u_dict[user]
        b = b_dict[bus]
        return u, b

    if train:
        return np.array([encode_helper(user, bus) for user, bus in arr]), u_dict, b_dict
    else:
        u_b_list = []

        for user, bus in arr:
            if user not in u_dict:
                u_dict[user] = len(u_dict)
            if bus not in b_dict:
                b_dict[bus] = len(b_dict)

            u_b_list.append(encode_helper(user, bus))


        return np.array(u_b_list), u_dict, b_dict

def RMSE(pred, truth):

    n = len(pred)
    inside = sum((pred - truth) ** 2) / n
    rmse = math.sqrt(inside)
    return rmse
'''
def pk_save_model(mod, MODEL_OUT_NAME):

    with open(MODEL_OUT_NAME, 'wb') as f_out:
        pickle.dump(model, f_out)

def pk_load_model(MODEL_OUT_NAME):

    with open(MODEL_OUT_NAME, 'rb') as f_out:
        mod = pickle.load(f_out)

    return mod
'''
if __name__ == '__main__':

    start = time.time()

    # set environment variables since im working in a conda environment
    # from: https://stackoverflow.com/questions/48260412/environment-variables-pyspark-python-and-pyspark-driver-python
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.path.dirname(sys.executable)
    '''
    EXTRA FEATURES:

    business.json
        attributes to get:
        stars
        review_count
        is_open

    users.json
        stars --> groupBy then get mean
        average business cool funny useful
        average user cool funny useful
        user number of reviews

    tip.json
        number of tips --> groupby by business then get count of reviews

    photo.json
        number of photos in business
    '''
    #FOLDER_PATH = os.path.join('..', 'resource', 'asnlib', 'publicdata')
    #IN_PATH = os.path.join('..', 'resource', 'asnlib', 'publicdata', 'yelp_train.csv')
    #VAL_PATH = os.path.join('..', 'resource','asnlib', 'publicdata', 'yelp_val_in.csv')
    
    sc = SparkContext('local[*]', 'hw3_2_2')
    sc.setLogLevel('ERROR')

    '''FOLDER_PATH = sys.argv[1]
    IN_PATH = os.path.join(FOLDER_PATH, 'yelp_train.csv')
    VAL_PATH = sys.argv[2]
    OUT_PATH = sys.argv[3]'''
   
    FOLDER_PATH = 'HW3StudentData'
    IN_PATH = os.path.join(FOLDER_PATH, 'yelp_train.csv')
    VAL_PATH = os.path.join(FOLDER_PATH, 'yelp_val.csv')
    OUT_PATH = 'out_2_2.csv'

    BUS_PATH = os.path.join(FOLDER_PATH, 'business.json')
    USER_PATH = os.path.join(FOLDER_PATH, 'review_train.json')
    TIP_PATH = os.path.join(FOLDER_PATH, 'tip.json')
    PHOTOS_PATH = os.path.join(FOLDER_PATH, 'photo.json')

    X_train, y_train = read_csv(IN_PATH, train = True)
    X_val, y_val = read_csv(VAL_PATH, train = True)

    param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 15, 20],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'min_child_weight': [5, 10, 15],
    'gamma': [0, 1, 2],
    'reg_lambda': [1, 2, 3],
    'alpha': [0, 0.5, 1],
    'colsample_bylevel': [0.5, 0.6, 0.7],
    'colsample_bynode': [0.5, 0.6, 0.7],
    'max_delta_step': [0, 1, 2],
    'random_state': [2000],
    'n_jobs': [-1]
}
    
    model = xgb.XGBRegressor(
    learning_rate=0.05,
    n_estimators=300,
    max_depth=15,
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=5,
    gamma=1,
    reg_lambda=2,
    alpha=0.5,
    random_state=2000,
    n_jobs=-1
)

    user_dict, bus_dict = get_label_encode_dict([X_train])
    X_train_enc, user_dict, bus_dict = encode_label(X_train, user_dict, bus_dict, train = True)
    X_val_enc, user_dict, bus_dict = encode_label(X_val, user_dict, bus_dict, train = False)
    
    # rdd reading and extracting features
    bus_rdd = rdd_read_json(BUS_PATH)
    user_rdd = rdd_read_json(USER_PATH)
    tip_rdd = rdd_read_json(TIP_PATH)
    photo_rdd = rdd_read_json(PHOTOS_PATH)

    # get features from business
    bus_feat_dict = dict(bus_rdd.map(lambda x: (x['business_id'], (x['stars'], x['review_count'], x['is_open']))).collect())
    user_feat_dict = dict(user_rdd.map(lambda x: (x['user_id'], x['stars'])).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1]))).collect())
    tip_feat_dict = dict(tip_rdd.map(lambda x: (x['business_id'], ['text'])).groupByKey().mapValues(len).collect())
    photo_feat_dict = dict(photo_rdd.map(lambda x: (x['business_id'], x['photo_id'])).groupByKey().mapValues(len).collect())
    tags_feat_dict = dict(user_rdd.map(lambda x: (x['business_id'], (x['useful'], x['funny'], x['cool']))).groupByKey().mapValues(lambda values: tuple(sum(x) / len(x) for x in zip(*values)
                                                                                                                                  )).collect())
    user_tags_feat_dict = dict(user_rdd.map(lambda x: (x['user_id'], (x['useful'], x['funny'], x['cool']))).groupByKey().mapValues(lambda values: tuple(sum(x) / len(x) for x in zip(*values)
                                                                                                                                  )).collect())
    user_num_reviews_dict = dict(user_rdd.map(lambda x: (x['user_id'], x['review_id'])).groupByKey().mapValues(len).collect())

    bus_feat_arr = [bus_feat_dict.get(bus, (0, 0, 0)) for bus in X_train[:, 1]]
    user_feat_arr = [user_feat_dict.get(user, 0) for user in X_train[:, 0]]
    tip_feat_arr = [tip_feat_dict.get(bus, 0) for bus in X_train[:, 1]]
    photo_feat_arr = [photo_feat_dict.get(bus, 0) for bus in X_train[:, 1]]
    tags_feat_arr = [tags_feat_dict.get(bus, (0,0,0)) for bus in X_train[:,1]]
    user_tags_arr = [user_tags_feat_dict.get(user, (0,0,0)) for user in X_train[:,0]]
    user_rev_arr = [user_num_reviews_dict.get(user, 0) for user in X_train[:,0]]
    
    #final_X_train = np.array([(i[0], i[1], j[0], j[1], j[2], k, l, m, n[0], n[1], n[2]) for i, j, k, l, m, n in zip(X_train_enc, bus_feat_arr, user_feat_arr, tip_feat_arr, photo_feat_arr, tags_feat_dict)])
    final_X_train = np.array([(j[0], j[1], k, l, m, n[0], n[1], n[2], o[0], o[1], o[2], p) for j, k, l, m, n, o, p in zip(bus_feat_arr, user_feat_arr, tip_feat_arr, photo_feat_arr, tags_feat_arr, user_tags_arr, user_rev_arr)])
    
    bus_feat_arr = [bus_feat_dict.get(bus, (0, 0, 0)) for bus in X_val[:, 1]]
    user_feat_arr = [user_feat_dict.get(user, 0) for user in X_val[:, 0]]
    tip_feat_arr = [tip_feat_dict.get(bus, 0) for bus in X_val[:, 1]]
    photo_feat_arr = [photo_feat_dict.get(bus, 0) for bus in X_val[:, 1]]
    tags_feat_arr = [tags_feat_dict.get(bus, (0,0,0)) for bus in X_val[:,1]]
    user_tags_arr = [user_tags_feat_dict.get(user, (0,0,0)) for user in X_val[:,0]]
    user_rev_arr = [user_num_reviews_dict.get(user, 0) for user in X_val[:,0]]
    
    #final_X_val = np.array([(i[0], i[1], j[0], j[1], j[2], k, l, m, n[0], n[1], n[2]) for i, j, k, l, m, n in zip(X_val_enc, bus_feat_arr, user_feat_arr, tip_feat_arr, photo_feat_arr, tags_feat_dict)])
    final_X_val = np.array([(j[0], j[1], k, l, m, n[0], n[1], n[2], o[0], o[1], o[2], p) for j, k, l, m, n, o, p in zip(bus_feat_arr, user_feat_arr, tip_feat_arr, photo_feat_arr, tags_feat_arr, user_tags_arr, user_rev_arr)])

    model.fit(final_X_train, y_train)

    #pk_save_model(model, 'save7.pk')
    #model = pk_load_model('save1.pk')

    pred = np.array(model.predict(final_X_val))
    train_pred = np.array(model.predict(final_X_train))

    train_rmse = RMSE(train_pred, y_train)
    rmse = RMSE(pred, y_val)
    
    with open(OUT_PATH, 'w') as f_out:
        f_out.write('user_id,business_id,prediction\n')
        
        for key_tuple, pred in zip(X_val, pred):
            out_str = f'{key_tuple[0]},{key_tuple[1]},{pred}\n'
            f_out.write(out_str)

    print(f'Train: {train_rmse}')
    print(f'RMSE: {rmse}')
    end = time.time()
    print(f'Duration: {end - start}') # 60 seconds

    sc.stop()
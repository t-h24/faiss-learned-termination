import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from util import mmap_fvecs

TEST_DIR = 'training_data/'
DB_DIR = '/mnt/hdd/conglonl/'
MODEL_DIR = 'training_model/'
EXPORT_DIR = 'training_results/'

def test_model(test_data, query_vecs, feature_idx, threshold, full_feature):
    dimensions = 960
    model_file_name = '{}GIST1M_HNSW16_model_thresh{}_Log_{}.txt'.format(
        MODEL_DIR, threshold, 'Full' if full_feature else 'Query')

    test_target = test_data[0].values.astype('float32')
    test_query = query_vecs[test_data[1].values]

    if full_feature:
        keep_idx = [2] + list(range(feature_idx * 4 + 3, feature_idx * 4 + 7))
        drop_idx = list(set(list(range(len(test_data.columns)))) - set(keep_idx))
        test_other = test_data.drop(drop_idx, axis=1).values
        test_feature = np.concatenate((test_query,test_other), axis=1)
    else:
        test_feature = test_query

    feature_name = ['F0_query_dim' + str(i) for i in range(dimensions)]
    if full_feature:
        feature_name += ['F1_d_start', 'F2_d_1st', 'F3_d_10th', 'F4_1st_to_start', 'F5_10th_to_start']

    # Load model
    gbm = lgb.Booster(model_file=model_file_name)
    start_time = time.time()
    y_pred = gbm.predict(test_feature, raw_score=True)
    print("Time taken: " + str(time.time() - start_time))

    # Take power of 2
    y_pred = np.maximum(0, y_pred)
    y_pred = np.power(2, y_pred)

    # Export test_target and y_pred
    export = '{}GIST1M_HNSW16_test_thresh{}_Log_{}.txt'.format(
        EXPORT_DIR, threshold, 'Full' if full_feature else 'Query')
    np.savetxt(export, np.column_stack((test_target, y_pred)), fmt=['%i', '%s'])

def main():
    # Load test data and query vectors
    test_data = pd.read_csv('{}GIST1M_HNSW16_test.tsv'.format(TEST_DIR), sep='\t', header=None)
    query_vecs = mmap_fvecs('{}gist_query.fvecs'.format(DB_DIR))
    thresholds = [381, 554, 801, 1260, 2441]

    for feature_idx, threshold in enumerate(thresholds):
        print("Testing model for threshold " + str(threshold))
        test_model(test_data, query_vecs, feature_idx, threshold, True)

    print("Testing model for query only")
    test_model(test_data, query_vecs, -1, 0, False)

if __name__ == '__main__':
    main()

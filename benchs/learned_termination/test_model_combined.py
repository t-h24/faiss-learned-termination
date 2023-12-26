import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from util import mmap_fvecs

TEST_DIR = 'training_data/'
DB_DIR = '/mnt/hdd/conglonl/'
MODEL_DIR = 'training_model/'
EXPORT_DIR = 'training_results/'

def test_model_combined(test_data, query_vecs, multiplier):
    thresholds = [381, 554, 801, 1260, 2441]
    dimensions = 960
    models = []
    for feature_idx, threshold in enumerate(thresholds):
        model_file_name = '{}GIST1M_HNSW16_model_thresh{}_Log_Full.txt'.format(
            MODEL_DIR, threshold)

        test_target = test_data[0].values.astype('float32')
        test_query = query_vecs[test_data[1].values]

        keep_idx = [2] + list(range(feature_idx * 4 + 3, feature_idx * 4 + 7))
        drop_idx = list(set(list(range(len(test_data.columns)))) - set(keep_idx))
        test_other = test_data.drop(drop_idx, axis=1).values
        test_feature = np.concatenate((test_query,test_other), axis=1)

        feature_name = ['F0_query_dim' + str(i) for i in range(dimensions)]
        feature_name += ['F1_d_start', 'F2_d_1st', 'F3_d_10th', 'F4_1st_to_start', 'F5_10th_to_start']

        # Load model
        gbm = lgb.Booster(model_file=model_file_name)
        models.append(gbm)

    start_time = time.time()
    y_pred = []
    for i in range(len(test_target)):
        model_idx = 0
        # Predict using the first model, if new threshold used use next model, until last model
        while model_idx < len(models):
            pred = models[model_idx].predict(test_feature[i].reshape(1, -1), raw_score=True)

            # Take power of 2
            pred = np.maximum(0, pred)
            pred = np.power(2, pred)
            pred = pred * multiplier
            if pred[0] >= thresholds[model_idx]:
                model_idx += 1
            else:
                break
        y_pred.append(pred[0])
    print("Time taken: " + str(time.time() - start_time))

    # Export test_target and y_pred
    export = '{}GIST1M_HNSW16_test_Combined_Log_Full_mult{}.txt'.format(EXPORT_DIR, multiplier)
    np.savetxt(export, np.column_stack((test_target, y_pred)), fmt=['%i', '%s'])

def main():
    # Load test data and query vectors
    test_data = pd.read_csv('{}GIST1M_HNSW16_test.tsv'.format(TEST_DIR), sep='\t', header=None)
    query_vecs = mmap_fvecs('{}gist_query.fvecs'.format(DB_DIR))

    # Note: Multiplier is needed for accurate results
    multiplier = 137.06

    print("Testing model (combined)")
    test_model_combined(test_data, query_vecs, multiplier)

if __name__ == '__main__':
    main()

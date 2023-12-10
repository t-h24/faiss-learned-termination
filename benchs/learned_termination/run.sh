#!/bin/sh

run="python -u bench_learned_termination.py"
train="python -u train_gbdt.py"

RESULT_DIR="results"
mkdir -p $RESULT_DIR

####### For each ANN index and each dataset, there are a few experiments that
####### need to be performed in order. Please uncomment the experiment that
####### you want to run.

# ### IVF index without quantization
# ### DEEP 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,700 -db DEEP10M -idx IVF4000,Flat -param search_mode=0 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 4000 -thread 10 -bsearch 1,1,700 -db DEEP10M -idx IVF4000,Flat -param search_mode=0 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 4000 -db DEEP10M -idx IVF4000,Flat -param search_mode=0,nprobe={1,2,3,6,11,19,23,28,38,58,606} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_naive_b1
# $run -mode 0 -batch 100 -cluster 4000 -db DEEP10M -idx IVF4000,Flat -param search_mode=0,nprobe={1,2,3,6,11,19,23,28,38,58,606} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_naive_b100
# $run -mode 0 -batch 10000 -cluster 4000 -db DEEP10M -idx IVF4000,Flat -param search_mode=0,nprobe={1,2,3,6,11,19,23,28,38,58,606} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_naive_b10000
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 4000 -thread 10 -thresh 1,2,3,6,11,19 -db DEEP10M -idx IVF4000,Flat -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_test
# $run -mode -2 -batch 1000000 -train 1 -cluster 4000 -thread 10 -thresh 1,2,3,6,11,19 -db DEEP10M -idx IVF4000,Flat -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 1,2,3,6,11,19 -db DEEP10M -idx IVF4000,Flat
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 4000 -thread 10 -thresh 6 -bsearch 1,1,5000 -db DEEP10M -idx IVF4000,Flat -param search_mode=2,pred_max=2074 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_tree6_b1_find
# $run -mode 1 -batch 1 -cluster 4000 -thresh 6 -db DEEP10M -idx IVF4000,Flat -param search_mode=2,pred_max=2074,nprobe={129,245,282,328,413,592,4725} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_tree6_b1
# $run -mode 1 -batch 100 -cluster 4000 -thresh 6 -db DEEP10M -idx IVF4000,Flat -param search_mode=2,pred_max=2074,nprobe={129,245,282,328,413,592,4725} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_tree6_b100
# $run -mode 1 -batch 10000 -cluster 4000 -thresh 6 -db DEEP10M -idx IVF4000,Flat -param search_mode=2,pred_max=2074,nprobe={129,245,282,328,413,592,4725} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_tree6_b10000
# # 7) Simple heuristic-based approach
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,500 -db DEEP10M -idx IVF4000,Flat -param search_mode=3 > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_heur_b1_find
# $run -mode 0 -cluster 4000 -db DEEP10M -idx IVF4000,Flat -param search_mode=3,nprobe={103,108,114,121,132,141,144,147,152,160,205} > $RESULT_DIR/result_DEEP10M_IVF4000_Flat_heur_b1

# ### IVF index without quantization
# ### SIFT 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,400 -db SIFT10M -idx IVF4000,Flat -param search_mode=0 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 4000 -thread 10 -bsearch 1,1,400 -db SIFT10M -idx IVF4000,Flat -param search_mode=0 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 4000 -db SIFT10M -idx IVF4000,Flat -param search_mode=0,nprobe={1,2,3,4,7,14,25,29,37,47,65,367} > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 4000 -thread 10 -thresh 2,3,4,7,14,25 -db SIFT10M -idx IVF4000,Flat -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_test
# $run -mode -2 -batch 1000000 -train 1 -cluster 4000 -thread 10 -thresh 2,3,4,7,14,25 -db SIFT10M -idx IVF4000,Flat -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 2,3,4,7,14,25 -db SIFT10M -idx IVF4000,Flat
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 4000 -thread 10 -thresh 7 -bsearch 1,1,5000 -db SIFT10M -idx IVF4000,Flat -param search_mode=2,pred_max=935 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_tree7_b1_find
# $run -mode 1 -batch 1 -cluster 4000 -thresh 7 -db SIFT10M -idx IVF4000,Flat -param search_mode=2,pred_max=935,nprobe={137,260,303,368,451,632,2867} > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_tree7_b1
# # 7) Simple heuristic-based approach
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,500 -db SIFT10M -idx IVF4000,Flat -param search_mode=3 > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_heur_b1_find
# $run -mode 0 -cluster 4000 -db SIFT10M -idx IVF4000,Flat -param search_mode=3,nprobe={106,112,119,126,138,148,150,154,161,170,246} > $RESULT_DIR/result_SIFT10M_IVF4000_Flat_heur_b1

# ### IVF index without quantization
# ### GIST 1M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 1000 -cluster 1000 -thread 10 -bsearch 1,1,200 -db GIST1M -idx IVF1000,Flat -param search_mode=0 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 1000 -thread 10 -bsearch 1,1,200 -db GIST1M -idx IVF1000,Flat -param search_mode=0 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 1000 -db GIST1M -idx IVF1000,Flat -param search_mode=0,nprobe={1,2,3,5,8,12,25,36,40,45,52,88,169} > $RESULT_DIR/result_GIST1M_IVF1000_Flat_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 1000 -cluster 1000 -thread 10 -thresh 3,5,8,12,25,36 -db GIST1M -idx IVF1000,Flat -param search_mode=1,pred_max=1000 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_test
# $run -mode -2 -batch 500000 -train 1 -cluster 1000 -thread 10 -thresh 3,5,8,12,25,36 -db GIST1M -idx IVF1000,Flat -param search_mode=1,pred_max=1000 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 3,5,8,12,25,36 -db GIST1M -idx IVF1000,Flat
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 1000 -cluster 1000 -thread 10 -thresh 12 -bsearch 1,1,5000 -db GIST1M -idx IVF1000,Flat -param search_mode=2,pred_max=518 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_tree12_b1_find
# $run -mode 1 -batch 1 -cluster 1000 -thresh 12 -db GIST1M -idx IVF1000,Flat -param search_mode=2,pred_max=518,nprobe={183,273,338,369,419,552,744} > $RESULT_DIR/result_GIST1M_IVF1000_Flat_tree12_b1
# # 7) Simple heuristic-based approach
# $run -mode 0 -batch 1000 -cluster 1000 -thread 10 -bsearch 1,1,500 -db GIST1M -idx IVF1000,Flat -param search_mode=3 > $RESULT_DIR/result_GIST1M_IVF1000_Flat_heur_b1_find
# $run -mode 0 -cluster 1000 -db GIST1M -idx IVF1000,Flat -param search_mode=3,nprobe={109,113,117,124,131,137,139,140,143,150,283} > $RESULT_DIR/result_GIST1M_IVF1000_Flat_heur_b1

#######################################################################################################

# ### HNSW index without quantization
# ### DEEP 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -thread 10 -bsearch 1,1,10000 -db DEEP10M -idx HNSW16 -param search_mode=0 > $RESULT_DIR/result_DEEP10M_HNSW16_naive_b1_find
# # 2) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -db DEEP10M -idx HNSW16 -param search_mode=0,efSearch={4,6,10,16,33,62,73,95,134,229,3850} > $RESULT_DIR/result_DEEP10M_HNSW16_naive_b1
# $run -mode 0 -batch 100 -db DEEP10M -idx HNSW16 -param search_mode=0,efSearch={4,6,10,16,33,62,73,95,134,229,3850} > $RESULT_DIR/result_DEEP10M_HNSW16_naive_b100
# $run -mode 0 -batch 10000 -db DEEP10M -idx HNSW16 -param search_mode=0,efSearch={4,6,10,16,33,62,73,95,134,229,3850} > $RESULT_DIR/result_DEEP10M_HNSW16_naive_b10000
# # 3) find the min. fixed number of distance evaluations (i.e., the termination condition we want to achieve) to reach a certain recall target for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -thread 10 -bsearch 1,1,10000 -db DEEP10M -idx HNSW16 -param search_mode=3 > $RESULT_DIR/result_DEEP10M_HNSW16_ndis_b1_find
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 3).
# $run -mode -1 -batch 10000 -thread 10 -thresh 191,265,368,547,1003 -db DEEP10M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_DEEP10M_HNSW16_test
# $run -mode -2 -batch 1000000 -train 1 -thread 10 -thresh 191,265,368,547,1003 -db DEEP10M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_DEEP10M_HNSW16_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 191,265,368,547,1003 -db DEEP10M -idx HNSW16
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -thread 10 -thresh 368 -bsearch 1,1,10000 -db DEEP10M -idx HNSW16 -param search_mode=2,pred_max=390437 > $RESULT_DIR/result_DEEP10M_HNSW16_tree368_b1_find
# $run -mode 1 -batch 1 -thresh 368 -db DEEP10M -idx HNSW16 -param search_mode=2,pred_max=390437,efSearch={135,257,402,471,576,736,1115,7259} > $RESULT_DIR/result_DEEP10M_HNSW16_tree368_b1
# $run -mode 1 -batch 100 -thresh 368 -db DEEP10M -idx HNSW16 -param search_mode=2,pred_max=390437,efSearch={135,257,402,471,576,736,1115,7259} > $RESULT_DIR/result_DEEP10M_HNSW16_tree368_b100
# $run -mode 1 -batch 10000 -thresh 368 -db DEEP10M -idx HNSW16 -param search_mode=2,pred_max=390437,efSearch={135,257,402,471,576,736,1115,7259} > $RESULT_DIR/result_DEEP10M_HNSW16_tree368_b10000

# ### HNSW index without quantization
# ### SIFT 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -thread 10 -bsearch 1,1,2000 -db SIFT10M -idx HNSW16 -param search_mode=0 > $RESULT_DIR/result_SIFT10M_HNSW16_naive_b1_find
# # 2) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -db SIFT10M -idx HNSW16 -param search_mode=0,efSearch={4,6,9,14,26,43,50,60,79,115,1111} > $RESULT_DIR/result_SIFT10M_HNSW16_naive_b1
# # 3) find the min. fixed number of distance evaluations (i.e., the termination condition we want to achieve) to reach a certain recall target for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -thread 10 -bsearch 1,1,10000 -db SIFT10M -idx HNSW16 -param search_mode=3 > $RESULT_DIR/result_SIFT10M_HNSW16_ndis_b1_find
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 3).
# $run -mode -1 -batch 10000 -thread 10 -thresh 179,241,335,481,817 -db SIFT10M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_SIFT10M_HNSW16_test
# $run -mode -2 -batch 1000000 -train 1 -thread 10 -thresh 179,241,335,481,817 -db SIFT10M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_SIFT10M_HNSW16_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 179,241,335,481,817 -db SIFT10M -idx HNSW16
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -thread 10 -thresh 241 -bsearch 1,1,4500 -db SIFT10M -idx HNSW16 -param search_mode=2,pred_max=62647 > $RESULT_DIR/result_SIFT10M_HNSW16_tree241_b1_find
# $run -mode 1 -batch 1 -thresh 241 -db SIFT10M -idx HNSW16 -param search_mode=2,pred_max=62647,efSearch={91,149,248,354,399,463,539,729,3816} > $RESULT_DIR/result_SIFT10M_HNSW16_tree241_b1

# ### HNSW index without quantization
# ### GIST 1M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 1000 -thread 10 -bsearch 1,1,20000 -db GIST1M -idx HNSW16 -param search_mode=0 > $RESULT_DIR/result_GIST1M_HNSW16_naive_b1_find
# # 2) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -db GIST1M -idx HNSW16 -param search_mode=0,efSearch={11,16,27,47,96,163,199,260,372,687,11853} > $RESULT_DIR/result_GIST1M_HNSW16_naive_b1
# # 3) find the min. fixed number of distance evaluations (i.e., the termination condition we want to achieve) to reach a certain recall target for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -thread 10 -bsearch 1,1,10000 -db GIST1M -idx HNSW16 -param search_mode=3 > $RESULT_DIR/result_GIST1M_HNSW16_ndis_b1_find
# 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 3).
# $run -mode -1 -batch 1000 -thread 10 -thresh 381,554,801,1260,2441 -db GIST1M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_GIST1M_HNSW16_test
# $run -mode -2 -batch 500000 -train 1 -thread 10 -thresh 381,554,801,1260,2441 -db GIST1M -idx HNSW16 -param search_mode=1 > $RESULT_DIR/result_GIST1M_HNSW16_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 381,554,801,1260,2441 -db GIST1M -idx HNSW16
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 1000 -thread 10 -thresh 1260 -bsearch 1,1,20000 -db GIST1M -idx HNSW16 -param search_mode=2,pred_max=696228 > $RESULT_DIR/result_GIST1M_HNSW16_tree1260_b1_find
# $run -mode 1 -batch 1 -thresh 1260 -db GIST1M -idx HNSW16 -param search_mode=2,pred_max=696228,efSearch={317,478,548,630,772,1282,14899} > $RESULT_DIR/result_GIST1M_HNSW16_tree1260_b1

# Print out average distance computations for a range of efSearches
# $run -mode -3 -batch 1 -db GIST1M -idx HNSW16 -param D_mode=1,search_mode=0,efRange={100,250,11853} > $RESULT_DIR/result_GIST1M_HNSW16_ef_dists
# Print out each query's distance computations for a efSearch value
# $run -mode -4 -batch 1 -db GIST1M -idx HNSW16 -param D_mode=1,search_mode=0,efSearch={11853} > $RESULT_DIR/result_GIST1M_HNSW16_ef_blind_11853

#######################################################################################################

# ### IVF index with quantization
# ### DEEP 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,1000 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=0 > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 4000 -thread 10 -bsearch 1,1,1000 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=0 > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 4000 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=0,nprobe={2,3,6,11,20,23,28,38,57,611} > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 4000 -thread 10 -thresh 2,3,6,11,20,23 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_test
# $run -mode -2 -batch 1000000 -train 1 -cluster 4000 -thread 10 -thresh 2,3,6,11,20,23 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 2,3,6,11,20,23 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 4000 -thread 10 -thresh 2 -bsearch 1,1,5500 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=2,pred_max=2031 > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_tree2_b1_find
# $run -mode 1 -batch 1 -cluster 4000 -thresh 2 -db DEEP10M -idx OPQ48_96,IVF4000,PQ48 -param search_mode=2,pred_max=2031,nprobe={43,90,177,284,319,371,464,646,5471} > $RESULT_DIR/result_DEEP10M_IVF4000_OPQ48_96_tree2_b1

# ### IVF index with quantization
# ### SIFT 10M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 4000 -thread 10 -bsearch 1,1,400 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=0 > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 4000 -thread 10 -bsearch 1,1,400 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=0 > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 4000 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=0,nprobe={2,3,4,7,14,25,28,36,47,63,398} > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 4000 -thread 10 -thresh 2,3,4,7,14,25 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_test
# $run -mode -2 -batch 1000000 -train 1 -cluster 4000 -thread 10 -thresh 2,3,4,7,14,25 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=1,pred_max=4000 > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 2,3,4,7,14,25 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 4000 -thread 10 -thresh 7 -bsearch 1,1,5000 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=2,pred_max=924 > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_tree7_b1_find
# $run -mode 1 -batch 1 -cluster 4000 -thresh 7 -db SIFT10M -idx OPQ64_128,IVF4000,PQ64 -param search_mode=2,pred_max=924,nprobe={162,325,376,456,573,788,3914} > $RESULT_DIR/result_SIFT10M_IVF4000_OPQ64_128_tree7_b1

# ### IVF index with quantization
# ### GIST 1M dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 1000 -cluster 1000 -thread 10 -bsearch 1,1,200 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=0 > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 1000 -thread 10 -bsearch 1,1,200 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=0 > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 1000 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=0,nprobe={3,5,7,12,23,35,38,43,53,86,165} > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 1000 -cluster 1000 -thread 10 -thresh 3,5,7,12,23,35 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=1,pred_max=1000 > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_test
# $run -mode -2 -batch 500000 -train 1 -cluster 1000 -thread 10 -thresh 3,5,7,12,23,35 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=1,pred_max=1000 > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 3,5,7,12,23,35 -db GIST1M -idx OPQ480_960,IVF1000,PQ480
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 1000 -cluster 1000 -thread 10 -thresh 5 -bsearch 1,1,5000 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=2,pred_max=500 > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_tree5_b1_find
# $run -mode 1 -batch 1 -cluster 1000 -thresh 5 -db GIST1M -idx OPQ480_960,IVF1000,PQ480 -param search_mode=2,pred_max=500,nprobe={33,70,148,217,243,266,339,447,779} > $RESULT_DIR/result_GIST1M_IVF1000_OPQ480_960_tree5_b1

#######################################################################################################

# ### IMI index with quantization
# ### DEEP 1B dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 16384 -thread 10 -bsearch 1,1,45000 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=0 > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 16384 -thread 10 -bsearch 1,1,45000 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=0 > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 16384 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=0,nprobe={1000,2464,3757,5792,9588,20728,44891} > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 16384 -thread 10 -thresh 20,51,145 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=1,pred_max=100000 > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_test
# $run -mode -2 -batch 10000 -train 1 -cluster 16384 -thread 10 -thresh 20,51,145 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=1,pred_max=100000 > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 20,51,145 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 16384 -thread 10 -thresh 20 -bsearch 1,1,30000 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=2,pred_max=100000 > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_tree20_b1_find
# $run -mode 1 -batch 1 -cluster 16384 -thresh 20 -db DEEP1000M -idx OPQ48_96,IMI2x14,PQ48 -param search_mode=2,pred_max=100000,nprobe={86,286,928,2280,3010,4207,6273,12923,27213} > $RESULT_DIR/result_DEEP1000M_IMI2x14_OPQ48_96_tree20_b1

# ### IMI index with quantization
# ### SIFT 1B dataset
# # 1) perform binary search to find the min. fixed configurations to reach different accuracy targets for testing queries.
# $run -mode 0 -batch 10000 -cluster 16384 -thread 10 -bsearch 1,1,35000 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=0 > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_naive_find
# # 2) perform binary search to find the min. fixed configurations to reach different accuracy targets for a sample of training vectors.
# $run -mode 0 -batch 10000 -train 1 -cluster 16384 -thread 10 -bsearch 1,1,35000 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=0 > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_train_find
# # 3) based on the min. config in the result file of 1), evaluate the performance of baseline.
# $run -mode 0 -batch 1 -cluster 16384 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=0,nprobe={1000,1915,2555,3638,5934,11622,21647} > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_naive_b1
# # 4) genreate training and testing data for the early terminaiton approach. The -thresh is chosen based on the min. fixed config in 2).
# $run -mode -1 -batch 10000 -cluster 16384 -thread 10 -thresh 27,65,178 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=1,pred_max=50000 > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_test
# $run -mode -2 -batch 10000 -train 1 -cluster 16384 -thread 10 -thresh 27,65,178 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=1,pred_max=50000 > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_train
# # 5) Using the training and testing data from 4), train the LightGBM decision tree models.
# $train -train 1 -thresh 27,65,178 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64
# # 6) Based on the performance estimation in the training log in 5), choose the -thresh and prediction model, and evaluate the performance. The pred_max is the Train ground truth max from the training log in 5).
# $run -mode 1 -batch 10000 -cluster 16384 -thread 10 -thresh 27 -bsearch 1,1,40000 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=2,pred_max=50000 > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_tree27_b1_find
# $run -mode 1 -batch 1 -cluster 16384 -thresh 27 -db SIFT1000M -idx OPQ64_128,IMI2x14,PQ64 -param search_mode=2,pred_max=50000,nprobe={133,420,1318,3094,3924,5140,7250,14272,30422} > $RESULT_DIR/result_SIFT1000M_IMI2x14_OPQ64_128_tree27_b1

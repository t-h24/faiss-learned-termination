/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include "HNSW.h"
#include "IndexFlat.h"
#include "IndexPQ.h"
#include "IndexScalarQuantizer.h"
#include "utils.h"


namespace faiss {

struct IndexHNSW;

struct ReconstructFromNeighbors {
    typedef Index::idx_t idx_t;
    typedef HNSW::storage_idx_t storage_idx_t;

    const IndexHNSW & index;
    size_t M; // number of neighbors
    size_t k; // number of codebook entries
    size_t nsq; // number of subvectors
    size_t code_size;
    int k_reorder; // nb to reorder. -1 = all

    std::vector<float> codebook; // size nsq * k * (M + 1)

    std::vector<uint8_t> codes; // size ntotal * code_size
    size_t ntotal;
    size_t d, dsub; // derived values

    explicit ReconstructFromNeighbors(const IndexHNSW& index,
                                      size_t k=256, size_t nsq=1);

    /// codes must be added in the correct order and the IndexHNSW
    /// must be populated and sorted
    void add_codes(size_t n, const float *x);

    size_t compute_distances(size_t n, const idx_t *shortlist,
                             const float *query, float *distances) const;

    /// called by add_codes
    void estimate_code(const float *x, storage_idx_t i, uint8_t *code) const;

    /// called by compute_distances
    void reconstruct(storage_idx_t i, float *x, float *tmp) const;

    void reconstruct_n(storage_idx_t n0, storage_idx_t ni, float *x) const;

    /// get the M+1 -by-d table for neighbor coordinates for vector i
    void get_neighbor_table(storage_idx_t i, float *out) const;

};


/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSW : Index {

    typedef HNSW::storage_idx_t storage_idx_t;

    // 0 = Original fixed configuration baseline.
    // 1 = Generate training data for prediction-based approach.
    // 2 = Prediction-based adaptive learned early termination.
    // 3 = ndis-based fixed configuration to find the minimum number
    //     of distance evaluations to reach certain accuracy targets. This is
    //     needed for generating training data and grid search on different
    //     intermediate search result features.
    int search_mode;

    // Maximum termination condition found in the training data.
    // This is used as an upper bound of the termination condition at online.
    long pred_max;

    // 0 = Return the distances of the closest neighbors sorted.
    // 1 = Return the number of distance computations taken.
    int D_mode = 0;

    // the link strcuture
    HNSW hnsw;

    // the sequential storage
    bool own_fields;
    Index *storage;

    ReconstructFromNeighbors *reconstruct_from_neighbors;

    explicit IndexHNSW (int d = 0, int M = 32);
    explicit IndexHNSW (Index *storage, int M = 32);

    ~IndexHNSW() override;

    // get a DistanceComputer object for this kind of storage
    virtual DistanceComputer *get_distance_computer() const = 0;

    void add(idx_t n, const float *x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset () override;

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(idx_t n, const float *x, idx_t k,
                        const storage_idx_t *nearest, const float *nearest_d,
                        float *distances, idx_t *labels, int nprobe = 1,
                        int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(
                        int k, const float *D, const idx_t *I);

    /// alternative graph building
    void init_level_0_from_entry_points(
                        int npt, const storage_idx_t *points,
                        const storage_idx_t *nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();

    // Load ground truth nearest neighbor database index
    // for finding ground truth minimum termination condition.
    void load_gt(long label);

    // Load the thresholds about when to make predictions.
    // This is related to the choice of intermediate search result features.
    void load_thresh(long thresh);

    // Load the prediction model.
    void load_model(char *file);
};



/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlat : IndexHNSW {
    IndexHNSWFlat();
    IndexHNSWFlat(int d, int M);
    DistanceComputer *
      get_distance_computer() const override;
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWPQ : IndexHNSW {
    IndexHNSWPQ();
    IndexHNSWPQ(int d, int pq_m, int M);
    void train(idx_t n, const float* x) override;
    DistanceComputer *
      get_distance_computer() const override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHNSWSQ : IndexHNSW {
    IndexHNSWSQ();
    IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M);
    DistanceComputer *
      get_distance_computer() const override;
};

/** 2-level code structure with fast random access
 */
struct IndexHNSW2Level : IndexHNSW {
    IndexHNSW2Level();
    IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M);
    DistanceComputer *
      get_distance_computer() const override;
    void flip_to_ivf();

    /// entry point for search
    void search (idx_t n, const float *x, idx_t k,
                 float *distances, idx_t *labels) const override;

};


}  // namespace faiss

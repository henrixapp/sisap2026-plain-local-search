#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "math_utils.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/utils/distances.h> // for metric types

template <int K> class IndexBuilder {
public:
  IndexBuilder(size_t d_in, size_t nlist, size_t nprobe)
      : d_in_(d_in), d_out_(d_in), nlist_(nlist), nprobe_(nprobe) {}

  void train_pca(const std::vector<float> &sample_data, float target_variance) {
    size_t n = sample_data.size() / d_in_;
    faiss::PCAMatrix pca(d_in_, d_in_);
    pca.train(n, sample_data.data());

    std::vector<float> eigenvalues(pca.eigenvalues.data(),
                                   pca.eigenvalues.data() + d_in_);
    d_out_ = math_utils::calculate_dynamic_d_out(eigenvalues, target_variance);
    std::cout << "Dynamic PCA target dimensions (d_out): " << d_out_
              << " for variance " << target_variance << "\n";
  }

  void build_and_train_index(const std::vector<float> &sample_data) {
    size_t n = sample_data.size() / d_in_;

    // 1. The base index (IVF with Scalar Quantizer)
    // Metric is INNER_PRODUCT because vectors will be normalized
    quantizer_ = std::make_unique<faiss::IndexFlatIP>(d_out_);
    ivf_sq_ = std::make_unique<faiss::IndexIVFScalarQuantizer>(
        quantizer_.get(), d_out_, nlist_, faiss::ScalarQuantizer::QT_8bit,
        faiss::METRIC_INNER_PRODUCT);
    static_cast<faiss::IndexIVF *>(ivf_sq_.get())->nprobe = nprobe_;

    // 2. The pipeline wrapper
    final_index_ = std::make_unique<faiss::IndexPreTransform>(ivf_sq_.get());

    // 3. Chain the transformations
    final_index_->prepend_transform(new faiss::NormalizationTransform(d_out_, 2.0));
    faiss::PCAMatrix *final_pca = new faiss::PCAMatrix(d_in_, d_out_);
    final_index_->prepend_transform(final_pca);

    // Train the entire pipeline using the sample
    std::cout << "Training the final index pipeline...\n";
    final_index_->train(n, sample_data.data());
    std::cout << "Training complete.\n";
  }

  void add_batch(size_t num_vectors, const std::vector<float> &batch_data) {
    final_index_->add(num_vectors, batch_data.data());
  }

  void search_batch(size_t num_vectors, const std::vector<float> &batch_data,
                    std::vector<float> &distances,
                    std::vector<int64_t> &labels) {
    distances.resize(num_vectors * K);
    labels.resize(num_vectors * K);

    // faiss index.search natively expects faiss::idx_t which is int64_t
    final_index_->search(num_vectors, batch_data.data(), K, distances.data(),
                         labels.data());
  }

private:
  size_t d_in_;
  size_t d_out_;
  size_t nlist_;
  size_t nprobe_;

  std::unique_ptr<faiss::Index> quantizer_;
  std::unique_ptr<faiss::Index> ivf_sq_;
  std::unique_ptr<faiss::IndexPreTransform> final_index_;
};

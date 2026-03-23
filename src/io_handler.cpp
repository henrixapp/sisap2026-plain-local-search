#include "io_handler.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

IOHandler::IOHandler(const std::string &input_path,
                     const std::string &input_dataset,
                     const std::string &output_path,
                     const std::string &output_dataset_labels,
                     const std::string &output_dataset_distances,
                     size_t batch_size)
    : input_path_(input_path), input_dataset_(input_dataset),
      output_path_(output_path), output_dataset_labels_(output_dataset_labels),
      output_dataset_distances_(output_dataset_distances),
      batch_size_(batch_size) {

  // Open input file
  input_file_ =
      std::make_unique<HighFive::File>(input_path_, HighFive::File::ReadOnly);
  input_ds_ = std::make_unique<HighFive::DataSet>(
      input_file_->getDataSet(input_dataset_));

  auto dimensions = input_ds_->getDimensions();
  if (dimensions.size() != 2) {
    throw std::runtime_error("Input dataset must be 2D.");
  }

  num_vectors_ = dimensions[0];
  dimension_ = dimensions[1];
}

size_t IOHandler::get_num_vectors() const { return num_vectors_; }
size_t IOHandler::get_dimension() const { return dimension_; }

void IOHandler::reset_read() { current_read_offset_ = 0; }

size_t IOHandler::read_batch(std::vector<half_float::half> &batch_data) {
  if (current_read_offset_ >= num_vectors_) {
    return 0;
  }

  size_t vectors_to_read =
      std::min(batch_size_, num_vectors_ - current_read_offset_);
  batch_data.resize(vectors_to_read * dimension_);

  input_ds_->select({current_read_offset_, 0}, {vectors_to_read, dimension_})
      .read_raw(batch_data.data());

  current_read_offset_ += vectors_to_read;
  return vectors_to_read;
}

std::vector<half_float::half> IOHandler::read_sample(size_t sample_size) {
  size_t actual_sample_size = std::min(sample_size, num_vectors_);
  std::vector<half_float::half> result(actual_sample_size * dimension_);

  // Generate random indices
  std::vector<size_t> indices(num_vectors_);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen(42); // fixed seed for reproducibility
  std::shuffle(indices.begin(), indices.end(), gen);

  // Read vector by vector into result
  // To speed up we could sort the sample indices and read them sequentially,
  // but random small reads might be slow. Sorting indices improves disk
  // caching.
  std::vector<size_t> selected_indices(indices.begin(),
                                       indices.begin() + actual_sample_size);
  std::sort(selected_indices.begin(), selected_indices.end());

  std::vector<half_float::half> single_vec(dimension_);
  for (size_t i = 0; i < actual_sample_size; ++i) {
    input_ds_->select({selected_indices[i], 0}, {1, dimension_})
        .read_raw(single_vec.data());
    std::copy(single_vec.begin(), single_vec.end(),
              result.begin() + i * dimension_);
  }

  return result;
}

void IOHandler::init_output(size_t num_vectors, size_t k) {
  output_file_ =
      std::make_unique<HighFive::File>(output_path_, HighFive::File::Truncate);

  HighFive::DataSpace space({num_vectors, k});
  HighFive::DataSetCreateProps props;
  props.add(HighFive::Chunking(
      std::vector<hsize_t>{std::min(batch_size_, num_vectors), k}));

  output_ds_labels_ =
      std::make_unique<HighFive::DataSet>(output_file_->createDataSet<int64_t>(
          output_dataset_labels_, space, props));

  output_ds_distances_ =
      std::make_unique<HighFive::DataSet>(output_file_->createDataSet<float>(
          output_dataset_distances_, space, props));

  current_write_offset_ = 0;
}

void IOHandler::write_batch(size_t num_vectors_in_batch,
                            const std::vector<int64_t> &labels,
                            const std::vector<float> &distances) {
  if (num_vectors_in_batch == 0)
    return;

  size_t k = labels.size() / num_vectors_in_batch;

  output_ds_labels_
      ->select({current_write_offset_, 0}, {num_vectors_in_batch, k})
      .write_raw(labels.data());

  output_ds_distances_
      ->select({current_write_offset_, 0}, {num_vectors_in_batch, k})
      .write_raw(distances.data());

  current_write_offset_ += num_vectors_in_batch;
}

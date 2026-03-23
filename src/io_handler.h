#pragma once

#include <highfive/H5File.hpp>
#include <highfive/half_float.hpp>
#include <memory>
#include <string>
#include <vector>

class IOHandler {
public:
  IOHandler(const std::string &input_path, const std::string &input_dataset,
            const std::string &output_path,
            const std::string &output_dataset_labels,
            const std::string &output_dataset_distances, size_t batch_size);

  // Get input dimensions
  size_t get_num_vectors() const;
  size_t get_dimension() const;

  // Reset read offset to start from the beginning
  void reset_read();

  // Read a batch of f16 vectors
  // Returns the number of vectors successfully read (0 if EOF)
  size_t read_batch(std::vector<half_float::half> &batch_data);

  // Read a uniform random sample of vectors
  std::vector<half_float::half> read_sample(size_t sample_size);

  // Initialize output HDF5 file and datasets
  void init_output(size_t num_vectors, size_t k);

  // Write a batch of results
  void write_batch(size_t num_vectors_in_batch,
                   const std::vector<int64_t> &labels,
                   const std::vector<float> &distances);

private:
  std::string input_path_;
  std::string input_dataset_;
  std::string output_path_;
  std::string output_dataset_labels_;
  std::string output_dataset_distances_;
  size_t batch_size_;

  std::unique_ptr<HighFive::File> input_file_;
  std::unique_ptr<HighFive::DataSet> input_ds_;

  std::unique_ptr<HighFive::File> output_file_;
  std::unique_ptr<HighFive::DataSet> output_ds_labels_;
  std::unique_ptr<HighFive::DataSet> output_ds_distances_;

  size_t current_read_offset_ = 0;
  size_t current_write_offset_ = 0;
  size_t num_vectors_ = 0;
  size_t dimension_ = 0;
};

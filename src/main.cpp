#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "index_builder.h"
#include "io_handler.h"
#include "math_utils.h"

class Timer {
public:
  Timer() : start_time_(std::chrono::high_resolution_clock::now()), last_step_time_(start_time_) {}

  void step(const std::string& step_name) {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = now - last_step_time_;
    std::cout << "[Timer] " << step_name << " took " << diff.count() << " seconds.\n";
    last_step_time_ = now;
  }

  void total() {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = now - start_time_;
    std::cout << "[Timer] Pipeline completed in " << diff.count() << " seconds.\n";
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_step_time_;
};

// Basic CLI parser
std::string get_arg(int argc, char **argv, const std::string &arg,
                    const std::string &default_val = "") {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == arg) {
      return std::string(argv[i + 1]);
    }
  }
  return default_val;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " --input <in.h5> --output <out.h5> [options...]\n";
    return 1;
  }

  std::string input_path = get_arg(argc, argv, "--input");
  std::string output_path = get_arg(argc, argv, "--output");
  std::string in_ds = get_arg(argc, argv, "--in_ds", "train");
  std::string out_ds_labels = get_arg(argc, argv, "--out_ds_labels", "labels");
  std::string out_ds_distances =
      get_arg(argc, argv, "--out_ds_distances", "distances");

  size_t batch_size = std::stoull(get_arg(argc, argv, "--batch_size", "50000"));
  float pca_variance = std::stof(get_arg(argc, argv, "--pca_var", "0.95"));

  // Template arg K is configured to 15 at compile time in index_builder.
  constexpr int K = 15;

  std::cout << "Starting FAISS pipeline...\n"
            << "Input path: " << input_path << "\nOutput path: " << output_path
            << "\n";
  Timer timer;

  IOHandler io(input_path, in_ds, output_path, out_ds_labels, out_ds_distances,
               batch_size);

  size_t num_vectors = io.get_num_vectors();
  size_t d_in = io.get_dimension();

  // Heuristics based on input size
  size_t nlist = static_cast<size_t>(4.0 * std::sqrt(num_vectors));
  size_t nprobe = std::max(32UL, nlist / 64);
  size_t sample_size = std::min(nlist * 39, num_vectors / 2);

  std::cout << "Data: " << num_vectors << " vectors of " << d_in
            << " dimensions.\n";
  std::cout << "Heuristics: nlist=" << nlist << ", nprobe=" << nprobe << "\n";
  timer.step("Initialization");

  IndexBuilder<K> builder(d_in, nlist, nprobe);

  std::cout << "Reading sample subset of size " << sample_size << "...\n";
  auto sample_f16 = io.read_sample(sample_size);
  auto sample_f32 = math_utils::convert_f16_to_f32(sample_f16);
  timer.step("Reading sample subset");

  std::cout << "Training PCA on sample...\n";
  builder.train_pca(sample_f32, pca_variance);
  timer.step("Training PCA");

  std::cout << "Building and training index...\n";
  builder.build_and_train_index(sample_f32);
  timer.step("Building and training index");

  std::cout << "Populating index...\n";
  io.reset_read();
  std::vector<half_float::half> batch_f16;
  while (size_t num_read = io.read_batch(batch_f16)) {
    auto batch_f32 = math_utils::convert_f16_to_f32(batch_f16);
    builder.add_batch(num_read, batch_f32);
  }
  timer.step("Populating index");

  std::cout << "Index populated. Creating output datasets...\n";
  io.init_output(num_vectors, K);
  timer.step("Creating output datasets");

  std::cout << "Searching index and writing results...\n";
  io.reset_read();
  std::vector<int64_t> labels;
  std::vector<float> distances;
  while (size_t num_read = io.read_batch(batch_f16)) {
    auto batch_f32 = math_utils::convert_f16_to_f32(batch_f16);
    builder.search_batch(num_read, batch_f32, distances, labels);
    for (auto &label : labels) {
      label += 1;
    }
    io.write_batch(num_read, labels, distances);
  }
  timer.step("Searching index and writing results");

  timer.total();

  return 0;
}

#include <highfive/H5File.hpp>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

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
    std::cerr
        << "Usage: " << argv[0]
        << " --true <ground_truth.h5> --approx <approximate.h5> [options...]\n";
    std::cerr
        << "Options:\n"
        << "  --true_ds      Dataset name in true file (default: labels)\n"
        << "  --approx_ds    Dataset name in approx file (default: labels)\n";
    return 1;
  }

  std::string gt_path = get_arg(argc, argv, "--true");
  std::string approx_path = get_arg(argc, argv, "--approx");
  std::string gt_ds_name = get_arg(argc, argv, "--true_ds", "labels");
  std::string approx_ds_name = get_arg(argc, argv, "--approx_ds", "labels");

  if (gt_path.empty() || approx_path.empty()) {
    std::cerr << "Error: You must provide both --true and --approx paths.\n";
    return 1;
  }

  try {
    HighFive::File gt_file(gt_path, HighFive::File::ReadOnly);
    HighFive::DataSet gt_ds = gt_file.getDataSet(gt_ds_name);
    auto gt_dims = gt_ds.getDimensions();

    HighFive::File approx_file(approx_path, HighFive::File::ReadOnly);
    HighFive::DataSet approx_ds = approx_file.getDataSet(approx_ds_name);
    auto approx_dims = approx_ds.getDimensions();

    if (gt_dims.size() != 2 || approx_dims.size() != 2) {
      std::cerr << "Error: Datasets must be 2-dimensional.\n";
      return 1;
    }

    if (gt_dims[0] != approx_dims[0] || gt_dims[1] < approx_dims[1]) {
      std::cerr << "Error: Dataset dimensions do not match.\n";
      std::cerr << "Ground truth: " << gt_dims[0] << "x" << gt_dims[1] << "\n";
      std::cerr << "Approximate: " << approx_dims[0] << "x" << approx_dims[1]
                << "\n";
      return 1;
    }

    size_t num_queries = gt_dims[0];
    size_t K = approx_dims[1];

    std::cout << "Computing recall for " << num_queries << " queries (K=" << K
              << ")...\n";

    // Reading dataset chunks could be faster, but let's just read everything if
    // it fits in memory. Assuming num_queries * K * 8 bytes is within RAM
    // limits (e.g. 1M * 100 * 8 = 800MB) For a more robust tool, we can process
    // row by row or chunk by chunk.
    std::vector<std::vector<int64_t>> gt_labels;
    gt_ds.read(gt_labels);

    std::vector<std::vector<int64_t>> approx_labels;
    approx_ds.read(approx_labels);

    double total_recall = 0.0;

    for (size_t i = 0; i < num_queries; ++i) {
      std::unordered_set<int64_t> gt_set(gt_labels[i].begin(),
                                         gt_labels[i].begin() + approx_dims[1]);
      int matches = 0;
      for (int64_t label : approx_labels[i]) {
        if (gt_set.count(label)) {
          matches++;
        }
      }
      total_recall += static_cast<double>(matches) / K;
    }

    double mean_recall = total_recall / num_queries;
    std::cout << "Mean Recall@" << K << ": " << mean_recall << std::endl;

  } catch (HighFive::Exception &err) {
    std::cerr << "HighFive error: " << err.what() << std::endl;
    return 1;
  } catch (std::exception &err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}

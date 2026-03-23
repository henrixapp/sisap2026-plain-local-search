load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _non_bcr_ext_impl(ctx):
    http_archive(
        name = "faiss",
        build_file = Label("//:faiss.BUILD"),
        sha256 = "0216d38d8c5c460433815b72d3cde6725eaa5d5770576277a2abc01ffc414d20",
        strip_prefix = "faiss-1.14.1",
        urls = ["https://github.com/facebookresearch/faiss/archive/refs/tags/v1.14.1.tar.gz"],
    )

    http_archive(
        name = "HighFive",
        build_file = Label("//:highfive.BUILD"),
        sha256 = "325cfbcf0c0296a6dd26f3b088801b7ebb8d6f109c0565c11d2d8c4af3253bff",
        strip_prefix = "highfive-3.3.0",
        urls = ["https://github.com/highfive-devs/HighFive/archive/refs/tags/v3.3.0.tar.gz"],
    )

    http_archive(
        name = "FP16",
        build_file = Label("//:fp16.BUILD"),
        sha256 = "65ace2f05fd9434b0acb7a7d3cc6cd96842ea6236b680594af932b359bedbfc1",
        strip_prefix = "FP16-3d2de1816307bac63c16a297e8c4dc501b4076df",
        urls = ["https://github.com/Maratyszcza/FP16/archive/3d2de1816307bac63c16a297e8c4dc501b4076df.tar.gz"],
    )

non_bcr_ext = module_extension(
    implementation = _non_bcr_ext_impl,
)

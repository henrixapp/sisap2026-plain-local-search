load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "faiss_cmake",
    cache_entries = {
        "FAISS_ENABLE_GPU": "OFF",
        "FAISS_ENABLE_PYTHON": "OFF",
        "BUILD_TESTING": "OFF",
    },
    lib_source = ":all_srcs",
    out_static_libs = ["libfaiss.a"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "fais_hdrs",
    hdrs = glob(["faiss/**/*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "faiss",
    visibility = ["//visibility:public"],
    deps = [
        ":fais_hdrs",
        ":faiss_cmake",
    ],
)

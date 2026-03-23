cc_binary(
    name = "low_dim_faiss",
    srcs = [
        "src/index_builder.cpp",
        "src/index_builder.h",
        "src/io_handler.cpp",
        "src/io_handler.h",
        "src/main.cpp",
        "src/math_utils.cpp",
        "src/math_utils.h",
    ],
    copts = [
        "-O3",
        "-std=c++17",
        "-I/usr/include/hdf5/serial",
    ],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu/hdf5/serial",
        "-lhdf5",
        "-lblas",  # FAISS depends on BLAS/LAPACK from system
        "-lgomp",
        "-llapack",
    ],
    deps = [
        ":half",
        "@FP16",
        "@HighFive",
        "@faiss",
    ],
)

cc_import(
    name = "half",
    hdrs = ["includes/half.hpp"],
    includes = ["includes"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "compute_recall",
    srcs = ["src/compute_recall.cpp"],
    copts = [
        "-O3",
        "-std=c++17",
        "-I/usr/include/hdf5/serial",
    ],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu/hdf5/serial",
        "-lhdf5",
    ],
    deps = [
        "@HighFive",
    ],
)

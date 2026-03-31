# Requirements

I used the h5 library for parsing the binary format. To simplify the linker stuff, I just used the gcc wrapper h5cc provided with the library.

The makefile should work directly if h5cc is available, otherwise you probably need to modify the makefile to compile.
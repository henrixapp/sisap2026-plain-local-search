#pragma once

#include <stdlib.h>

// aligned_alloc is only specified for sizes that are a multiple of the
// alignment, so round every request up rather than relying on the allocator
// being lenient about it.
static inline void *alloc64(size_t bytes)
{
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

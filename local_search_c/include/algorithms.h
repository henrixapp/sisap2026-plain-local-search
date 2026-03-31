#pragma once

// Everything here assumes sorted lists

// Computes the union of A and B and stores the result in C.
static inline int set_union(const int *A, int a, const int *B, int b, int *C)
{
    int i = 0, j = 0, k = 0;
    while (i < a && j < b)
    {
        if (A[i] < B[j])
        {
            C[k++] = A[i++];
        }
        else if (A[i] > B[j])
        {
            C[k++] = B[j++];
        }
        else
        {
            C[k++] = A[i++];
            j++;
        }
    }
    while (i < a)
    {
        C[k++] = A[i++];
    }
    while (j < b)
    {
        C[k++] = B[j++];
    }
    return k;
}

// Computes the difference A setminus B and stores the result in C.
static inline int set_difference(const int *A, int a, const int *B, int b, int *C)
{
    int i = 0, j = 0, k = 0;
    while (i < a && j < b)
    {
        if (A[i] < B[j])
        {
            C[k++] = A[i++];
        }
        else if (A[i] > B[j])
        {
            j++;
        }
        else
        {
            i++;
            j++;
        }
    }
    while (i < a)
    {
        C[k++] = A[i++];
    }
    return k;
}

// Remove any elements that are not in sorted order from A.
static inline int remove_duplicates(int *A, int a)
{
    int p = 1;
    for (int i = 1; i < a; i++)
    {
        if (A[i] > A[p - 1])
        {
            A[p++] = A[i];
        }
    }
    return p;
}

// Test if A is a sorted
static inline int set_is_sorted(const int *A, int a)
{
    for (int i = 0; i < a - 1; i++)
    {
        if (A[i] >= A[i + 1])
            return 0;
    }
    return 1;
}

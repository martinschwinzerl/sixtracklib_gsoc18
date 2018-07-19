#include <cstdio>
#include <cassert>

// #include "sixtracklib/sixtracklib.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

// extern void run(double **indata, double **outdata, int npart );

__global__ void test( double* x, int npart )
{
    if( npart > 0 )
    {
        printf( "numbers : %.8f\r\n", x[ 0 ] );
        printf( "numbers : %.8f\r\n", x[ 1 ] );
        printf( "numbers : %.8f\r\n", x[ 2 ] );
        printf( "numbers : %.8f\r\n", x[ 3 ] );
        printf( "numbers : %.8f\r\n", x[ 4 ] );
        printf( "numbers : %.8f\r\n", x[ 5 ] );
        printf( "numbers : %.8f\r\n", x[ 6 ] );
    }

    return;
}

int main()
{
    int npart = 10;

    double* host_particle_buffer = 0;
    double* dev_particle_buffer  = 0;

    cudaError_t err = cudaSuccess;

    unsigned int device_flags = 0u;
    cudaGetDeviceFlags( &device_flags );

    if( ( device_flags & cudaDeviceMapHost ) != cudaDeviceMapHost )
    {
        printf( "pinned memory not available with the "
                "cuda device -> aborting\r\n" );

        return 0;
    }

    err = cudaHostAlloc( ( void** )&host_particle_buffer, npart * 240u,
                         cudaHostAllocMapped );

    assert( err == cudaSuccess );
    assert( host_particle_buffer != 0 );

    err = cudaHostGetDevicePointer(
        ( void** )&dev_particle_buffer, host_particle_buffer, 0u );
    assert( err == cudaSuccess );

    if( npart > 0 )
    {
        host_particle_buffer[  0 ] = 1.2345;
        host_particle_buffer[  1 ] = 2.2345;
        host_particle_buffer[  2 ] = 3.2345;
        host_particle_buffer[  3 ] = 4.2345;
        host_particle_buffer[  4 ] = 5.2345;
        host_particle_buffer[  5 ] = 6.2345;
        host_particle_buffer[  6 ] = 7.2345;
    }

    test<<< 1, 1 >>>( dev_particle_buffer, npart );


    err = cudaFreeHost( host_particle_buffer );
    host_particle_buffer = 0;
    assert( err == cudaSuccess );

    return 0;
}

/* end: studies/study10/run_sample_fodo.c */

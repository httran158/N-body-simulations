#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

const int GSIZE = 50;
const int NUM_PARTICLES = GSIZE * GSIZE * GSIZE;  // The number of Particles
const double G = 1.0e-7;                          // G
const double M = 10.0;                            // m
const double dt = 1.0e-2;                         // time step
const int nSteps = 500;                           // The number of steps
const int exportInterval = 10;                    // Export Interval
const double EPSILON = 1.0e-3;                    // Small parameter

/* Number of threads per block */
const int THREADS_PER_BLOCK = 512;

// cuda kernel : compute particle force Fi = sum_j G * m_i m_j (X_j - X_i)/|(X_j - X_i)|^2/3
__global__ void computeParticleForce(int numOfParticles, double *x, double *y, double *z, double *focx, double *focy, double *focz) {
    int i = threadIdx.x + (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x; /* thread ID = particle index */
    // 'i' is the cuda thread id. We use 'i' as the index of particles. Only when i < numOfParticles, compute the force.
    // (x,y,z) is the coordinate. (focx,focy,foxz) is the force.

    if (i < numOfParticles){
            focx[i] = 0.0;
            focy[i] = 0.0;
            focz[i] = 0.0;
		// Loop over particles that exert force
            for (int j = 0; j < NUM_PARTICLES; j++)
            {
                // calculate the net force
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];
                double drSquared = dx * dx + dy * dy + dz * dz + EPSILON;
                double drPowerN32 = 1.0 / (drSquared * std::sqrt(drSquared));
                focx[i] += G * M * M * dx * drPowerN32;
                focy[i] += G * M * M * dy * drPowerN32;
                focz[i] += G * M * M * dz * drPowerN32;
            }
    }
}

// cuda kernel : compute particle speed, U = sqrt(u * u + v * v + w * w)
__global__ void computeSpeed(int numOfParticles, double *sp, double *u, double *v, double *w) {
    int i = threadIdx.x + (blockIdx.x + gridDim.x * blockIdx.y) * blockDim.x; /* thread ID = particle index */
    // 'i' is the cuda thread id. We use 'i' as the index of particles. Only when i < numOfParticles, compute the speed.
    // (u,v,w) is the velocity.
    
    if (i < numOfParticles){
            sp[i] = std::sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
    }
}

// timing method
double tsecond() {
    struct timeval tm;
    double t;
    static int base_sec = 0, base_usec = 0;

    gettimeofday(&tm, NULL);
    if (base_sec == 0 && base_usec == 0) {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    } else {
        t = (double)(tm.tv_sec - base_sec) +
            ((double)(tm.tv_usec - base_usec)) / 1.0e6;
    }
    return t;
}

// export data
void exportParticle(std::string fn, int size, double *px, double *py,
                    double *pz, double *pu, double *pv, double *pw) {
    std::ofstream ofile;
    ofile.open(fn.c_str());
    ofile << std::setprecision(16);
    ofile << std::scientific;
    for (int i = 0; i < size; i++) {
        ofile << px[i] << ' ' << py[i] << ' ' << pz[i] << ' ' << pu[i] << ' '
              << pv[i] << ' ' << pw[i] << std::endl;
    }
    ofile.close();
}

// main
int main(int argc, char **argv) {
    /// Initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate arrays of coordinates, x,y,z on Host
    double *x_cord = new double[NUM_PARTICLES];
    double *y_cord = new double[NUM_PARTICLES];
    double *z_cord = new double[NUM_PARTICLES];

    // Allocate arrays of velocities, u,v,w on Host
    double *u_vel = new double[NUM_PARTICLES];
    double *v_vel = new double[NUM_PARTICLES];
    double *w_vel = new double[NUM_PARTICLES];

    // set initial particle coordinate and velocity on Host
    for (int i = 0; i < NUM_PARTICLES; i++) {
        double x = static_cast<double>(i % GSIZE) / GSIZE;
        double y = static_cast<double>((i / GSIZE) % GSIZE) / GSIZE;
        double z = static_cast<double>(i / GSIZE / GSIZE) / GSIZE;
        x_cord[i] = x;
        y_cord[i] = y;
        z_cord[i] = z;
        u_vel[i] = y;
        v_vel[i] = -x;
        w_vel[i] = 0;
    }

    // initial export
    int count = 0;
    std::ostringstream fileName;
    fileName << "CUDA_" << std::setfill('0') << std::setw(3) << count << ".txt";
    exportParticle(fileName.str(), NUM_PARTICLES, x_cord, y_cord, z_cord, u_vel,v_vel, w_vel);
    count++;

    // Allocate arrays of force, f,g,h on Device
    double *x_force_d;
    double *y_force_d;
    double *z_force_d;
    cudaMalloc((void **)&x_force_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&y_force_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&z_force_d,NUM_PARTICLES * sizeof(double));

    // Allocate arrays of coordinates, x,y,z on Device and copy the coordinates from Host to Device
    double *x_cord_d;
    double *y_cord_d;
    double *z_cord_d;
    cudaMalloc((void **)&x_cord_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&y_cord_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&z_cord_d,NUM_PARTICLES * sizeof(double));

    // Allocate arrays of velocities, u,v,w on Device and copy the velocities from Host to Device
    double *u_vel_d;
    double *v_vel_d;
    double *w_vel_d;
    cudaMalloc((void **)&u_vel_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&v_vel_d,NUM_PARTICLES * sizeof(double));
    cudaMalloc((void **)&w_vel_d,NUM_PARTICLES * sizeof(double));

    // Allocate arrays of the speed on Device
    double *speed_d;
    cudaMalloc((void **)&speed_d, NUM_PARTICLES * sizeof(double));

    // Determine CUDA Thead Blocks
    int bl = NUM_PARTICLES / THREADS_PER_BLOCK + 1;  // number of blocks
    int bx = min(bl, 512);                           // number of block for x
    int by = bl / 512 + 1;                           // number of block for y
    dim3 dimblock(bx, by, 1);                        //  block grid

    // Dummy variable to compute sum with cublas 'cublasDdot'
    double one = 1.0;
    double *one_d;
    cudaMalloc((void **)&one_d, sizeof(double));
    cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice);
    
    cudaMemcpy(x_cord_d, x_cord, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_cord_d, y_cord, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(z_cord_d, z_cord, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_vel_d, u_vel, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_vel_d, v_vel, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(w_vel_d, w_vel, NUM_PARTICLES * sizeof(double), cudaMemcpyHostToDevice);
    // Propagate Particles
    for (int step = 1; step <= nSteps; step++) {
        const double tStart = tsecond();  // Start timing

        // Compute Force on Device
        computeParticleForce<<<dimblock, THREADS_PER_BLOCK>>>(NUM_PARTICLES, x_cord_d, y_cord_d, z_cord_d, x_force_d, y_force_d, z_force_d);

        // Update velocities on Device using cublas
        double alpha = dt / M;
        /////////////////////////////////////////
        cublasDaxpy(handle, NUM_PARTICLES, &alpha, x_force_d, 1, u_vel_d, 1);
        cublasDaxpy(handle, NUM_PARTICLES, &alpha, y_force_d, 1, v_vel_d, 1);
        cublasDaxpy(handle, NUM_PARTICLES, &alpha, z_force_d, 1, w_vel_d, 1);
        /////////////////////////////////////////
        
        // Update coordinates on Device using cublas
        /////////////////////////////////////////
        cublasDaxpy(handle, NUM_PARTICLES, &dt, u_vel_d, 1, x_cord_d, 1);
        cublasDaxpy(handle, NUM_PARTICLES, &dt, v_vel_d, 1, y_cord_d, 1);
        cublasDaxpy(handle, NUM_PARTICLES, &dt, w_vel_d, 1, z_cord_d, 1);
        /////////////////////////////////////////

        if (step % exportInterval == 0) {
            // Export particle coordinates and velocities to a file.
            cudaMemcpy(x_cord, x_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(y_cord, y_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(z_cord, z_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(u_vel, u_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(v_vel, v_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(w_vel, w_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
            std::ostringstream fileName;
            fileName << "CUDA_" << std::setfill('0') << std::setw(3) << count << ".txt";
            exportParticle(fileName.str(), NUM_PARTICLES, x_cord, y_cord, z_cord, u_vel, v_vel, w_vel);
            count++;
        }

        // Compute the mass center and maximum speed
        double mcx = 0.0;
        double mcy = 0.0;
        double mcz = 0.0;
        double maxSpeed = 0.0;
        cublasDdot(handle, NUM_PARTICLES, x_cord_d, 1, one_d, 0, &mcx);
        cublasDdot(handle, NUM_PARTICLES, y_cord_d, 1, one_d, 0, &mcy);
        cublasDdot(handle, NUM_PARTICLES, z_cord_d, 1, one_d, 0, &mcz);
        computeSpeed<<<dimblock, THREADS_PER_BLOCK>>>(NUM_PARTICLES, speed_d, u_vel_d, v_vel_d, w_vel_d);
        int idx;
        cublasIdamax(handle, NUM_PARTICLES, speed_d, 1, &idx);
        cudaMemcpy(&maxSpeed, speed_d + (idx - 1), sizeof(double), cudaMemcpyDeviceToHost);

        mcx /= NUM_PARTICLES;
        mcy /= NUM_PARTICLES;
        mcz /= NUM_PARTICLES;

        const double tEnd = tsecond();  // End timing
        std::cout << "Step " << step << " : " << tEnd - tStart << " seconds\n";
        std::cout << "Mass Center (" << mcx << "," << mcy << "," << mcz << ")\n";
        std::cout << "Maximum Speed = " << maxSpeed << std::endl;
    }
    // copy the coordinates and velocities from Devicce to Host
    cudaMemcpy(x_cord, x_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_cord, y_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z_cord, z_cord_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_vel, u_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_vel, v_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(w_vel, w_vel_d, NUM_PARTICLES * sizeof(double), cudaMemcpyDeviceToHost);

    // cleanup
    delete[] x_cord;
    delete[] y_cord;
    delete[] z_cord;
    delete[] u_vel;
    delete[] v_vel;
    delete[] w_vel;
    cudaFree(x_cord_d);
    cudaFree(y_cord_d);
    cudaFree(z_cord_d);
    cudaFree(x_force_d);
    cudaFree(y_force_d);
    cudaFree(z_force_d);
    cudaFree(u_vel_d);
    cudaFree(v_vel_d);
    cudaFree(w_vel_d);
    cudaFree(speed_d);
    return 0;
}

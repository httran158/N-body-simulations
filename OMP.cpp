#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <omp.h>
const int GSIZE = 50;
const int NUM_PARTICLES = GSIZE * GSIZE * GSIZE; // The number of Particles
const double G = 1.0e-7;          // G
const double M = 10.0;            // m
const double dt = 1.0e-2;         // time step
const int nSteps = 500;           // The number of steps
const int exportInterval = 10;    // Export Interval
const double EPSILON = 1.0e-3;    // Small parameter
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
    t = (double) (tm.tv_sec - base_sec) + ((double) (tm.tv_usec - base_usec)) /
1.0e6;
  }
  return t;
}
// export data
void exportParticle(std::string fn,int size,double *px,double *py,double *pz,double
*pu,double *pv,double *pw){
    std::ofstream ofile;
    ofile.open(fn.c_str());
    ofile << std::setprecision(16);
    ofile << std::scientific;
    for (int i = 0 ; i < size ; i++){
        ofile << px[i] << ' ' << py[i] << ' ' << pz[i] << ' '
            << pu[i] << ' ' << pv[i] << ' '<< pw[i] << std::endl;
    }
    ofile.close();
}
// main
int main(int argc, char **argv){
    int numproc = 1;
    int myid = 0;
    // Allocate x,y,z
    double *x_cord = new double[NUM_PARTICLES];
    double *y_cord = new double[NUM_PARTICLES];
    double *z_cord = new double[NUM_PARTICLES];
    
    // Allocate u,v,w
    double *u_vel = new double[NUM_PARTICLES];
    double *v_vel = new double[NUM_PARTICLES];
    double *w_vel = new double[NUM_PARTICLES];
    // set initial particle coordinate and velocity
    #pragma omp parallel for
    for (int i = 0 ; i < NUM_PARTICLES ; i++){
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
    fileName << "OMP_" << std::setfill('0') << std::setw(3) << count << "_" <<
std::setfill('0') << std::setw(2) << myid << ".txt";
    
exportParticle(fileName.str(),NUM_PARTICLES,x_cord,y_cord,z_cord,u_vel,v_vel,w_vel)
;
    count++;
    // Allocate force, f,g,h
    double *x_force = new double[NUM_PARTICLES];
    double *y_force = new double[NUM_PARTICLES];
    double *z_force = new double[NUM_PARTICLES];
    // Propagate Particles
    for (int step = 1; step <= nSteps; step++) {
        const double tStart = tsecond(); // Start timing
        // zeros force
        std::fill(x_force,x_force+NUM_PARTICLES,0.0);
        std::fill(y_force,y_force+NUM_PARTICLES,0.0);
        std::fill(z_force,z_force+NUM_PARTICLES,0.0);
        // Loop over particles that experience force
        #pragma omp parallel for
        for (int i = 0; i < NUM_PARTICLES; i++) {
            // Components of force on particle i
            // Loop over particles that exert force
            for (int j = 0; j < NUM_PARTICLES; j++) {
                // calculate the net force
                double dx = x_cord[j] - x_cord[i];
                double dy = y_cord[j] - y_cord[i];
                double dz = z_cord[j] - z_cord[i];
                double drSquared = dx * dx + dy * dy + dz * dz + EPSILON;
                double drPowerN32 = 1.0 / (drSquared * std::sqrt(drSquared));
                
                x_force[i] += G * M * M * dx * drPowerN32;
                y_force[i] += G * M * M * dy * drPowerN32;
                z_force[i] += G * M * M * dz * drPowerN32;
            }
        }
        // Update velocity and coordinate
        #pragma omp parallel for
        for (int i = 0 ; i < NUM_PARTICLES; i++) {
            u_vel[i] += x_force[i] * dt / M;
            v_vel[i] += y_force[i] * dt / M;
            w_vel[i] += z_force[i] * dt / M;
            x_cord[i] += u_vel[i] * dt;
            y_cord[i] += v_vel[i] * dt;
            z_cord[i] += w_vel[i] * dt;
        }
        if (step % exportInterval == 0){
            std::ostringstream fileName;
            fileName << "OMP_" << std::setfill('0') << std::setw(3) << count <<
"_" << std::setfill('0') << std::setw(2) << myid << ".txt";
            
exportParticle(fileName.str(),NUM_PARTICLES,x_cord,y_cord,z_cord,u_vel,v_vel,w_vel)
;
            count++;
        }
        // Compute the mass center and maximum speed
        double mcx = 0.0;
        double mcy = 0.0;
        double mcz = 0.0;
        double maxSpeed = 0.0;
        #pragma omp parallel for reduction(+:mcx,mcy,mcz) reduction(max:maxSpeed)
        for (int i = 0; i < NUM_PARTICLES; i++) {
            mcx += x_cord[i];
            mcy += y_cord[i];
            mcz += z_cord[i];
            double sp = std::sqrt(u_vel[i] * u_vel[i] + v_vel[i] * v_vel[i] +
w_vel[i] * w_vel[i]);
            maxSpeed = std::max(maxSpeed,sp);
        }
        mcx /= NUM_PARTICLES;
        mcy /= NUM_PARTICLES;
        mcz /= NUM_PARTICLES;
        const double tEnd = tsecond(); // End timing
        std::cout << "Step " << step << " : " << tEnd - tStart << " seconds\n";
        std::cout << "Mass Center (" << mcx << "," << mcy << "," << mcz << ")\n";
        std::cout << "Maximum Speed = " << maxSpeed << std::endl;
    }
    // cleanup
    delete [] x_cord;
    delete [] y_cord;
    delete [] z_cord;
    delete [] u_vel;
    delete [] v_vel;
    delete [] w_vel;
    return 0;
}

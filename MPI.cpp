#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <mpi.h>
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
    // Initialize
    MPI_Init(&argc, &argv);
    int numproc;
    int myid;
    MPI_Comm_size(MPI_COMM_WORLD,&numproc);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    
    int mystart = (NUM_PARTICLES / numproc) * myid;
    int myend;
    if (NUM_PARTICLES % numproc > myid) {
        mystart += myid;
        myend = mystart + (NUM_PARTICLES / numproc) + 1;
        } else {
        mystart += NUM_PARTICLES % numproc;
        myend = mystart + (NUM_PARTICLES / numproc);
        }
    
    int mysize = myend - mystart;
    // Allocate x,y,z
    double *x_cord = new double[mysize];
    double *y_cord = new double[mysize];
    double *z_cord = new double[mysize];
    
    // Allocate u,v,w
    double *u_vel = new double[mysize];
    double *v_vel = new double[mysize];
    double *w_vel = new double[mysize];
    // set initial particle coordinate and velocity
    for (int i = 0 ; i < mysize ; i++){
        double x = static_cast<double>((i + mystart) % GSIZE) / GSIZE;
        double y = static_cast<double>(((i + mystart) / GSIZE) % GSIZE) / GSIZE;
        double z = static_cast<double>((i + mystart) / GSIZE / GSIZE) / GSIZE;
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
    fileName << "MPI_" << std::setfill('0') << std::setw(3) << count << "_" <<
std::setfill('0') << std::setw(2) << myid << ".txt";
    exportParticle(fileName.str(),mysize,x_cord,y_cord,z_cord,u_vel,v_vel,w_vel);
    count++;
    // Allocate force, f,g,h
    double *x_force = new double[mysize];
    double *y_force = new double[mysize];
    double *z_force = new double[mysize];
    
    int bufferSize = (NUM_PARTICLES / numproc) + 1;
    double *x_cord_proc = new double[bufferSize];
    double *y_cord_proc = new double[bufferSize];
    double *z_cord_proc = new double[bufferSize];
    // Propagate Particles
    for (int step = 1; step <= nSteps; step++) {
        const double tStart = tsecond(); // Start timing
        // zeros force
        std::fill(x_force,x_force+mysize,0.0);
        std::fill(y_force,y_force+mysize,0.0);
        std::fill(z_force,z_force+mysize,0.0);
        // loop over processes
        for (int proc = 0 ; proc < numproc ; proc++){
            int remainder = NUM_PARTICLES % numproc;
    int procSize;
            if (proc < remainder){
                procSize = NUM_PARTICLES / numproc + 1;
            } else {procSize = NUM_PARTICLES / numproc;
                }
            // broadcast
            if (proc == myid){ // prepare for sending
                std::copy(x_cord,x_cord + mysize, x_cord_proc);
                std::copy(y_cord,y_cord + mysize, y_cord_proc);
                std::copy(z_cord,z_cord + mysize, z_cord_proc);
            }
            MPI_Bcast(x_cord_proc, mysize, MPI_DOUBLE, myid, MPI_COMM_WORLD);
            MPI_Bcast(y_cord_proc, mysize, MPI_DOUBLE, myid, MPI_COMM_WORLD);
            MPI_Bcast(z_cord_proc, mysize, MPI_DOUBLE, myid, MPI_COMM_WORLD);
            // Loop over particles that experience force
            for (int i = 0; i < procSize; i++) {
                // Components of force on particle i
                // Loop over particles that exert force
                for (int j = 0; j < procSize; j++) {
                    // calculate the net force
                    double dx = x_cord_proc[j] - x_cord_proc[i];
                    double dy = y_cord_proc[j] - y_cord_proc[i];
                    double dz = z_cord_proc[j] - z_cord_proc[i];
                    double drSquared = dx * dx + dy * dy + dz * dz + EPSILON;
                    double drPowerN32 = 1.0 / (drSquared * std::sqrt(drSquared));
                    
                    x_force[i] += G * M * M * dx * drPowerN32;
                    y_force[i] += G * M * M * dy * drPowerN32;
                    z_force[i] += G * M * M * dz * drPowerN32;
                }
            }
        }
        
        // Update velocity and coordinate
        for (int i = 0 ; i < mysize; i++) {
            u_vel[i] += x_force[i] * dt / M;
            v_vel[i] += y_force[i] * dt / M;
            w_vel[i] += z_force[i] * dt / M;
            x_cord[i] += u_vel[i] * dt;
            y_cord[i] += v_vel[i] * dt;
            z_cord[i] += w_vel[i] * dt;
        }
        if (step % exportInterval == 0){
            std::ostringstream fileName;
            fileName << "MPI_" << std::setfill('0') << std::setw(3) << count <<
"_" << std::setfill('0') << std::setw(2) << myid << ".txt";
            
exportParticle(fileName.str(),mysize,x_cord,y_cord,z_cord,u_vel,v_vel,w_vel);
            count++;
        }
        // Compute the mass center and maximum speed
        double mcx = 0.0;
        double mcy = 0.0;
        double mcz = 0.0;
        double maxSpeed = 0.0;
        for (int i = 0; i < mysize; i++) {
            mcx += x_cord[i];
            mcy += y_cord[i];
            mcz += z_cord[i];
            double sp = std::sqrt(u_vel[i] * u_vel[i] + v_vel[i] * v_vel[i] +
w_vel[i] * w_vel[i]);
            maxSpeed = std::max(maxSpeed,sp);
        }
        double xSum = 0.0;
        double ySum = 0.0;
        double zSum = 0.0;
        double maxV = 0.0;
        
        MPI_Reduce(&mcx, &xSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mcy, &ySum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mcz, &zSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&maxSpeed, &maxV, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
    delete [] x_force;
    delete [] y_force;
    delete [] z_force;
    delete [] x_cord_proc;
    delete [] y_cord_proc;
    delete [] z_cord_proc;
    MPI_Finalize();
    return 0;

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
    // get myid and # of processors
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
        double x = static_cast<double>((i + myid*mysize) % GSIZE) / GSIZE;
        double y = static_cast<double>(((i + myid*mysize)/ GSIZE) % GSIZE) / GSIZE;
        double z = static_cast<double>((i + myid*mysize)/ GSIZE / GSIZE) / GSIZE;
        x_cord[i] = x;
        y_cord[i] = y;
        z_cord[i] = z;
        u_vel[i] = y;
        v_vel[i] = -x;
        w_vel[i] = 0;
    }
    // export
    int step = 0;
    const double tStart = tsecond(); // Start timing
    std::ostringstream fileName;
    fileName << "MPI_" << std::setfill('0') << std::setw(3) << step << "_"
             << std::setfill('0') << std::setw(2) << myid << ".txt";
    exportParticle(fileName.str(),mysize,x_cord,y_cord,z_cord,u_vel,v_vel,w_vel);
    // Compute the mass center and maxinum speed
    double mcx = 0.0;
    double mcy = 0.0;
    double mcz = 0.0;
    double maxSpeed = 0.0;
    for (int i = 0; i < mysize; i++) {
        mcx += x_cord[i];
        mcy += y_cord[i];
        mcz += z_cord[i];
        double sp = std::sqrt(u_vel[i] * u_vel[i] + v_vel[i] * v_vel[i] + w_vel[i]
* w_vel[i]);
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
    if (myid == 0){
        std::cout << "Step " << step << " : " << tEnd - tStart << " seconds\n";
        std::cout << "Mass Center (" << xSum/NUM_PARTICLES << "," <<
ySum/NUM_PARTICLES << "," << zSum/NUM_PARTICLES << ")\n";
        std::cout << "Maximum Speed = " << maxV << std::endl;
    }
    // cleanup
    delete [] x_cord;
    delete [] y_cord;
    delete [] z_cord;
    delete [] u_vel;
    delete [] v_vel;
    delete [] w_vel;
    MPI_Finalize();
    return 0;
}

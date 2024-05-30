# N-body-simulations

N-body simulation is a technique used in physics and astronomy to model the motion of a group of particles or objects, such as stars, planets, or galaxies, that interact with each other through gravitational forces. It is a computational approach that allows researchers to study the behavior of complex systems that cannot be easily observed or analyzed analytically.

The basic idea behind N-body simulation is to divide the system into a large number of individual particles and to calculate the forces acting on each particle based on its position and mass, as well as the positions and masses of all the other particles in the system. The resulting equations of motion can then be numerically solved to predict the future positions and velocities of each particle.

One of the main challenges in N-body simulation is the sheer number of particles that need to be modeled. For example, simulating the dynamics of a galaxy with billions of stars would require billions of calculations for each time step, which can quickly become computationally prohibitive. To address this issue, various techniques have been developed to optimize the simulation and reduce the number of calculations required. These include tree-based algorithms, which group particles into clusters based on their distance from each other, and time-stepping methods, which allow for larger time steps to be taken without sacrificing accuracy.

N-body simulation has a wide range of applications in physics and astronomy. In astrophysics, it is used to study the formation and evolution of galaxies, star clusters, and planetary systems. In particle physics, it is used to simulate the behavior of particles in high-energy collisions, such as those that occur at particle accelerators like the Large Hadron Collider. It is also used in molecular dynamics simulations to study the behavior of molecules and proteins.

One of the key advantages of N-body simulation is its ability to generate highly realistic and detailed simulations of complex systems. By accurately modeling the interactions between particles, researchers can gain insights into the underlying physics and dynamics of the system, and test various hypotheses and theoretical models. This can lead to new discoveries and advancements in our understanding of the universe.

The general form of particle-particle interaction in N-body problems is given by

$$
F_i=Gm_i\sum_{j\neq i}\frac{m_j(X_j-X_i)}{|X_j-X_i|^3}
$$

where $\mathbf{X}_i=(x_i,y_i,z_i)$ are the position vectors of particles. This equation expresses $\mathbf{F}_i=(f_i,g_i,h_i)$ which is the force on particle $i$ exerted by all other particles. $G$ is a constant (gravitational constant) and $m$ are mass, which is assumed to all same. Each component are:

$$
f_i=Gm^2\sum_{j\neq i}\frac{x_j-x_i}{[(x_j-x_i)^2+(y_j-y_i)^2+(z_j-z_i)^2]^{\frac{3}{2}}}
$$

$$
g_i=Gm^2\sum_{j\neq i}\frac{y_j-y_i}{[(x_j-x_i)^2+(y_j-y_i)^2+(z_j-z_i)^2]^{\frac{3}{2}}} 
$$

$$
h_i=Gm^2\sum_{j\neq i}\frac{z_j-z_i}{[(x_j-x_i)^2+(y_j-y_i)^2+(z_j-z_i)^2]^{\frac{3}{2}}}
$$

Using the forward Euler method for the integration of differential equations, we can express the components of particle velocity $(u,v,w)$ at the end of the simulation time step $\Delta t$ as

$$
u_i(t+\Delta t)=u_i(t)+f_i\Delta t / m_i
$$

$$
v_i(t+\Delta t)=v_i(t)+g_i\Delta t / m_i
$$

$$
w_i(t+\Delta t)=w_i(t)+h_i\Delta t / m_i
$$

and the coordinates at the end of the time step as

$$
x_i(t+\Delta t) = x_i(t) + u_i(t+\Delta t)\Delta t \\
$$

$$
y_i(t+\Delta t) = y_i(t) + v_i(t+\Delta t)\Delta t \\
$$

$$
z_i(t+\Delta t) = z_i(t) + w_i(t+\Delta t)\Delta t
$$

## About this project

This project is about running N-body simulations of 1250 particles, 500 steps with different degree of parallelization.

Methods: OpenMP, MPI, OpenAcc, and CUDA.

All simulations were run on Tulane University's cluster Cypress and Louisiana State University's cluster LONI.

The code used MPI (multi-nodes), OpenMP (CPU), OpenAcc (GPU) and CUDA (GPU).

It takes more than 24 hours to run the sequential code on a single node, ~2 hours for MPI on 20 nodes, ~1.3 hours for OpenMP on 48 threads, ~4 mins for OpenAcc (0.35s/step), and ~2.5 mins (0.3s/step) for CUDA. As we can see, GPU is the more excellent way to accelerate the code compared to CPU.

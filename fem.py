import numpy
import matplotlib.pyplot as plt 
from numpy.linalg import solve
import sys
numpy.set_printoptions(linewidth=numpy.inf)

def quadrature(order):
# gaussian quadrature

    xl = numpy.zeros((4,4))
    w = numpy.zeros((4,4))

    # order 1
    xl[0,0] = 0
    w[0,0] = 2

    # order 2
    xl[0,1] = -1./numpy.sqrt(3)
    xl[1,1] = -xl[0,1]
    w[0,1] = 1.
    w[1,1] = w[0,1]

    # order 3
    xl[0,2] = -numpy.sqrt(3./5)
    xl[1,2] = 0
    xl[2,2] = -xl[0,2]
    w[0,2] = 5./9
    w[1,2] = 8./9
    w[2,2] = w[0,2]

    # order 4
    xl[0,3] = -0.8611363116
    xl[1,3] = -0.3399810436
    xl[2,3] = -xl[1,3]
    xl[3,3] = -xl[0,3]
    w[0,3] = 0.3478548451
    w[1,3] = 0.6521451549
    w[2,3] = w[1,3]
    w[3,3] = w[0,3]

    return xl[:,order-1],w[:,order-1]

def shape(x,nodes):
# basis functions

    n = len(x)
    if(nodes == 2):
        # linear
        phi = numpy.zeros((2,n))
        dphi = numpy.zeros((2,n))
        phi[0,:] = 0.5*(1-x)
        phi[1,:] = 0.5*(1+x)
        dphi[0,:] = -0.5
        dphi[1,:] = 0.5
    elif(nodes == 3):
        # quadratic
        phi = numpy.zeros((3,n))
        dphi = numpy.zeros((3,n))
        phi[0,:] = x*(x-1)*0.5
        phi[1,:] = 1-x**2
        phi[2,:] = x*(x+1)*0.5
        dphi[0,:] = x-0.5
        dphi[1,:] = -2*x
        dphi[2,:] = x+0.5
    elif(nodes == 4):
        phi = numpy.zeros((4,n))
        dphi = numpy.zeros((4,n))
        phi[0,:] = (1./9 -x*x)*(x-1)*(9./16)
        phi[1,:] = (27./16)*(1-x**2)*(1./3 - x)
        phi[2,:] = (27./16)*(1-x**2)*(1./3 + x)
        phi[3,:] = (1./9 -x*x)*(x+1)*(-9./16)
        dphi[0,:] = (-9./16)*(3*x*x-2*x-1./9)
        dphi[1,:] = (27./16)*(3*x*x-(2./3)*x-1)
        dphi[2,:] = (27./16)*(-3*x*x-(2./3)*x+1)
        dphi[3,:] = (-9./16)*(-3*x*x-2*x+1./9)

    return phi,dphi

def elem(x1,x2,nodes,order,f,k,c,b):
# local stiffness matrix

    fe = numpy.zeros((nodes,))
    ke = numpy.zeros((nodes,nodes))
    dx = (x2-x1)/2.
    xl, w = quadrature(order)
    phi, dphi = shape(xl,nodes)
    x = x1 + dx*(xl + 1)
    for i in range(nodes):
        fe[i] += dx*(f(x)*phi[i]*w).sum()
        for j in range(nodes):
            ke[i,j] += ((k(x)*dphi[i]*dphi[j]/dx + c(x)*phi[i]*dphi[j] + b(x)*phi[i]*phi[j]*dx)*w).sum()
    
    return ke,fe

def plot_sol(u,x,p):
# approx solutions

    xh = 0
    uh = 0
    du = 0
    if(p == 2):
        for j in range(0,len(x)-2,2):
            aux1 = numpy.array([x[j],x[j+1],x[j+2]])
            aux2 = numpy.array([u[j],u[j+1],u[j+2]])
            a = numpy.polyfit(aux1,aux2,p)
            xa = numpy.linspace(x[j],x[j+2],10)
            xh = numpy.hstack((xh,xa))
            uh = numpy.hstack((uh,a[0]*xa*xa + a[1]*xa + a[2]))
            du = numpy.hstack((du,2*a[0]*xa + a[1]))
    elif(p == 1):
        for j in range(len(x)-1):
            aux1 = numpy.array([x[j],x[j+1]])
            aux2 = numpy.array([u[j],u[j+1]])
            a = numpy.polyfit(aux1,aux2,p)
            xa = numpy.linspace(x[j],x[j+1],100)
            xh = numpy.hstack((xh,xa))
            uh = numpy.hstack((uh,a[0]*xa + a[1]))
            du = numpy.hstack((du,a[0]*numpy.ones((len(xa)))))
    elif(p == 3):
        for j in range(0,len(x)-3,3):
            aux1 = numpy.array([x[j],x[j+1],x[j+2],x[j+3]])
            aux2 = numpy.array([u[j],u[j+1],u[j+2],u[j+3]])
            a = numpy.polyfit(aux1,aux2,p)
            xa = numpy.linspace(x[j],x[j+3],100)
            xh = numpy.hstack((xh,xa))
            uh = numpy.hstack((uh,a[0]*xa**3 + a[1]*xa*xa + a[2]*xa + a[3]))
            du = numpy.hstack((du,3*a[0]*xa*xa + 2*a[1]*xa + a[2]))

    return xh[1:], uh[1:], du[1:]

# main
# exact solution
v = lambda x:x - numpy.sinh(x)/numpy.sinh(1.)
dv = lambda x:1 - numpy.cosh(x)/numpy.sinh(1.)

# set parameters of equation
c = lambda x: 0
b = lambda x: 1
kappa = lambda x: 1
source = lambda x: x

# set parameters of fem
# element nodes, order of quadrature, grid nodes
# number of elements, element h, size of stiffness matrix 

enodes = 2
band = enodes - 1
order = 1
nnode = 10
nelem = nnode -1
he = 1./(nnode-1)
size = band*nelem+1

# stiffness matrix generation
K = numpy.zeros((size,size))
F = numpy.zeros((size,))
xi = 0
for i in range(nelem):
    k, f = elem(xi,xi+he,enodes,order,source,kappa,c,b)
    for j in range(enodes):
        F[i*band + j] += f[j]
        for l in range(enodes):
            K[i*band + j,i*band + l] += k[j,l]
    xi +=he

# forcing Dirichlet boundaries    
u0 = 0
u1 = 0
K[0,:] = numpy.eye(size)[0,:]
# K[:,0] = numpy.eye(size)[:,0]
K[-1,:] = numpy.eye(size)[-1,:]
# K[:,-1] = numpy.eye(size)[:,-1]
F[0] = u0
F[-1] = u1

# solving system
u = solve(K,F)
# print(k,f)
# print('\n',K)
# print(F)
# print(K.shape)
# print(u)

# plot solutions
x = numpy.linspace(0,1,size)
xv = numpy.linspace(0,1,100)
xh, uh, du = plot_sol(u,x,band)
# print(uh)
plt.subplot(1,2,2);plt.plot(xh,du,label='approx du');plt.plot(xv,dv(xv),'m--',label='exact');plt.legend()
plt.subplot(1,2,1);plt.plot(xh,uh,label='approx solution');plt.plot(xv,v(xv),'m--',label='exact');plt.legend();plt.show()

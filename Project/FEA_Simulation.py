import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def mesh_generator(a, b, c, M, N, P):
    # a: length of the rectangular computational domain;
    # b: width of the rectangular computational domain;
    # c: height of the rectangular computational domain;
    # M: number of element in the x direction;
    # N: number of element in the y direction;
    # P: number of element in the z direction;
    nodeNumberCoordMatrix = np.zeros([(M + 1)*(N + 1)*(P + 1), 3]) #Matrix storing coordinates of nodes
    elmntNumberCoordMatrix = np.zeros([M * N * P, 8], dtype=int)   #Matrix storing nodes corresponding to elements
    
    #Generate node coordinates
    for k in range(0, P + 1):         #Loop through number of element in the z direction
        for j in range(0, N + 1):     #Loop through number of element in the y direction
            for i in range(0, M + 1): #Loop through number of element in the x direction
                
                #Calculate node number
                nodeNumber = i + j * (M + 1) + k * (M + 1) * (N + 1)
                
                #X, Y and Z coordinates
                x = i * a/M
                y = j * b/N
                z = k * c/P
                
                #Store coordinates of nodes in Matrix
                nodeNumberCoordMatrix[nodeNumber, 0] = x
                nodeNumberCoordMatrix[nodeNumber, 1] = y
                nodeNumberCoordMatrix[nodeNumber, 2] = z
    
    #Generate element connectivity
    for k in range(0, P):             #Loop through number of element in z direction
        for j in range(0, N):         #Loop through number of element in y direction
            for i in range(0, M):     #Loop through number of element in x direction
                
                #Calculate element number
                elmtNumber = i + j * M + k * M * N
                
                #Calculate associated nodes
                node1 = i + j * (M + 1) + k * (M + 1) * (N + 1)
                node2 = node1 + 1
                node3 = node2 + M + 1
                node4 = node3 - 1
                node5 = node1 + (M + 1) * (N + 1)
                node6 = node2 + (M + 1) * (N + 1)
                node7 = node3 + (M + 1) * (N + 1)
                node8 = node4 + (M + 1) * (N + 1)
                
                #Store element connectivity in Matrix
                elmntNumberCoordMatrix[elmtNumber, 0] = node1
                elmntNumberCoordMatrix[elmtNumber, 1] = node2
                elmntNumberCoordMatrix[elmtNumber, 2] = node3
                elmntNumberCoordMatrix[elmtNumber, 3] = node4
                elmntNumberCoordMatrix[elmtNumber, 4] = node5
                elmntNumberCoordMatrix[elmtNumber, 5] = node6
                elmntNumberCoordMatrix[elmtNumber, 6] = node7
                elmntNumberCoordMatrix[elmtNumber, 7] = node8
    
    return elmntNumberCoordMatrix, nodeNumberCoordMatrix

# Function to calculate the derivates of the shape function 
# and correspoding weight
# Output: N(8 x 8), Nxi(8 x 8), Neta(8 x 8), w(8)  
def gausspointmethod():
    N = np.zeros([8, 8])
    
    #Matrix storing partial derivative of shape functions with respect to xi1
    Nxi1 = np.zeros([8, 8])
    
    #Matrix storing partial derivatives of shape functions with respect to xi2
    Nxi2 = np.zeros([8, 8])
    
    #Matrix storing partial derivatives of shape functions with respect to xi3
    Nxi3 = np.zeros([8, 8])
    
    #Number of Gauss points
    nInt = 8
    
    #Number of nodes in element
    nElmts = 8
    
    #Gaussian weigth coefficient
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    #Gauss point coordinates xi1 (x-direction)
    xi1 = np.array([1/math.sqrt(3),   1/math.sqrt(3), 
                  -1/math.sqrt(3),  -1/math.sqrt(3),
                   1/math.sqrt(3),   1/math.sqrt(3),
                  -1/math.sqrt(3),  -1/math.sqrt(3)])
    
    #Gauss point coordinates xi2 (y-direction)
    xi2 = np.array([-1/math.sqrt(3),  1/math.sqrt(3), 
                     1/math.sqrt(3), -1/math.sqrt(3),
                    -1/math.sqrt(3),  1/math.sqrt(3),
                     1/math.sqrt(3), -1/math.sqrt(3)])
    
    #Gauss point coordinates xi3 (z-direction)
    xi3 = np.array([-1/math.sqrt(3), -1/math.sqrt(3),
                    -1/math.sqrt(3), -1/math.sqrt(3),
                     1/math.sqrt(3),  1/math.sqrt(3),
                     1/math.sqrt(3),  1/math.sqrt(3)])
    
    #Xi1 coefficient of nodes in parent domain
    coeffXi1 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    
    #Xi2 coefficient of nodes in parent domain
    coeffXi2 = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
    
    #Xi3 coefficient of nodes in parent domain
    coeffXi3 = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Looping over element nodes and Gauss points to calculate
    # values of shape functions, shape functions 
    # derivatives at individual Gauss points
    for a in range(0, nElmts):
        for l in range(0, nInt):
            N[a, l]    = 0.125 * (1.0 + coeffXi1[a]*xi1[l]) * (1.0 + coeffXi2[a]*xi2[l]) * (1.0 + coeffXi3[a]*xi3[l])
            Nxi1[a, l] = 0.125 * coeffXi1[a] * (1.0 + coeffXi2[a] * xi2[l]) * (1.0 + coeffXi3[a] * xi3[l])
            Nxi2[a, l] = 0.125 * coeffXi2[a] * (1.0 + coeffXi1[a] * xi1[l]) * (1.0 + coeffXi3[a] * xi3[l])
            Nxi3[a, l] = 0.125 * coeffXi3[a] * (1.0 + coeffXi1[a] * xi1[l]) * (1.0 + coeffXi2[a] * xi2[l])
            
    return N, Nxi1, Nxi2, Nxi3, w

# Function to calculate the jacobian and the partial derivative of the shape function 
# with respect to x, y and z
# Output: j(8), dNdx(8x8), dNdy(8x8), dNdz(8x8)
def jacobianFunction(Nxi1, Nxi2, Nxi3, nodeCoords):
    
    #Array of partial derivative of x with respect to xi1, xi2 and xi3
    xXi1 = np.zeros(8, dtype=float)
    xXi2 = np.zeros(8, dtype=float)
    xXi3 = np.zeros(8, dtype=float)
    
    #Array of partial derivative of y with respect to xi1, xi2 and xi3
    yXi1 = np.zeros(8, dtype=float)
    yXi2 = np.zeros(8, dtype=float)
    yXi3 = np.zeros(8, dtype=float)
    
    #Array of partial derivative of z with respect to xi1, xi2 and xi3
    zXi1 = np.zeros(8, dtype=float)
    zXi2 = np.zeros(8, dtype=float)
    zXi3 = np.zeros(8, dtype=float)
    
    #Array holding the values of the determinants
    j = np.zeros(8, dtype=float)
    
    #Number of Gauss points
    nInt = 8
    
    #Number of element nodes
    nElmts = 8
    
    #Matrix holding the values of partial derivative of N with respect to x, y and z
    dNdx = np.zeros([8, 8])
    dNdy = np.zeros([8, 8])
    dNdz = np.zeros([8, 8])
    
    #Loop to calculate Jacobian of each Gauss points
    for l in range(0, nInt):
        for a in range(0, nElmts):
            #Calculating partial derivative of x with respect to xi1, xi2 and xi3
            xXi1[l] = xXi1[l] + Nxi1[a, l] * nodeCoords[a][0]
            xXi2[l] = xXi2[l] + Nxi2[a, l] * nodeCoords[a][0]
            xXi3[l] = xXi3[l] + Nxi3[a, l] * nodeCoords[a][0]
            
            #Calculating partial derivative of y with respect to xi1, xi2 and xi3
            yXi1[l] = yXi1[l] + Nxi1[a, l] * nodeCoords[a][1]
            yXi2[l] = yXi2[l] + Nxi2[a, l] * nodeCoords[a][1]
            yXi3[l] = yXi3[l] + Nxi3[a, l] * nodeCoords[a][1]
            
            #Calculating partial derivative of z with respect to xi1, xi2 and xi3
            zXi1[l] = zXi1[l] + Nxi1[a, l] * nodeCoords[a][2]
            zXi2[l] = zXi2[l] + Nxi2[a, l] * nodeCoords[a][2]
            zXi3[l] = zXi3[l] + Nxi3[a, l] * nodeCoords[a][2]
        
        #Calculating determinant of 3x3 matrix holding partial derivatives of x, y and z with respect to xi1, xi2 and xi3
        x = np.matrix([[xXi1[l], xXi2[l], xXi3[l]],
                       [yXi1[l], yXi2[l], yXi3[l]],
                       [zXi1[l], zXi2[l], zXi3[l]]])
        
        #Calculating the jacobian
        j[l] = np.linalg.det(x)
    
    #Loop to calculate the partial derivative of the shape function with respect to x, y and z 
    for l in range(0, nInt):
        # Calculating the coefficient to inverse the matrix of partial derivative
        # of the cartesian coordinates in the current domain
        # with respect to the natural coordinates in the parent domain
        coeff11 = yXi2[l] * zXi3[l] - yXi3[l] * zXi2[l]
        coeff12 = yXi3[l] * zXi1[l] - yXi1[l] * zXi3[l]
        coeff13 = yXi1[l] * zXi2[l] - yXi2[l] * zXi1[l]
        coeff21 = zXi2[l] * xXi3[l] - zXi3[l] * xXi2[l]
        coeff22 = zXi3[l] * xXi1[l] - zXi1[l] * xXi3[l]
        coeff23 = zXi1[l] * xXi2[l] - zXi2[l] * xXi1[l]
        coeff31 = xXi2[l] * yXi3[l] - xXi3[l] * yXi2[l]
        coeff32 = xXi3[l] * yXi1[l] - xXi1[l] * yXi3[l]
        coeff33 = xXi1[l] * yXi2[l] - xXi2[l] * yXi1[l]
        for a in range(0, nElmts):
            # Calculating the partial derivative of the shape function with respect to x, y and z
            dNdx[a, l] = (Nxi1[a, l] * coeff11 + Nxi2[a, l] * coeff12 + Nxi3[a, l] * coeff13) / j[l]
            dNdy[a, l] = (Nxi1[a, l] * coeff21 + Nxi2[a, l] * coeff22 + Nxi3[a, l] * coeff23) / j[l]
            dNdz[a, l] = (Nxi1[a, l] * coeff31 + Nxi2[a, l] * coeff32 + Nxi3[a, l] * coeff33) / j[l]

    return j, dNdx, dNdy, dNdz

# Function to calculate B Matrix
# Output: B(6x3)
def calcOfBMatrix(nEn, nInt, dNdx, dNdy, dNdz):
    #Initilization B matrix
    B = np.zeros([8, 8], dtype=object)
    for a in range(0, nEn): #Loop over number of element nodes
        for l in range(0, nInt): #Loop over number of Gauss points
            B[a, l] = [[dNdx[a,l],         0,         0],
                       [        0, dNdy[a,l],         0],
                       [        0,         0, dNdz[a,l]],
                       [        0, dNdz[a,l], dNdy[a,l]],
                       [dNdz[a,l],         0, dNdx[a,l]],
                       [dNdy[a,l], dNdx[a,l],        0]]
    return B


#Function to calculate the local stiffness matrix
def localElementstiffnessmatrix(N, Nxi1, Nxi2, Nxi3, w, nodecoords, D, bodyForce):
    #Number of Gauss Points
    nInt = 8
    
    #Number of Element Nodes
    nEn = 8
    
    #2 DOFs per node
    nDOF = 3
    
    #Initilization k matrix
    k = np.zeros([8, 8], dtype=object)
    
    #Initilization local element stiffness matrix
    ke = np.zeros([24, 24], dtype=float)
    
    #Initilization right hand side vector
    fe = np.zeros([24, 1], dtype=float)
    
    #Calculate jacobian
    jac, dNdx, dNdy, dNdz = jacobianFunction(Nxi1, Nxi2, Nxi3, nodecoords)

    B = calcOfBMatrix(nEn, nInt, dNdx, dNdy, dNdz)

    for a in range(0, nEn): #Loop over number of element nodes
        for b in range(0, nEn): #Loop over number of element nodes
            for l in range(0, nInt): #Loop over number of Gauss points
                k[a, b] = k[a, b] + ((np.transpose(B[a][l])).dot(D)).dot(B[b][l]) * jac[l] * w[l] 
                

    
    for a in range(0, nEn): #Loop over number of element nodes
        for b in range(0, nEn): #Loop over number of element nodes
            for i in range(0, nDOF): #Loop over number of degree of freedoms
                for j in range(0, nDOF): #Loop over number of degree of freedoms
                    p = nDOF * a + i
                    q = nDOF * b + j
                    ke[p, q] = k[a, b][i][j]

    for a in range(0, nEn): #Loop over number of element nodes
        for i in range(0, nDOF): #Loop over number of degree of freedoms
            p = nDOF * a + i
            for l in range(0, nInt): #Loop over number of Gauss points
                fe[p] = fe[p] + N[a, l] * bodyForce[i] * jac[l] * w[l]
    
    return ke, fe

#Function to assemble and calculate global stiffness matrix and force vector without boundary conditions
def assembleGlobalStiffnessFoceVectorNoBC(IEN, nodenumbercoordarray, D, bodyForce):
    
    #Number of Element Nodes
    nEn = 8
    #Number of degree of freedom
    nDOF = 3
    
    #Number of total elements
    nEl = IEN.shape[0]
    
    #Number of nodes
    nSys = nodenumbercoordarray.shape[0]
    
    #Initialize stiffness matrix and force vector
    K = np.zeros([nDOF*nSys, nDOF*nSys], dtype=float)
    F = np.zeros([nDOF*nSys], dtype=float)
    
    # Call Function to calculate the derivates of the shape function 
    # and correspoding weight
    N, Nxi1, Nxi2, Nxi3, w = gausspointmethod()
    
    #Loop through number of elements
    for e in range(0, nEl):
        #Store node coordinates of respective element
        nodeCoords = np.zeros([8, 3], dtype=float)
        x = np.zeros(8, dtype=float)
        y = np.zeros(8, dtype=float)
        z = np.zeros(8, dtype=float)
        
        #Obtain node numbers for specific element
        node1 = int(IEN[e, 0])
        node2 = int(IEN[e, 1])
        node3 = int(IEN[e, 2])
        node4 = int(IEN[e, 3])
        node5 = int(IEN[e, 4])
        node6 = int(IEN[e, 5])
        node7 = int(IEN[e, 6])
        node8 = int(IEN[e, 7])
        
        #x-coordinates of nodes
        x[0] = nodenumbercoordarray[node1, 0]
        x[1] = nodenumbercoordarray[node2, 0]
        x[2] = nodenumbercoordarray[node3, 0]
        x[3] = nodenumbercoordarray[node4, 0]
        x[4] = nodenumbercoordarray[node5, 0]
        x[5] = nodenumbercoordarray[node6, 0] 
        x[6] = nodenumbercoordarray[node7, 0]
        x[7] = nodenumbercoordarray[node8, 0]
        
        #y-coordinates of nodes
        y[0] = nodenumbercoordarray[node1, 1]
        y[1] = nodenumbercoordarray[node2, 1]
        y[2] = nodenumbercoordarray[node3, 1]
        y[3] = nodenumbercoordarray[node4, 1]
        y[4] = nodenumbercoordarray[node5, 1]
        y[5] = nodenumbercoordarray[node6, 1]
        y[6] = nodenumbercoordarray[node7, 1]
        y[7] = nodenumbercoordarray[node8, 1]
        
        #z-coordinates of nodes
        z[0] = nodenumbercoordarray[node1, 2]
        z[1] = nodenumbercoordarray[node2, 2]
        z[2] = nodenumbercoordarray[node3, 2]
        z[3] = nodenumbercoordarray[node4, 2]
        z[4] = nodenumbercoordarray[node5, 2]
        z[5] = nodenumbercoordarray[node6, 2]
        z[6] = nodenumbercoordarray[node7, 2]
        z[7] = nodenumbercoordarray[node8, 2]
        
        #Assemble in node coordinates
        nodeCoords[:, 0] = x
        nodeCoords[:, 1] = y
        nodeCoords[:, 2] = z
        
        #Call function to calculate local element stiffness matrix and force vector
        ke, fe = localElementstiffnessmatrix(N, Nxi1, Nxi2, Nxi3, w, nodeCoords, D, bodyForce)

        for a in range(0, nEn): #Loop through number of nodes per elements
            for i in range(0, nDOF): #Loop through number of degree of freedom
                p = nDOF * a + i #Calculate the position of the node in local element stiffness matrix
                A = IEN[e, a] * nDOF + i % nDOF #Calculate the position of the node in the global stiffness matrix and force vector
                for b in range(0, nEn): #Loop though number of nodes per elements
                    for j in range(0, nDOF): #Loop over number of degree of freedoms
                        q = nDOF * b + j #Calculate the position of the node in local element stiffness matrix
                        B = IEN[e, b] * nDOF + j % nDOF #Calculate the position of the node in global stiffness matrix
                        K[A, B] = K[A, B] + ke[p, q] #Assemble
                F[A] = F[A] + fe[p]
    
    return K, F

#Function to assemble global stiffness matrix and force vector with boundary conditions
def assembleGlobalStiffnessFoceVector(IEN, nodenumbercoordarray, D, bodyForce, DBX, DBY, DBZ):
    #Number of nodes
    nSys = nodenumbercoordarray.shape[0]
    
    #Call function to assemble global stiffness matrix and force vector without boundary conditions
    K, F = assembleGlobalStiffnessFoceVectorNoBC(IEN, nodenumbercoordarray, D, bodyForce)
    
    #Number of degree of freedom
    nDOF = 3
    
    #Loop through the BC to change entry of force vector with boundary conditions
    for DOF in DBX:
        gA = DBX[DOF]
        for B in range(0, nDOF*nSys):
            F[B] = F[B] - K[B, DOF] * gA
    
    for DOF in DBY:
        gA = DBY[DOF]
        for B in range(0, nDOF*nSys):
            F[B] = F[B] - K[B, DOF] * gA
            
    for DOF in DBZ:
        gA = DBZ[DOF]
        for B in range(0, nDOF*nSys):
            F[B] = F[B] - K[B, DOF] * gA
    
    #Set all entries with boundary conditions to zeros for x, y and z
    for DOF in DBX:
        for B in range(0, nDOF*nSys):
            K[DOF, B] = 0.0
            K[B, DOF] = 0.0
    
    for DOF in DBY:
        for B in range(0, nDOF*nSys):
            K[B, DOF] = 0.0
            K[DOF, B] = 0.0
    
    for DOF in DBZ:
        for B in range(0, nDOF*nSys):
            K[B, DOF] = 0.0
            K[DOF, B] = 0.0
    
    #Calculate the diagonal of the matrix
    diag = 0.0
    for A in range(0, nDOF*nSys):
        diag = diag + K[A, A]
    
    diag = diag / (nDOF*nSys)
    
    #Loop to set all entries with boundary condition to diagonal value
    for DOF in DBX:
        gA = DBX[DOF]
        K[DOF, DOF] = diag
        F[DOF] = diag * gA
    
    for DOF in DBY:
        gA = DBY[DOF]
        K[DOF, DOF] = diag
        F[DOF] = diag * gA

    for DOF in DBZ:
        gA = DBZ[DOF]
        K[DOF, DOF] = diag
        F[DOF] = diag * gA
    
    return K, F

#Function to build the material matrix
#Input: 
    #E, modulus of elasticity
    #v, poisson's ratio
#Output:
    #D, Material Matrix
def materialMatrixFunc(E, v):
    lam = (v * E)/((1 + v) * (1 - 2 * v))
    G = E / (2 * (1 + v))
    D = [[lam + 2 * G,           lam,           lam,   0,   0,   0],
         [        lam,   lam + 2 * G,           lam,   0,   0,   0],
         [        lam,           lam,   lam + 2 * G,   0,   0,   0],
         [          0,             0,             0,   G,   0,   0],
         [          0,             0,             0,   0,   G,   0],
         [          0,             0,             0,   0,   0,   G]]
    
    return D

#Function to calculate the stress for each node
def stressCalcFunction(IEN, nodenumbercoordarray, U, D):
    nEn = 8 #Number of nodes per element
    nInt = 8 #Number of gauss point per element
    nEl = IEN.shape[0] #Total number of elements
    nDOF = 3 #Number of DOF
    nNodes = nodenumbercoordarray.shape[0] #Total number of nodes

    #Initialize dictionary to store final average stress for each element
    elementalNodalStress = {}
    
    #Initialize dictionary to store force at each element
    elementalForce = {}
    
    #Initialize dictionary to store force at each node
    nodalForce = {}

    # Call Function to calculate the derivates of the shape function 
    # and correspoding weight
    N, Nxi1, Nxi2, Nxi3, w = gausspointmethod()
    
    #Loop to calculate the average stress for each element
    for e in range(0, nEl):
        
        #Initialize array to temporarly store the average of all stresses of nodes in one element 
        stresses = np.zeros(6, dtype=object)
        
        #Initialize array to store the final strain for each node in element
        strains = []

        #Initialize matrix to store node coordinates of respective element
        nodeCoords = np.zeros([8, 3], dtype=float)
        
        #Initialize array to store the x, y and z coordinates of the node
        x = np.zeros(8, dtype=float)
        y = np.zeros(8, dtype=float)
        z = np.zeros(8, dtype=float)
        
        #Obtain node number for specific element
        node1 = int(IEN[e, 0])
        node2 = int(IEN[e, 1])
        node3 = int(IEN[e, 2])
        node4 = int(IEN[e, 3])
        node5 = int(IEN[e, 4])
        node6 = int(IEN[e, 5])
        node7 = int(IEN[e, 6])
        node8 = int(IEN[e, 7])
        
        #x-coordinates
        x[0] = nodenumbercoordarray[node1, 0]
        x[1] = nodenumbercoordarray[node2, 0]
        x[2] = nodenumbercoordarray[node3, 0]
        x[3] = nodenumbercoordarray[node4, 0]
        x[4] = nodenumbercoordarray[node5, 0]
        x[5] = nodenumbercoordarray[node6, 0] 
        x[6] = nodenumbercoordarray[node7, 0]
        x[7] = nodenumbercoordarray[node8, 0]
        
        #y-coordinates
        y[0] = nodenumbercoordarray[node1, 1]
        y[1] = nodenumbercoordarray[node2, 1]
        y[2] = nodenumbercoordarray[node3, 1]
        y[3] = nodenumbercoordarray[node4, 1]
        y[4] = nodenumbercoordarray[node5, 1]
        y[5] = nodenumbercoordarray[node6, 1]
        y[6] = nodenumbercoordarray[node7, 1]
        y[7] = nodenumbercoordarray[node8, 1]
        
        #z-coordinates
        z[0] = nodenumbercoordarray[node1, 2]
        z[1] = nodenumbercoordarray[node2, 2]
        z[2] = nodenumbercoordarray[node3, 2]
        z[3] = nodenumbercoordarray[node4, 2]
        z[4] = nodenumbercoordarray[node5, 2]
        z[5] = nodenumbercoordarray[node6, 2]
        z[6] = nodenumbercoordarray[node7, 2]
        z[7] = nodenumbercoordarray[node8, 2]
        
        #Assemble in node coordinates
        nodeCoords[:, 0] = x
        nodeCoords[:, 1] = y
        nodeCoords[:, 2] = z
        
        #Calculate jacobian
        jac, dNdx, dNdy, dNdz = jacobianFunction(Nxi1, Nxi2, Nxi3, nodeCoords)

        #Loop to calculate strain of each node
        for i in range(0, nEn):
            
            #Initialize array to store x, y and z displacement of each node
            ua = np.zeros([nDOF], dtype=float)
            
            #Initialize array to temporarly store strain of each node
            strain = np.zeros(nDOF*2)
            
            #Loop to calculate the strain
            for l in range(0, nInt):
                #Calculate and find the displacement of each node
                A = IEN[e, l] * nDOF
                B = IEN[e, l] * nDOF + 1
                C = IEN[e, l] * nDOF + 2
                ua[0] = U[A]
                ua[1] = U[B]
                ua[2] = U[C]
                B = [[dNdx[l,i],         0,         0],
                     [        0, dNdy[l,i],         0],
                     [        0,         0, dNdz[l,i]],
                     [        0, dNdz[l,i], dNdy[l,i]],
                     [dNdz[l,i],         0, dNdx[l,i]],
                     [dNdy[l,i], dNdx[l,i],         0]]
                #Calculate the strain for the I-th Gauss point in element
                strain = strain + np.matmul(B, ua)
            
            #Store in the final strain array
            strains.append(strain)
        #Calculate the stress for the I-th Gauss point in element
        stress = [np.matmul(D, strain) for strain in strains]
        
        #Initialize array to store x, y and z force of each element
        elementalForce[e] = np.zeros(nDOF)
        
        #Loop to calculate force of each element
        for i in range(0, nEn):
            
            #Get the node number for the specific element
            nodeNumber = IEN[e, i]
            
            #If node number is not an entry of the final node force dictionary
            if nodeNumber not in nodalForce:
                
                #Create an array of size 3
                nodalForce[nodeNumber] = np.zeros(nDOF)
                
            #Loop to calculate the nodal force
            for l in range(0, nInt):
                B = [[dNdx[i,l],         0,         0],
                     [        0, dNdy[i,l],         0],
                     [        0,         0, dNdz[i,l]],
                     [        0, dNdz[i,l], dNdy[i,l]],
                     [dNdz[i,l],         0, dNdx[i,l]],
                     [dNdy[i,l], dNdx[i,l],         0]]
                nodalForce[nodeNumber] += (np.matmul(np.transpose(B), stress[i]) * jac[l] * w[l] - N[i, l] * bodyForce * jac[l] * w[l])
        
        #Initialize M matrix
        M = np.zeros((nInt, nInt))
        
        #Loop to calculate M matrix using the quadrature rule
        for a in range(0, nEn):
            for b in range(0, nEn):
                for l in range(0, nInt):
                    M[a, b] = M[a, b] + N[a, l] * N[b, l] * jac[l] * w[l]

        #Loop through number of stresses (sigma11, sigma22, sigma33, sigma23, sigma13, sigma12)
        #For each stress use quadrature rule to calculate F vector
        for l in range(0, 6):
            F = np.zeros((nInt))
            for a in range(0, nEn):
                for b in range(0, nInt):
                    F[a] = F[a] + N[a, b] * stress[b][l] * jac[b] * w[b]

            #Solve linear system to obtain stress at elemental node
            stress_ = np.linalg.solve(M, F)
            stresses[l] = stress_
        elementalNodalStress[e] = stresses
    
    #This section will average the nodal stresses over the number of shared elements
    #Dictionary to store final stress value at each node
    stressAtEachNode = {}
    
    #Dictionary to store shared elements between nodes
    sharedElementsForAllNodes = {}
    
    #Loop over number of nodes
    for nodeNumber in range(0, nNodes):
        #Initialize array for each node in the dictionary size 6 for each sigma
        stressAtEachNode[nodeNumber] = np.zeros(6)
        
        #Initialize array to store the shared elements
        sharedElementsForOneNode = []
        
        #Loop over the number of elements
        for elmtNumber in range(0, nEl):
            for i in range(0, nEn):
                if nodeNumber == IEN[elmtNumber, i]: #Access AllElement matrix and store element if node number is in the element
                    sharedElementsForOneNode.append(elmtNumber)
        sharedElementsForAllNodes[nodeNumber] = sharedElementsForOneNode

    #Loop over the number of stresses to the sum of all sigmas for each node over the shared elements
    for l in range(0, 6):
        for nodeNumber in range(0, nNodes):
            sharedElementsForOneNode = sharedElementsForAllNodes[nodeNumber]
            for elmtNumber in sharedElementsForOneNode:
                localNodeNumber = 0 #Store number of elements which are shared by one node
                for i in range(0, nEn):
                    if nodeNumber == IEN[elmtNumber, localNodeNumber]:
                        stressAtEachNode[nodeNumber][l] += elementalNodalStress[elmtNumber][l][localNodeNumber]
                        break
                    else:
                        localNodeNumber += 1
                        
    #Calculate the average
    for nodeNumber in range(0, nNodes):
        stressAtEachNode[nodeNumber] = stressAtEachNode[nodeNumber]/len(sharedElementsForAllNodes[nodeNumber])  
    
    return stressAtEachNode, nodalForce
        
#Dirichlet function for test 1
def dirichletBoundaryConditionTest1(nodenumbercoordarray, t):
    nDOF = 3
    dirichletDicX = {}
    dirichletDicY = {}
    dirichletDicZ = {}
    nSys = nodenumbercoordarray.shape[0]
    for i in range(0, nSys):
        x = nodenumbercoordarray[i, 0]
        y = nodenumbercoordarray[i, 1]
        z = nodenumbercoordarray[i, 2]
        if x == 0.0:
            A = i * nDOF
            dirichletDicX[A] = 0.0
            if y == 0.0:
                if z == 0.0 or z == 1.0:
                    B = i * nDOF + 1
                    dirichletDicY[B] = 0.0
            elif z == 0.0:
                if y == 0.0 or y == 1.0:
                    C = i * nDOF + 2
                    dirichletDicZ[C] = 0.0
        elif x == 3.0:
            A = i * nDOF
            dirichletDicX[A] = 3.0*t
    
    return dirichletDicX, dirichletDicY, dirichletDicZ

#Dirichlet function for test 2 and 3
def dirichletBoundaryConditionTest2(nodenumbercoordarray, t):
    nDOF = 3
    dirichletDicX = {}
    dirichletDicY = {}
    dirichletDicZ = {}
    nSys = nodenumbercoordarray.shape[0]
    for i in range(0, nSys):
        x = nodenumbercoordarray[i, 0]
        if x == 0.0:
            A = i * nDOF
            dirichletDicX[A] = 0.0
            B = i * nDOF + 1
            dirichletDicY[B] = 0.0
            C = i * nDOF + 2
            dirichletDicZ[C] = 0.0
        elif x == 3.0:
            A = i * nDOF
            dirichletDicX[A] = 3.0*t
            B = i * nDOF + 1
            dirichletDicY[B] = 0.0
            C = i * nDOF + 2
            dirichletDicZ[C] = 0.0
    return dirichletDicX, dirichletDicY, dirichletDicZ

#Function to plot the original mesh vs the deformed mesh.
#Code was used and adapted from:
#https://stackoverflow.com/questions/70911608/plot-3d-cube-and-draw-line-on-3d-in-python
def plotDisplacement(elmntnumbercoord_array, nodenumbercoordarray, M, N, P, U1, U2, U3):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for i in range(M * N * P):
        x = [0.0] * 8
        y = [0.0] * 8
        z = [0.0] * 8
        x_new = [0.0] * 8
        y_new = [0.0] * 8
        z_new = [0.0] * 8

        #Obtain node number for specific element
        node1 = int(elmntnumbercoord_array[i, 0])
        node2 = int(elmntnumbercoord_array[i, 1])
        node3 = int(elmntnumbercoord_array[i, 2])
        node4 = int(elmntnumbercoord_array[i, 3])
        node5 = int(elmntnumbercoord_array[i, 4])
        node6 = int(elmntnumbercoord_array[i, 5])
        node7 = int(elmntnumbercoord_array[i, 6])
        node8 = int(elmntnumbercoord_array[i, 7])
        
        #x-coordinates
        x[0] = nodenumbercoordarray[node1, 0]
        x[1] = nodenumbercoordarray[node2, 0]
        x[2] = nodenumbercoordarray[node3, 0]
        x[3] = nodenumbercoordarray[node4, 0]
        x[4] = nodenumbercoordarray[node5, 0]
        x[5] = nodenumbercoordarray[node6, 0]
        x[6] = nodenumbercoordarray[node7, 0]
        x[7] = nodenumbercoordarray[node8, 0]

        #y-coordinates
        y[0] = nodenumbercoordarray[node1, 1]
        y[1] = nodenumbercoordarray[node2, 1]
        y[2] = nodenumbercoordarray[node3, 1]
        y[3] = nodenumbercoordarray[node4, 1]
        y[4] = nodenumbercoordarray[node5, 1]
        y[5] = nodenumbercoordarray[node6, 1]
        y[6] = nodenumbercoordarray[node7, 1]
        y[7] = nodenumbercoordarray[node8, 1]
        
        #z-coordinates
        z[0] = nodenumbercoordarray[node1, 2]
        z[1] = nodenumbercoordarray[node2, 2]
        z[2] = nodenumbercoordarray[node3, 2]
        z[3] = nodenumbercoordarray[node4, 2]
        z[4] = nodenumbercoordarray[node5, 2]
        z[5] = nodenumbercoordarray[node6, 2]
        z[6] = nodenumbercoordarray[node7, 2]
        z[7] = nodenumbercoordarray[node8, 2]

        # plotting cube
        # Initialize a list of vertex coordinates for each face
        # faces = [np.zeros([5,3])]*3
        orginalFaces = []
        orginalFaces.append(np.zeros([5,3]))
        orginalFaces.append(np.zeros([5,3]))
        orginalFaces.append(np.zeros([5,3]))
        orginalFaces.append(np.zeros([5,3]))
        orginalFaces.append(np.zeros([5,3]))
        orginalFaces.append(np.zeros([5,3]))
        
        # Bottom face
        orginalFaces[0][:,0] = [x[0], x[3], x[2], x[1], x[0]]
        orginalFaces[0][:,1] = [y[0], y[3], y[2], y[1], y[0]]
        orginalFaces[0][:,2] = [z[0], z[3], z[2], z[1], z[0]]
        
        # Top face
        orginalFaces[1][:,0] = [x[4], x[7], x[6], x[5], x[4]]
        orginalFaces[1][:,1] = [y[4], y[7], y[6], y[5], y[4]]
        orginalFaces[1][:,2] = [z[4], z[7], z[6], z[5], z[4]]
        
        # Left Face
        orginalFaces[2][:,0] = [x[0], x[3], x[7], x[4], x[0]]
        orginalFaces[2][:,1] = [y[0], y[3], y[7], y[4], y[0]]
        orginalFaces[2][:,2] = [z[0], z[3], z[7], z[4], z[0]]
        
        # Right Face
        orginalFaces[3][:,0] = [x[1], x[2], x[6], x[5], x[1]]
        orginalFaces[3][:,1] = [y[1], y[2], y[6], y[5], y[1]]
        orginalFaces[3][:,2] = [z[1], z[2], z[6], z[5], z[1]]
        
        # Front face
        orginalFaces[4][:,0] = [x[0], x[1], x[5], x[4], x[0]]
        orginalFaces[4][:,1] = [y[0], y[1], y[5], y[4], y[0]]
        orginalFaces[4][:,2] = [z[0], z[1], z[5], z[4], z[0]]
        
        # Back face
        orginalFaces[5][:,0] = [x[3], x[2], x[6], x[7], x[3]]
        orginalFaces[5][:,1] = [y[3], y[2], y[6], y[7], y[3]]
        orginalFaces[5][:,2] = [z[3], z[2], z[6], z[7], z[3]]
        ax.add_collection3d(Poly3DCollection(orginalFaces, facecolors='white', linewidths=1, edgecolors='k', alpha=.25))

        #Deformed x-coordinates
        x_new[0] = x[0] + U1[node1]
        x_new[1] = x[1] + U1[node2]
        x_new[2] = x[2] + U1[node3]
        x_new[3] = x[3] + U1[node4]
        x_new[4] = x[4] + U1[node5]
        x_new[5] = x[5] + U1[node6]
        x_new[6] = x[6] + U1[node7]
        x_new[7] = x[7] + U1[node8]

        #Deformed y-coordinates
        y_new[0] = y[0] + U2[node1]
        y_new[1] = y[1] + U2[node2]
        y_new[2] = y[2] + U2[node3]
        y_new[3] = y[3] + U2[node4]
        y_new[4] = y[4] + U2[node5]
        y_new[5] = y[5] + U2[node6]
        y_new[6] = y[6] + U2[node7]
        y_new[7] = y[7] + U2[node8]

        #Deformed z-coordinates
        z_new[0] = z[0] + U3[node1]
        z_new[1] = z[1] + U3[node2]
        z_new[2] = z[2] + U3[node3]
        z_new[3] = z[3] + U3[node4]
        z_new[4] = z[4] + U3[node5]
        z_new[5] = z[5] + U3[node6]
        z_new[6] = z[6] + U3[node7]
        z_new[7] = z[7] + U3[node8]
        
        # plotting cube
        # Initialize a list of vertex coordinates for each face
        # faces = [np.zeros([5,3])]*3
        deformedFaces = []
        deformedFaces.append(np.zeros([5,3]))
        deformedFaces.append(np.zeros([5,3]))
        deformedFaces.append(np.zeros([5,3]))
        deformedFaces.append(np.zeros([5,3]))
        deformedFaces.append(np.zeros([5,3]))
        deformedFaces.append(np.zeros([5,3]))
        
        # Bottom face
        deformedFaces[0][:,0] = [x_new[0], x_new[3], x_new[2], x_new[1], x_new[0]]
        deformedFaces[0][:,1] = [y_new[0], y_new[3], y_new[2], y_new[1], y_new[0]]
        deformedFaces[0][:,2] = [z_new[0], z_new[3], z_new[2], z_new[1], z_new[0]]
        
        # Top face
        deformedFaces[1][:,0] = [x_new[4], x_new[7], x_new[6], x_new[5], x_new[4]]
        deformedFaces[1][:,1] = [y_new[4], y_new[7], y_new[6], y_new[5], y_new[4]]
        deformedFaces[1][:,2] = [z_new[4], z_new[7], z_new[6], z_new[5], z_new[4]]
        
        # Left Face
        deformedFaces[2][:,0] = [x_new[0], x_new[3], x_new[7], x_new[4], x_new[0]]
        deformedFaces[2][:,1] = [y_new[0], y_new[3], y_new[7], y_new[4], y_new[0]]
        deformedFaces[2][:,2] = [z_new[0], z_new[3], z_new[7], z_new[4], z_new[0]]
        
        # Right Face
        deformedFaces[3][:,0] = [x_new[1], x_new[2], x_new[6], x_new[5], x_new[1]]
        deformedFaces[3][:,1] = [y_new[1], y_new[2], y_new[6], y_new[5], y_new[1]]
        deformedFaces[3][:,2] = [z_new[1], z_new[2], z_new[6], z_new[5], z_new[1]]
        
        # Front face
        deformedFaces[4][:,0] = [x_new[0], x_new[1], x_new[5], x_new[4], x_new[0]]
        deformedFaces[4][:,1] = [y_new[0], y_new[1], y_new[5], y_new[4], y_new[0]]
        deformedFaces[4][:,2] = [z_new[0], z_new[1], z_new[5], z_new[4], z_new[0]]
        
        # Back face
        deformedFaces[5][:,0] = [x_new[3], x_new[2], x_new[6], x_new[7], x_new[3]]
        deformedFaces[5][:,1] = [y_new[3], y_new[2], y_new[6], y_new[7], y_new[3]]
        deformedFaces[5][:,2] = [z_new[3], z_new[2], z_new[6], z_new[7], z_new[3]]
        
        ax.add_collection3d(Poly3DCollection(deformedFaces, facecolors='white', linewidths=1, edgecolors='r', alpha=.25)) 

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.5)
    ax.set_zlim(0, 1.5)
    plt.show()

#Function to create VTK file for paraview (Stress, displacement)
def writeVTK(allNode, U, allElmt, stressAtEachNode, nodalForce, tNew):
    nodeDisp = {}
    nNodes = allNode.shape[0]
    nElmts = allElmt.shape[0]
    nDOF = 3
    for i in range(nNodes):
        A = i * nDOF
        B = i * nDOF + 1
        C = i * nDOF + 2
        dispX = U[A]
        dispY = U[B]
        dispZ = U[C]
        nodeDisp[i] = (dispX, dispY, dispZ)
    with open("0_" + "{:.3f}".format(tNew).replace('.', '') + ".vtk",
              'w') as VTK:
        VTK.write("# vtk DataFile Version 2.0\n")
        VTK.write("Data at time " + "{:.3f}".format(tNew) + "\n")
        VTK.write("ASCII\n")
        VTK.write("DATASET UNSTRUCTURED_GRID\n")
        VTK.write("POINTS " + str(len(allNode)) + " float\n")
        for i in range(nNodes):
            A = i * nDOF
            B = i * nDOF + 1
            C = i * nDOF + 2
            coordStr = str(allNode[i, 0] + U[A]) + " " \
                + str(allNode[i, 1] + U[B]) + " " \
                + str(allNode[i, 2] + U[C]) + "\n"
            VTK.write(coordStr)
        VTK.write("CELLS " + str(nElmts) + " "
                  + str(8*nElmts + nElmts) + "\n")
        for i in range(nElmts):
            elmtStr = str(8) + " " + str(allElmt[i][0]) + " " \
                                   + str(allElmt[i][1]) + " " \
                                   + str(allElmt[i][2]) + " " \
                                   + str(allElmt[i][3]) + " " \
                                   + str(allElmt[i][4]) + " " \
                                   + str(allElmt[i][5]) + " " \
                                   + str(allElmt[i][6]) + " " \
                                   + str(allElmt[i][7]) + "\n"
            VTK.write(elmtStr)
        VTK.write("CELL_TYPES " + str(nElmts) + "\n")
        for i in range(nElmts):
            VTK.write("12\n")
        
        VTK.write("POINT_DATA " + str(nNodes) + "\n")
        VTK.write("SCALARS Stress11 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][0])+"\n")
        VTK.write("SCALARS Stress22 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][1])+"\n")
        VTK.write("SCALARS Stress33 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][2])+"\n")
        VTK.write("SCALARS Stress23 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][3])+"\n")
        VTK.write("SCALARS Stress13 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][4])+"\n")
        VTK.write("SCALARS Stress12 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(stressAtEachNode[i][5])+"\n")

        VTK.write("SCALARS Force1 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(nodalForce[i][0])+"\n")
        VTK.write("SCALARS Force2 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(nodalForce[i][1])+"\n")
        VTK.write("SCALARS Force3 float 1\n")
        VTK.write("LOOKUP_TABLE default\n")
        for i in range(nNodes):
            VTK.write(str(nodalForce[i][2])+"\n")
        
        VTK.write("VECTORS Displacement float\n")
        for i in range(nNodes):
            dispStr = str(nodeDisp[i][0]) + " " \
                + str(nodeDisp[i][1]) + " " \
                + str(nodeDisp[i][2]) + "\n"
            VTK.write(dispStr)
    return 0

'''
#Test 1
a = 3
b = 1
c = 1

M = 12
N = 4
P = 4

#Material properties
E = 1.0e+9
v = 0.25
DforDisp = materialMatrixFunc(E, v)
DforStress = materialMatrixFunc(E, v)
DforStress[3][3] = 2.0 * DforStress[3][3]
DforStress[4][4] = 2.0 * DforStress[4][4]
DforStress[5][5] = 2.0 * DforStress[5][5]
bodyForce = np.zeros(3, dtype=float)

elmntNumberCoordMatrix, nodeNumberCoordMatrix = mesh_generator(a, b, c, M, N, P)

#Time increment
deltaT = 0.02

#Total time step 
totalTime = 0.2

#Current time step
t = 0.0

while t <= totalTime:
    dirichletDicX, dirichletDicY, dirichletDicZ = dirichletBoundaryConditionTest1(nodeNumberCoordMatrix, t)
    K, F = assembleGlobalStiffnessFoceVector(elmntNumberCoordMatrix, nodeNumberCoordMatrix, DforDisp, bodyForce, dirichletDicX, dirichletDicY, dirichletDicZ)
    U = np.linalg.inv(K).dot(F)
    stressAtEachNode, nodalForce = stressCalcFunction(elmntNumberCoordMatrix, nodeNumberCoordMatrix, U, DforStress)
    U1 = U[::3]
    U2 = U[1::3]
    U3 = U[2::3]
    plotDisplacement(elmntNumberCoordMatrix, nodeNumberCoordMatrix, M, N, P, U1, U2, U3)
    writeVTK(nodeNumberCoordMatrix, U, elmntNumberCoordMatrix, stressAtEachNode, nodalForce, t)
    t = t + deltaT
'''

'''
#Test 2
a = 3
b = 1
c = 1

M = 12
N = 4
P = 4

#Material properties
E = 1.0e+9
v = 0.25
DforDisp = materialMatrixFunc(E, v)
DforStress = materialMatrixFunc(E, v)
DforStress[3][3] = 2.0 * DforStress[3][3]
DforStress[4][4] = 2.0 * DforStress[4][4]
DforStress[5][5] = 2.0 * DforStress[5][5]
bodyForce = np.zeros(3, dtype=float)

elmntNumberCoordMatrix, nodeNumberCoordMatrix = mesh_generator(a, b, c, M, N, P)

#Time increment
deltaT = 0.02

#Total time step 
totalTime = 0.2

#Current time step
t = 0.1

while t <= totalTime:
    dirichletDicX, dirichletDicY, dirichletDicZ = dirichletBoundaryConditionTest2(nodeNumberCoordMatrix, t)
    K, F = assembleGlobalStiffnessFoceVector(elmntNumberCoordMatrix, nodeNumberCoordMatrix, DforDisp, bodyForce, dirichletDicX, dirichletDicY, dirichletDicZ)
    U = np.linalg.inv(K).dot(F)
    stressFinal, nodalForce = stressCalcFunction(elmntNumberCoordMatrix, nodeNumberCoordMatrix, U, DforStress)
    print(nodalForce[100])
    assert 1 == 0
    U1 = U[::3]
    U2 = U[1::3]
    U3 = U[2::3]
    plotDisplacement(elmntNumberCoordMatrix, nodeNumberCoordMatrix, M, N, P, U1, U2, U3)
    writeVTK(nodeNumberCoordMatrix, U, elmntNumberCoordMatrix, stressFinal, nodalForce, t)
    t = t + deltaT
'''

'''
#Test 3
a = 3
b = 1
c = 1

M = 12
N = 4
P = 4

#Material properties
E = 1.0e+9
v = 0.25
DforDisp = materialMatrixFunc(E, v)
DforStress = materialMatrixFunc(E, v)
DforStress[3][3] = 2.0 * DforStress[3][3]
DforStress[4][4] = 2.0 * DforStress[4][4]
DforStress[5][5] = 2.0 * DforStress[5][5]
bodyForce = np.array([0.0, 0.0, -0.5e+08])

elmntNumberCoordMatrix, nodeNumberCoordMatrix = mesh_generator(a, b, c, M, N, P)

#Time increment
deltaT = 0.02

#Total time step 
totalTime = 0.2

#Current time step
t = 0.1

while t <= totalTime:
    dirichletDicX, dirichletDicY, dirichletDicZ = dirichletBoundaryConditionTest2(nodeNumberCoordMatrix, t)
    K, F = assembleGlobalStiffnessFoceVector(elmntNumberCoordMatrix, nodeNumberCoordMatrix, DforDisp, bodyForce, dirichletDicX, dirichletDicY, dirichletDicZ)
    U = np.linalg.inv(K).dot(F)
    stressFinal, nodalForce = stressCalcFunction(elmntNumberCoordMatrix, nodeNumberCoordMatrix, U, DforStress)
    U1 = U[::3]
    U2 = U[1::3]
    U3 = U[2::3]
    print(nodalForce[100])
    assert 1 == 0
    plotDisplacement(elmntNumberCoordMatrix, nodeNumberCoordMatrix, M, N, P, U1, U2, U3)
    writeVTK(nodeNumberCoordMatrix, U, elmntNumberCoordMatrix, stressFinal, nodalForce, t)
    t = t + deltaT
'''
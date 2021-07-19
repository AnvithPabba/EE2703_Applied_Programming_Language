# EE2703 Assignmnt 2
# EE19B070
# Solving the circuit


import sys
import numpy
import math
try:
    arg1=sys.argv[1]     #input the file to read, from command line
except Exception:
    print("Please enter a filename as command line arguement") #Error if no arguement is passed in command line
    exit()
try:
    with open(arg1) as f:     #opens the file
        lines=f.readlines()   #creates the list of lines with each line as strings
except IOError:
    print("Invalid Filename")  #Error if the input filename is not in directory
    exit()
CIRCUIT=".circuit"   #variables
END=".end"
AC_KEY=".ac"
start=-1
end=-2
ac=0    #ac stores 0 if there's no AC source
AC=0
n1=0
n2=0
for a in range(0,len(lines)):
    if lines[a][:len(CIRCUIT)]==CIRCUIT:
        n1 += 1     #identifies if multiple .circuits are epresent and gives an error if more than 1 is present
        start=a     #identifies start line of the block
    elif lines[a][:len(END)]==END:
        n2 += 1     #identifies if multiple .ends are epresent and gives an error if more than 1 is present
        end=a       #identifies end line of the block
    elif lines[a][:len(AC_KEY)]==AC_KEY:
        if AC!=0:
            continue
        ac=lines[a].split()   #identifies ac source
        AC+=a
if AC<end:
    AC=0
if start>end or n1>1 or n2>1:
    print("Invalid Circuit definition.")   #Validates circuit block
    exit()
circuit_block=lines[start+1:end]     #circuit block

parsing=[]     #list of lists of information from netlist file
for i in range(0,len(circuit_block)):
    line=circuit_block[i].split("#")[0].split() #removes the comments in each line, if present
    if len(line)==0:   #this is for certain cases where an entire line is a comment
        continue
    l2=[]
    for j in range(0,len(line)):
        l2.append(line[j])
    parsing.append(l2)
#for DC
if AC==0:
    for i in parsing:
        if i[0]=='.ac':
            z=parsing.index(i)
            del parsing[z]


class CircElement():                    #Defining a class fr each element of the circuit (Not Necessary)
    
    def __init__(self,info):
        self.name = info[0]             #Assigning certain parameters of the elements
        self.terminal1 = info[1]
        self.terminal2 = info[2]
        self.value = info[3]

k = (len(parsing));      #initialising a list that contains all the elements
element = ['']*k;     
for n in range(k):
    element[n] = CircElement(parsing[n]);


#Creating a dictionary to store all the nodes
nodes={0:'GND'}   #initialising a dictionary that stores the node values
V=['V0']      # X matrix : node voltages + currents through voltage sources
for i in range(0,len(parsing)):
    if parsing[i][1] in nodes.values():
        if parsing[i][2] in nodes.values():
            continue
        else:
            nodes[len(nodes)]=parsing[i][2]
            V.append('V'+nodes[len(nodes)-1])
    else:
        nodes[len(nodes)]=parsing[i][1]
        V.append('V'+nodes[len(nodes)-1])
nodekey=list(nodes.keys())     #list of all the keys of the dictionary nodes
nodeval=list(nodes.values())   #list of all the values of the dictionary nodes
Nv=0    # no. of voltage sources

for i in range(0,len(parsing)):
    if parsing[i][0][0]=='V':
        Nv+=1
        V.append('i '+'through source '+(parsing[i][0]))
Neq=len(nodes)+Nv    # total no. of equations which is also the dimension of matrix

#Creating matrices M and B of required dimensions with all entries as 0
Mat=[]
M=[]
for i in range(0,Neq):
    Mat.append(complex(0,0))
for i in range(0,Neq):
    M.append(Mat)
M=numpy.array(M)
B=numpy.array(Mat)

if AC!=0:
    w=2*math.pi*float(ac[-1])

#Calculating the no. of current sources and updating the B matrix at required nodes other than GND
Ni=0    # no. of current sources
for i in range(0,len(parsing)):
    if parsing[i][0][0]=='I':
        if parsing[i][1]!='GND':
            if AC!=0:
                B[nodekey[nodeval.index(parsing[i][1])]]-=complex((float(parsing[i][-2])/2)*math.cos(float(parsing[i][-1])),float(parsing[i][-2])*math.sin(float(parsing[i][-1])))
            else:
                B[nodekey[nodeval.index(parsing[i][1])]]-=float(parsing[i][-1])
        if parsing[i][2]!='GND':
            if AC!=0:
                B[nodekey[nodeval.index(parsing[i][2])]]+=complex((float(parsing[i][-2])/2)*math.cos(float(parsing[i][-1])),float(parsing[i][-2])*math.sin(float(parsing[i][-1])))
            else:
                B[nodekey[nodeval.index(parsing[i][2])]]+=float(parsing[i][-1])
        Ni+=1

# representing the voltage sources and currents through them
p=0    # dummy variable
for i in range(0,len(parsing)):
    if parsing[i][0][0]=='V':
        M[len(nodes)+p][nodekey[nodeval.index(parsing[i][1])]]+=1
        M[len(nodes)+p][nodekey[nodeval.index(parsing[i][2])]]-=1
        if AC!=0:
            B[len(nodes)+p]+=complex(float(parsing[i][-2])*math.cos(float(parsing[i][-1])),float(parsing[i][-2])*math.sin(float(parsing[i][-1])))
        else:
            B[len(nodes)+p]+=float(parsing[i][-1])
        if parsing[i][1]!='GND':
            M[nodekey[nodeval.index(parsing[i][1])]][len(nodes)+p]-=1
        if parsing[i][2]!='GND':
            M[nodekey[nodeval.index(parsing[i][2])]][len(nodes)+p]+=1
        p+=1

#dictinaries for values of resistors, capacitors and inductors
resistors={}
for i in range(0,len(parsing)):
    if parsing[i][0][0]=='R':
        resistors[parsing[i][0]]=float(parsing[i][-1])

capacitors={}
for i in range(0,len(parsing)):
    if parsing[i][0][0]=='C':
        capacitors[parsing[i][0]]=complex(0,-1/(w*float(parsing[i][-1])))

inductors={}
for i in range(0,len(parsing)):
    if parsing[i][0][0]=='L':
        inductors[parsing[i][0]]=complex(0,w*float(parsing[i][-1]))

# altering the corresponding elements of matrix by +or-(1/impedance) where impedence is present
#in other words, writing equations in matrix form
for j in range(0,len(nodes)):
    a=nodes[j]
    if a=='GND':
        M[0][0]=1
    else:
        for i in range(0,len(parsing)):
            if parsing[i][1]==a:
                b=parsing[i][2]
                if parsing[i][0][0]=='V' or parsing[i][0][0]=='I':
                    continue
            elif parsing[i][2]==a:
                b=parsing[i][1]
                if parsing[i][0][0]=='V' or parsing[i][0][0]=='I':
                    continue
            else:
                continue
            c=nodekey[nodeval.index(b)]
            if parsing[i][0][0]=='R':
                M[j][j]+=1/resistors[parsing[i][0]]
                M[j][c]-=1/resistors[parsing[i][0]]
            elif parsing[i][0][0]=='C':
                M[j][j]+=1/capacitors[parsing[i][0]]
                M[j][c]-=1/capacitors[parsing[i][0]]
            elif parsing[i][0][0]=='L':
                M[j][j]+=1/inductors[parsing[i][0]]
                M[j][c]-=1/inductors[parsing[i][0]]

x = numpy.linalg.solve(M, B)  #solving linear equations
print('Matrix A is:\n{}\n' .format(M))
print('Matrix b is:\n{}\n' .format(B))

# printing the unkonowns
for i in range(0,len(V)):
    if AC==0:
        print('{} = {}'.format(V[i],numpy.real(x[i])))
    else:
        print('{} = {}'.format(V[i],x[i]))

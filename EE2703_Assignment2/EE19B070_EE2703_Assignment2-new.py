import sys
import numpy as np
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
for a in range(0,len(lines)):
    if lines[a][:len(CIRCUIT)]==CIRCUIT:
        start=a     #identifies start line of the block
    elif lines[a][:len(END)]==END:
        end=a       #identifies end line of the block
    elif lines[a][:len(AC_KEY)]==AC_KEY:
        if AC!=0:
            continue
        ac=lines[a].split()   #identifies ac source
        AC+=a
if AC<end:
    AC=0
if start>end:
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


class CircElement():
    
    def __init__(self,info):
        self.name = info[0]
        if info[1] == 'GND':
           self.terminal1 = '0'
        else: 
            self.terminal1 = info[1]
        
        if info[2] == 'GND':
           self.terminal2 = '0'
        else: 
            self.terminal2 = info[2]

        self.value = info[3]

k = (len(parsing));
element = ['']*k;     
for n in range(k):
    element[n] = CircElement(parsing[n]);

NumVolSource = 0;
for n in range(k):
    h = element[n].name;
    if h[0] == "V":
        NumVolSource += 1

num_nodes = 0
for n in range(len(parsing)):

        if (float(element[n].terminal1) >= num_nodes or float(element[n].terminal1) >= num_nodes) and float(element[n].terminal1) > float(element[n].terminal2) :
            num_nodes = float(element[n].terminal1)
        elif (float(element[n].terminal1) >= num_nodes or float(element[n].terminal1) >= num_nodes) and float(element[n].terminal1) < float(element[n].terminal2) :
            num_nodes = float(element[n].terminal2)
        else:
            continue
print("The total number of nodes present are : {} , (excluding GND) ".format(num_nodes))


Mat_length = int(num_nodes + NumVolSource)
B = np.zeros((int(Mat_length),1))
A = np.zeros((int(Mat_length),int(Mat_length)))

if AC!=0:
    w = math.pi*float(2*ac[-1])

for k2 in range(int(Mat_length)):
    for k1 in range(int(Mat_length)):

        if k2 == k1 and k1 < num_nodes:
            for n in range(len(parsing)):
                if float(element[n].terminal1) ==k1+1 or float(element[n].terminal2) ==k1+1:
                    if element[n].name[0] =='R':
                        A[k2][k1] += 1/(float(element[n].value))
                    elif element[n].name[0] =='C':
                        A[k2][k1] += float(w*(element[n].value))
                    elif element[n].name[0] =='L':
                        A[k2][k1] += 1/float(w*(element[n].value))

        if k2!=k1 and k1 < num_nodes:
            for n in range(len(parsing)):
                if (float(element[n].terminal1) ==k1+1 and float(element[n].terminal2) ==k2+1) or (float(element[n].terminal1) ==k2+1 and float(element[n].terminal2) ==k1+1):
                    if element[n].name[0] =='R':
                        A[k2][k1] -= 1/(float(element[n].value))
                    elif element[n].name[0] =='C':
                        A[k2][k1] -= float(w*(element[n].value))
                    elif element[n].name[0] =='L':
                        A[k2][k1] -= 1/float(w*(element[n].value))

        if k2 < num_nodes and k1 >= num_nodes:
            for n in range(len(parsing)):
                if (float(element[n].terminal1) ==k1+1 and float(element[n].terminal2) ==k2+1) or (float(element[n].terminal1) ==k2+1 and float(element[n].terminal2) ==k1+1):
                    if element[n].name[0] =='V' and float(element[n].terminal1) ==k1+1:
                        A[k2][k1] -= float(element[n].value)
                    if element[n].name[0] =='V' and float(element[n].terminal2) ==k1+1:
                        A[k2][k1] += float(element[n].value)

    if k2>= num_nodes:
        for n in range(len(parsing)):
            if (element[n].name[0] == 'V') and (float(element[n].name[1]) == k2 - num_nodes +1):
                if int(float(element[n].terminal2) - 1) !=-1:
                    A[k2][int(float(element[n].terminal2) - 1)] = 1
                else:
                    A[k2][int(float(element[n].terminal2) - 1)] = 0

                if int(float(element[n].terminal1) - 1) !=-1:
                    A[k2][int(float(element[n].terminal1) - 1)] = -1
                else:
                    A[k2][int(float(element[n].terminal1) - 1)] = 0


for k1 in range(int(Mat_length)):
    if k1 < num_nodes:
        for n in range(len(parsing)):
            if element[n].name[0] =='I':
                if (float(element[n].terminal1) ==k1+1):
                    B[k1][1] -= float(element[n].value)
                if (float(element[n].terminal2) ==k1+1):
                    B[k1][1] += float(element[n].value)

    if k1 >= num_nodes:
        for n in range(len(parsing)):
            if element[n].name[0] =='V':
                if (int(element[n].name[1]) == k1 - num_nodes+1):
                    B[k1][0] = float(element[n].value)

print(NumVolSource)
print(A)
print(B)

x = np.linalg.solve(A,B)

for i in range(0,NumVolSource):
    if AC==0:
        print('V{} = {}'.format(i,np.real(x[i][0])))
    else:
        print('V{} = {}'.format(i,x[i][0]))



    









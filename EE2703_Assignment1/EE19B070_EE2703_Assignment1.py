"""
        EE2703 Applied Programming Lab - 2020
            EE19B070 Assignment-1 Solution
                    Anvith Pabba
"""

from sys import argv, exit

####The only extra added code is from line 47 to line 52, the rest is taken from the sample solution.

"""
It's recommended to use constant variables than hard-coding them everywhere.
For example, if you decide to change the command from '.circuit' to '.start' later,
    you only need to change the constant
"""
CIRCUIT = '.circuit'
END = '.end'

"""
It's a good practice to check if the user has given required and only the required inputs
Otherwise, show them the expected usage.
"""
if len(argv) != 2:
    print('\nUsage: %s <inputfile>' % argv[0])
    exit()

"""
The use might input a wrong file name by mistake.
In this case, the open function will throw an IOError.
Make sure you have taken care of it using try-catch
"""

try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1; end = -2
        for line in lines:              # extracting circuit definition start and end lines
            if CIRCUIT == line[:len(CIRCUIT)]:
                start = lines.index(line)
            elif END == line[:len(END)]:
                end = lines.index(line)
                break
        if start >= end:                # validating circuit block
            print('Invalid circuit definition')
            exit(0)

#Added Code Start         
 
        parsing = []                                    #Initialising The Parsing list
        for n in range(start+1,end):                    #Parsing Should Only Contain The Lines In Between .circuit And .end, Not The Comments Above Or Below
            remove_comment = lines[n].split("#")        #To Remove The Comments That May Be In Each Individual Line
            parsing.append(remove_comment[0].split())   #In A Line, Remove The Comment And Then Split It And Then Append It
                                                        #Remove_comment[1] Contains The Comment, Whereas Remove_comment[0] Contains The Circuit Information
        print("\nThe Parsed Ciruit, With Tokens Stored Is:\n{}" .format(parsing)) #Printing The Output

        class CircElement():                    #Defining a class for an element of the circuit   
            def __init__(self,info):
                self.name = info[0]             #Assigning certain parameters of the elements
                self.terminal1 = info[1]
                self.terminal2 = info[2]
                self.value = info[3]

        k = (len(parsing));      #initialising a list that contains all the elements
        element = ['']*k;     
        for n in range(k):
            element[n] = CircElement(parsing[n]);

        print("\nTesting if the class is properly defined,\nvalue of R2 is:",element[1].value)



#Added code end

        print("\nThe reversed input is:")
        for line in reversed([' '.join(reversed(line.split('#')[0].split())) for line in lines[start+1:end]]):
            print(line)                 # print output

except IOError:
    print('Invalid file')
    exit()

# EE19B070
# Anvith Pabba

"""
Here, the actual work is done in a single line (before "# print output").
It takes care of
    1. Removing comments
    2. Splitting the words
    3. Reversing words
    4. Reversing lines
    
Compare this with how you would do the same in a C code.
"""

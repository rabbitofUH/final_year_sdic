#!/home/thierry/anaconda3/bin/python
import struct
import numpy as np
import sys
import matplotlib.pyplot as plt

#Global variable
AutoFs = 1 / 41.3414
def main(argv):
    [data, nbiter, fulliter, ggp] = read_ggp()
    gwpdim = int(data[3])
    time=np.linspace(data[0]*AutoFs,data[1]*AutoFs,fulliter)
    print("Number of gwp: " + str(gwpdim))
    [geo_iter, nbatom, all_geo] = extract_geo(gwpdim, nbiter)
    print("Numger of atoms: " + str(nbatom))
    if (geo_iter != nbiter):
        print("ERROR: inconsistency of size between ggp and all_geo")
    while (True):
        print("1 = print dynamics time information")
        print("2 = print ggp")
        print("3 = Evaluate product channel fraction")
        print("\n0 = exit")

        b = input("Task: ")
        if b == "1":
            print("Tinit  = " + str(data[0]) + " a.u.")
            print("Tfinal = " + str(data[1]) + " a.u.")
            print("step   = " + str(data[2]) + " a.u.")
            print("nbiter = " + str(nbiter) + " a.u.\n")
        elif b == "2":
            index = int(input("Which gwp (1-" + str(gwpdim) + ")?")) - 1
            time = int(input("Which iteration (0-" + str(nbiter - 1) + ")?"))
            print(ggp[index][time])
            print("")
        elif b == "3":
            print("Choose atom for distance (1-" + str(nbatom) + ")")
            atom1 = int(input("Atom 1:"))-1
            atom2 = int(input("Atom 2:"))-1
            max_dist = float(input("Criteria for product in angstrom:"))
            if (0 <= atom1 < nbatom and 0 <= atom2 < nbatom):
                distance = checkddtraj(gwpdim, nbiter, atom1, atom2, time, all_geo)
                prod_dens = checkbondbreak(max_dist, gwpdim, nbiter,time, distance, ggp)
                plotprod(time[0:nbiter], prod_dens)
            else:
                print("Atom number not recognized")
        elif b == "0":
            exit()
        else:
            print("Task number not recognized")


def read_ggp():
    # initialise variable
    i = 0
    iter = 0
    nbdata = 5
    f = open(r"check", "rb+")
    data_check = [False for il in range(nbdata)]
    # All tag are written with 6 characters long
    data_tag = [b'tinit>', b'tfinal', b'out1> ', b'gwpdim', b'ggp>  ']
    # d= double precision and i= integer
    data_type = ['d', 'd', 'd', 'i', 'd']
    data = np.zeros(nbdata - 1)
    ggp = []

    while (True):
        f.seek(i)
        data_read = f.read(1)
        if data_read == b'<':
            f.seek(i + 1)
            data_read = f.read(6)
            for ind_data in range(nbdata):
                if data_read == data_tag[ind_data]:
                    i = i + 32 + 4 + 4
                    f.seek(i)
                    if ind_data < 4:
                        if (data_type[ind_data] == 'd'):
                            data_binary = f.read(8)
                        elif (data_type[ind_data] == 'i'):
                            data_binary = f.read(4)
                        data[ind_data] = struct.unpack(data_type[ind_data], data_binary)[0]
                        data_check[ind_data] = True
                        if ind_data == 3:
                            gwpdim = int(data[ind_data])
                    # read ggp. Assume gwpdim is found 1st. Will fail if gwpdim is not found
                    if ind_data == 4:
                        ggptemp = np.zeros(gwpdim)
                        for index in range(gwpdim):
                            data_binary = f.read(8)
                            ggptemp[index] = struct.unpack(data_type[ind_data], data_binary)[0]
                        ggp.append(ggptemp)
                        data_check[ind_data] = True
                        iter = iter + 1
        # check for end of file
        elif data_read == b'':
            break
        i = i + 1
    i = 0
    ggp = np.array(ggp).T
    # Check that everything was found and is consistent
    nbiter = int((data[1] - data[0]) / data[2]) + 1
    if iter != nbiter:
        print('read_check: WARNING! ' + str(iter) + ' dataset were found and ' + str(nbiter) + ' dataset were expected')
    for ind_data in range(nbdata):
        if (data_check[ind_data] == False):
            print('read_check: Error: ' + str(data_tag[ind_data]) + ' not found')
            exit()
    print('read_check: all data found with ' + str(iter) + ' ggp datasets')
    return [data, iter, nbiter, ggp]


def extract_geo(nbfile, nbiter):
    b = []
    full_iter = []
    for index in range(nbfile):
        filename = 'ddtraj_' + str(index + 1) + '.txt'
        file = open(filename, "r")
        curr_file = []
        curr_geo = []
        iter = 0
        iatom = 0
        for line in file.readlines():
            curline = line.split()
            if (len(curline) == 6) and (curline[0] == "Number") and (iatom > 0):
                maxatom = iatom
                iatom = 0
                curr_file.append(np.array(curr_geo))
                curr_geo = []
                iter = iter + 1
            if (len(curline) == 6) and (curline[0] != "Number"):
                # extract cartesian geometry line by line and convert it into float (need to check if it works properly)
                # alternative to float conversion with astype is map (need to look online the structure)
                curr_xyz = list(map(float, curline[3:6]))
                curr_geo.append(curr_xyz)
                iatom = iatom + 1
        curr_file.append(np.array(curr_geo))
        iter = iter + 1
        b.append(np.array(curr_file))
        # Put a printing statement around there to see if data extracted is correct
        file.close()
        full_iter.append(iter)
    all_check = all(element == full_iter[0] for element in full_iter)
    if (all_check):
        print("extract_geo: Extrated all geometry with " + str(full_iter[0]) + " elements per file")
    else:
        print("extract_geo: ERROR: Inconsistency in number of geometries for each file")
        print(full_iter)
    print("Geometry matrix shape:" + str(np.shape(b)))
    return [full_iter[0], maxatom, b]


def checkddtraj(nbfile, nbiter, atom1, atom2, time, b):
    distance = np.zeros((nbfile, nbiter))
    outputdist = open("bond_length.dat","w")
    for iter in range(nbiter):
        outputdist.write("%7.3f" % (int(time[iter])))
        for index in range(nbfile):
            distance[index][iter] = np.linalg.norm(b[index][iter][atom1] - b[index][iter][atom2])
            outputdist.write(" %11.6f"%(distance[index][iter]))
        outputdist.write("\n")
    return distance


def checkbondbreak(max_dist, gwpdim, nbiter, time, distance, ggp):
    prod_dens = np.zeros((nbiter))
    output = open("bond_break_pop.dat", "w")
    for iter in range(nbiter):
        for index in range(gwpdim):
            if (distance[index][iter] >= max_dist):
                prod_dens[iter] = prod_dens[iter] + ggp[index][iter]
        output.write("%7.3f" %(time[iter]) + " %11.6f" %(prod_dens[iter]) + "\n")
    output.close()
    return prod_dens


# this section is to plot the eight different compoents
def plotprod(time,result):
    plt.plot(time,result)
    plt.xlabel("time")
    plt.ylabel("product of density")
    plt.show()



if __name__ == "__main__":
    main(sys.argv[1:])

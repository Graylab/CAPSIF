#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 18:54:43 2022

@author: sudhanshu
CAPSIF:G PDB utility files

"""
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from Bio.PDB.Polypeptide import index_to_one
import pandas as pd
from sklearn.linear_model import LinearRegression


def aa_1_letter_code():
    seq=''
    for i in range(20):
        seq = seq+ index_to_one(i)
    return seq



class pdb_functions:
    def __init__(self):
        self.pdbfl = ""
        self.pdb_data = []
        self.xyz_data = []
        self.atm_array = []
        self.info = ''' some function to read and write pdbfiles and
        converstion to different formats.'''
        self.pdb_data_continuous = []
    def _pdb_file_(self,pdbf):
        self.pdbfl = pdbf

    def _pdb_splitter_(self):
        '''

        #
        #  1 -  6        Record name     "ATOM  "
        #  7 - 11        Integer         Atom serial number.
        # 13 - 16        Atom            Atom name.
        # 17             Character       Alternate location indicator.
        # 18 - 20        Residue name    Residue name.
        # 22             Character       Chain identifier.
        # 23 - 26        Integer         Residue sequence number.
        # 27             AChar           Code for insertion of residues.
        # 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
        # 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        # 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.

        '''
        fl=open(self.pdbfl,'r')
        data=fl.readlines()
        fl.close()
        out=[];
        for l in data:
            if len(l)>5:
                if l[0:4]=='ATOM' or l[0:6]=='HETATM' :
                    srn = int(l[6:11].strip())
                    atmi = l[12:17].strip()

                    if atmi.count(" ") > 0:
                        if atmi.endswith("A"):
                            new_atom_i = atmi.split(" ")[0]
                            atmi = new_atom_i

                    resi = l[17:20].strip()
                    chain = l[21].strip()
                    resno = int(l[22:26].strip())
                    xi = float(l[30:38].strip())
                    yi = float(l[38:46].strip())
                    zi = float(l[46:54].strip())
                    atm_id = l[77:].strip()
                    out.append([srn,atmi,resi,chain,resno,xi,yi,zi,atm_id])

        return out

    def dummy_pdb_data_CB(self,size):
        data =[]
        for i in range(size):
            data.append([i+1,'CB','ALA','A',i+1,0,0,0,'C'])

        self.dummy_pdb_data = data



    def _pdb_continuous_residues_(self):
        """Removes chain change so residues number are different"""
        current_res = self.pdb_data[0][4]
        counter = 1
        pdb_data_continuous=[]
        for i in self.pdb_data:
            if not i[4] == current_res:
                counter +=1
                current_res=i[4]

            pdb_data_continuous.append(i[0:4]+[counter]+i[5:])

        self.pdb_data_continuous = pdb_data_continuous

    def renumber_atoms(self):
        for i, line in enumerate(self.pdb_data):
            self.pdb_data[i][0] = i+1


    def renumber_residues(self):
        prev_res = self.pdb_data[0][4]
        curr_res = 1

        for i in range(len(self.pdb_data)):
            if self.pdb_data[i][4] == prev_res:
                self.pdb_data[i][4] = curr_res
            else:
                prev_res = self.pdb_data[i][4]
                curr_res +=1
                self.pdb_data[i][4] = curr_res



    def read_pdb(self,pdbf):
        self.pdbfl = pdbf
        self.pdb_data = self._pdb_splitter_()
        self._pdb_continuous_residues_()
        self._xyz_data_()
        self._atm_array_()

    def _xyz_data_(self):
        xyz_d = []
        for i in self.pdb_data:
            xyz_d.append(i[5:8])
        self.xyz_data = xyz_d

    def _atm_array_(self):
        self.atm_array = [i[1] for i in self.pdb_data]



    def import_pdb_models(pdbfl):
        fidmp=open(pdbfl,'r')
        data_mp=fidmp.readlines()
        fidmp.close()
        coord_mp=[]
        tmp_frame=[]
        for i in data_mp:
            if i.find('ENDMDL')>-1:
                # print tmp_frame
                coord_mp.append(tmp_frame)
                tmp_frame=[]
            if len(i)>5:
                if i[0:4]=='ATOM':
                    # srn=int(strip(l[6:11]))
                    # atmi=strip(l[12:17])
                    # resi=strip(l[17:20])
                    # chain=strip(l[21])
                    # resno=int(strip(l[22:26]))
                    xi=float(i[30:38].strip())
                    yi=float(i[38:46].strip())
                    zi=float(i[46:54].strip())
                    tmp_frame.append([xi,yi,zi])
        return coord_mp



    def pdb2atmdata(self,atms,pdbfile):
        fl=open(pdbfile,'r')
        data=fl.readlines()
        fl.close()
        out=[];
        for line in data:
            spl_line=line.split()
            if len(spl_line)>6:
                atmind=2
            else:
                atmind=1
            if spl_line[0]== 'ATOM':
                if spl_line[atmind]==atms:
                    if len(spl_line)>6:
                        out.append([float(spl_line[5]),float(spl_line[6]),float(spl_line[7])])
                    else:
                        out.append([float(spl_line[4]),float(spl_line[5]),float(spl_line[6])])
        return np.array(out)


    def p_data_string(self,en):
        strn=('ATOM  '+repr(en[0]).rjust(5)+" "+en[1].ljust(4)+" "+en[2].ljust(3)+" "+repr(en[4]).rjust(5) +
              "    "+repr(en[5]).rjust(8)+repr(en[6]).rjust(8)+repr(en[7]).rjust(8)+"                     "+en[8].rjust(2)+"\n")
        return strn

    def pdb_write_from_xyz(self, xyz_cord, pdb_file_name):
        en = self.pdb_data
        counter = -1
        fid=open(pdb_file_name,'w+')
        for i in xyz_cord:
            counter=counter+1
            en_nm=en[counter][1]+" "
            if len(en_nm)>4:
                justi=5
                atmn_nm=" "+en_nm.ljust(justi)
            else:
                justi=4
                atmn_nm="  "+en_nm.ljust(justi)
            strn='ATOM  '+repr(en[counter][0]).rjust(5)+atmn_nm+""+en[counter][2].ljust(3)+en[counter][3].rjust(2)+repr(en[counter][4]).rjust(4)+"    "+('%.3f' % i[0]).rjust(8)+('%.3f' % i[1]).rjust(8)+('%.3f' % i[2]).rjust(8)+" "*22+en[counter][8].rjust(2)+"\n"
        # print strn
            fid.write(strn)
        fid.write("END\n")
        fid.close()

    def write_pdb (self,pname):
        xyz_cord = self.xyz_data
        self.pdb_write_from_xyz(xyz_cord,pname)

    def save_pdb(self,pname):
        self.write_pdb(pname)

    def dump_pdb(self,pname):
        self.write_pdb(pname)

    def pdb_write(self,pname):
        self.write_pdb(pname)

    def coord_of_atom_from_xyz_data(self, atom_name):
        counter = 0
        for i in self.pdb_data:
            if i[1]==atom_name:
                return self.xyz_data[counter]
            counter +=1
        return 0

    def index_number_of_atom(self, atom_name):
        counter = 0
        for i in self.pdb_data:
            if i[1]==atom_name:
                return counter
            counter +=1
        return -1

    def remove_line_x(self, line_num):
        print("line number starts from 0")
        if line_num == -1:
            return
        vac_pdb = []
        for i in range(len(self.pdb_data)):
            if i == line_num :
                continue
            vac_pdb.append(self.pdb_data[i])

        self.pdb_data = vac_pdb
        del vac_pdb
        self._xyz_data_()

    def refresh(self):
        self._xyz_data_()
        self._atm_array_()
        #renumbder_atom had to add

    def use_continuous_data(self):
        self.pdb_data = self.pdb_data_continuous

    def refresh_from_pdb_data(self):
        self.refresh()


    def refresh_from_xyz(self):
        for i in range(len(self.xyz_data)):
            self.pdb_data[i][5] = self.xyz_data[i][0]
            self.pdb_data[i][6] = self.xyz_data[i][1]
            self.pdb_data[i][7] = self.xyz_data[i][2]

        self._atm_array_()

    def rename_atom(self, old_name, new_name):
        for i in range(len(self.pdb_data)):
            if self.pdb_data[i][1] == old_name:
                self.pdb_data[i][1] = new_name
                return

    def rename_atom_type (self, index_id, new_type):
        self.pdb_data[index_id][8] = new_type

    def add_new_atom(self, xyz_cord, name , type_atom):
        self.xyz_data = np.append(self.xyz_data, [xyz_cord], axis=0)
        #print(self.xyz_data)
        len_pdb_data = len(self.pdb_data)
        prev_pdb_line = self.pdb_data[-1]
        self.pdb_data.append([ prev_pdb_line[0]+1, name , prev_pdb_line[2],
                              prev_pdb_line[3],  prev_pdb_line[4],
                              xyz_cord[0], xyz_cord[1], xyz_cord[2], type_atom] )

        self._atm_array_()



    def residue_data(self,residue_num):
        return_arr = []
        for i in self.pdb_data:
            if i[4] == residue_num:
                return_arr.append(i)
        return return_arr

    def coord_of_atom_of_residue(self, atom_name,residue_num):
        data = self.residue_data(residue_num)
        for i in data:
           # print(i)
            if i[1] == atom_name:
                return np.array(i[5:8])



def extract_CB(pdb_file, outfile=""):

    #pdb_file = "/home/sudhanshu/HDD2/projects2/utils/PDB_from_Morgan/MLN_pyranose_benchmark_set/only_protein/3OEB_protein.pdb"
    if outfile=="":
        outfile = pdb_file[:-4]+"_CB.pdb"

    pdb_f = pdb_functions()
    pdb_f.read_pdb(pdb_file)
    pdb_f.use_continuous_data()

    cb_pdb_data = []

    for i in range(50000):
        d = pdb_f.residue_data(i+1)
        CB = 'CB'
        if len(d) == 0:
            break

        #print (CB)
        for j in d:
            if j[2] == 'GLY':
                CB = 'CA'

            if j[1] == CB:
                cb_pdb_data.append(j)
                break


    pdb_f.pdb_data = cb_pdb_data
    pdb_f.refresh_from_pdb_data()
    pdb_f.write_pdb(outfile)





def rsquare_data(dx,dy): #d11 : experminetal data X data
    d11 = np.copy(dx)
    d22 = np.copy(dy)
    nan_vals = d11+d22
    d1 = d11[~np.isnan(nan_vals)]
    d2 = d22[~np.isnan(nan_vals)]
    d1 = d1.reshape((-1,1))
    model = LinearRegression()
    model.fit(d1, d2)
    r_sq = model.score(d1, d2)
    slope = model.coef_
    y_pred = model.predict(d1)

    d1_min_pos = np.where(d1==min(d1))[0][0]
    d1_max_pos = np.where(d1==max(d1))[0][0]
    # print(d1_max_pos)

    x_points = np.array([np.min(d1), np.max(d1)])
    y_points = np.array([y_pred[d1_min_pos], y_pred[d1_max_pos]])

    xy_out = [x_points, y_points]

    #print(r_sq)

    return r_sq,slope,xy_out,~np.isnan(nan_vals)



def aa_parameters_all():
    aa_seq = aa_1_letter_code()

    aa_parameters = {}


    in_file = "../data_preparation/data_files/aa1.csv"

    data =pd.read_csv(in_file,delimiter=",")

    correct_aa_seq = [np.where(data.aa1 == i)[0][0] for i in aa_seq ]

    data = pd.DataFrame(data, index= correct_aa_seq)
    data = data.reset_index(drop=True)

    # normalization in 0 to 1 range
    for i in data.columns[2:]:
        data[i] = (data[i] - min(data[i]))/max(data[i])


    # for i in data.columns
    aa_parameters['info'] = ['Hydropathy', 'radius', 'Aromaphilicity', 'Hbond_D', 'Hbond_A']

    for i in range(20):
        vector =[]
        for j in ["Hydropathy" ,"Volume(A3)" , "Aromaphilicity","H_bond_Doner","H_bond_Acceptor"]:
            val = data[j][i]
            if j == "Volume(A3)":
                val = val**(1/3)

            vector.append(val)

        aa_parameters[data.aa3[i]] = vector

    # some patches of non-20 amino acids
    aa_parameters['MSE'] = aa_parameters['MET']
    aa_parameters['GLX'] = aa_parameters['GLN']
    aa_parameters['CSO'] = aa_parameters['CYS']
    return aa_parameters

class scores:
    pass

def TM_score(pdb1,pdb2):
    tm_align_exe = "/home/sudhanshu/bin/my_scripts/TMalign"
    data = os.popen(tm_align_exe + " "+pdb1 + " "+pdb2).read()
    data = data.splitlines()
    out = scores()
    counter = 0
    for i in data:
        if i.startswith('Aligned'):
            isplit = i.split()
            out.rmsd= float(isplit[4][:-1])

        elif i.startswith('TM-score'):
            counter += 1
            isplit = i.split()
            if counter == 1:
                out.TM_score1 = float(isplit[1])
            elif counter == 2:
                out.TM_score2 = float(isplit[1])


        if counter == 2:
            break

    return out




# for runnig a cleaning code on multiple threads
class done_data_recorder:
    def __init__(self, file_name, data_type):
        self.file_name = file_name
        self.data_type = data_type
        if not os.path.exists(self.file_name):
            self.__init_run__()

    def __init_run__(self):
        fid = open(self.file_name,"w+")
        fid.write("INIT_SSS ")
        fid.close()

    def add_val(self, val):
        if self.data_type == int:
            add_data = "'%d'" % val

        else:
            add_data = "'"+val+"'"

        self.lock_the_file(self.file_name)
        fid = open(self.file_name,"a")
        fid.write(add_data)
        fid.close()
        self.unlock_the_file(self.file_name)

    def check_val_exist(self, val):
        if self.data_type == int:
            add_data = "'%d'" % val

        else:
            add_data = "'"+val+"'"

        fid = open(self.file_name,"r")
        data = fid.readlines()[0]

        if data.count(add_data)  > 0:
            return True

        return False

    def lock_the_file(self, filenm):
        lock_file = filenm+".lock"
        while 1:
            if os.path.exists(lock_file):
                time.sleep(np.random.rand(1)[0]*1)
                print("waiting to lock..")
            else:
                print("locking the file")
                fid_pdb_done = open(lock_file,"w+")
                fid_pdb_done.close()
                break;

    def unlock_the_file(self, filenm):
        print("unlocking the file")
        lock_file = filenm+".lock"
        os.remove(lock_file)

    def reset(self):
        self.__init_run__()




# def ss_randname( sz ):
# def rosetta_score_file_read(scorefile):
# def ss_char( n_string):
# def ss_exist( nm,typ):
# def ss_print( data, yon):
# def ss_importdata( *inputs):
# def ss_isnumber( val):
# def ss_filetype( filenm):
# def ss_list_all_files_with_ext(dir_path, extn):


# def extract_pdb_data( all_data):
# def mode8point3(val):
# def extract_xyz_data( all_data,types):


# def ss_distance(F,G):
#class CHI_energy_from_gaussian_file():


#def normal_vector_of_the_plane(xx,yy,zz):
#def angle_to_plane(xyz_atom, acceptor_C_cord, xyz_ring):

#class geometry_adjustment:

#class link_id_handler():

# MIT License

# Copyright (c) 2019 Massachusetts Institute of Technology 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The following is adapted from a polychrom loop extrusion example
# see https://github.com/open2c/polychrom/tree/master/examples/loopExtrusion

# This version of the code facilitates on-the-fly adjustment of numbers of loop extruders in the sim

#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np
cimport numpy as np

import cython
cimport cython


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class LEFTranslocatorDirectional(object):
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight

    cdef cython.double [:] stallFalloff
    cdef cython.double [:] stallFalloff_typeII
    cdef cython.double [:] stallFalloff_typeIII
    cdef cython.double [:] falloff
    cdef cython.double [:] falloff_typeII
    cdef cython.double [:] falloff_typeIII
    cdef cython.double [:] pause
    cdef cython.double [:] pause_typeII
    cdef cython.double [:] pause_typeIII

    cdef cython.double [:] cumEmission
    cdef cython.double [:] permeability
    cdef cython.long [:] LEFs1
    cdef cython.long [:] LEFs2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:] occupied 
    cdef cython.long [:] LEFtype
    cdef cython.long [:] active

    cdef int use_lef_permeability
    cdef int permeable
    
    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProb, stallFalloffProb, permeability, numLEF,
                 deathProbII=0, stallFalloffProbII=0, pauseProbII=0, 
                 deathProbIII=0, stallFalloffProbIII=0, pauseProbIII=0,
                 LEFtype=0,
                 activeStatus=1
                ):
        emissionProb[0] = 0
        emissionProb[len(emissionProb)-1] = 0
        emissionProb[stallProbLeft > 0.9] = 0        
        emissionProb[stallProbRight > 0.9] = 0        
        
        self.N = len(emissionProb)
        self.M = numLEF
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.stallLeft[0] = 1 # stall LEFs at two chain ends
        self.stallRight[self.N-1] = 1 

        self.falloff = deathProb
        if type(deathProbII) in [int, float, np.float64, np.double]:
            self.falloff_typeII = deathProbII*np.ones(self.N, dtype=np.double)
        else:
            self.falloff_typeII = deathProbII
        if type(deathProbIII) in [int, float, np.float64, np.double]:
            self.falloff_typeIII = deathProbIII*np.ones(self.N, dtype=np.double)
        else:
            self.falloff_typeIII = deathProbIII

        self.pause = pauseProb
        if type(pauseProbII) in [int, float, np.float64, np.double]:
            self.pause_typeII = pauseProbII*np.ones(self.N, dtype=np.double)
        else:
            self.pause_typeII = pauseProbII
        if type(pauseProbIII) in [int, float, np.float64, np.double]:
            self.pause_typeIII = pauseProbIII*np.ones(self.N, dtype=np.double)
        else:
            self.pause_typeIII = pauseProbIII

        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.LEFs1 = np.zeros((self.M), int)
        self.LEFs2 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)

        self.permeability = permeability 
        if (numLEF>0) and (np.max(permeability)>0):
            self.permeable = 1
        else:
            self.permeable = 0

        if type(LEFtype) in [int, float, np.float64, np.double]:
            self.LEFtype = np.zeros(self.N, dtype=int) + int(LEFtype)
        else:
            self.LEFtype = LEFtype
        if type(activeStatus) in [int, float, np.float64, np.double]:
            self.active = np.zeros(self.N, dtype=int) + int(activeStatus)
        else:
            self.active = activeStatus

        self.use_lef_permeability=1 # wrote this in for flexibility in code later
        self.occupied = np.zeros(self.N, int)

        self.stallFalloff = stallFalloffProb
        if type(stallFalloffProbII) in [int, float, np.float64, np.double]:
            self.stallFalloff_typeII = stallFalloffProbII*np.ones(self.N, np.double)
        else:
            self.stallFalloff_typeII = stallFalloffProbII
        if type(stallFalloffProbIII) in [int, float, np.float64, np.double]:
            self.stallFalloff_typeIII = stallFalloffProbIII*np.ones(self.N, np.double)
        else:
            self.stallFalloff_typeIII = stallFalloffProbIII

        self.occupied[0] = 1
        self.occupied[self.N - 1] = 1
        self.maxss = 1000000
        self.curss = 99999999

        for ind in xrange(self.M):
            if self.active[ind] == 1:
                self.birth(ind)
            else:
                self.LEFs1[ind]=-1
                self.LEFs2[ind]=-1


    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos >= self.N - 1:
                print("bad value", pos, self.cumEmission[len(self.cumEmission)-1])
                continue 
            if pos <= 0:
                print("bad value", pos, self.cumEmission[0])
                continue 
 
            
            if self.occupied[pos] == 1:
                continue
            if self.occupied[pos+1] == 1:
                continue

            self.LEFs1[ind] = pos
            self.LEFs2[ind] = pos+1
            self.occupied[pos] = 1
            self.occupied[pos+1] = 1

            return

    cdef death(self):
        cdef int i,jj
        cdef double falloff1, falloff2 
        cdef double falloff 
        cdef int multi_occ1, multi_occ2 # variables to indicate whether site of interest is occupied by multiple LEFs
        multi_occ1=0 # set to 0 so that in case self.permeable==False, we can still flip the site occupied flag off
        multi_occ2=0

        for i in xrange(self.M):
            if not self.active[i]:
                continue

            if self.stalled1[i] == 0:
                if self.LEFtype[i]==0:
                    falloff1 = self.falloff[self.LEFs1[i]]
                elif self.LEFtype[i]==1:
                    falloff1 = self.falloff_typeII[self.LEFs1[i]]
                else:
                    falloff1 = self.falloff_typeIII[self.LEFs1[i]]
            else: 
                if self.LEFtype[i]==0:
                    falloff1 = self.stallFalloff[self.LEFs1[i]]
                elif self.LEFtype[i]==1:
                    falloff1 = self.stallFalloff_typeII[self.LEFs1[i]]
                else:
                    falloff1 = self.stallFalloff_typeIII[self.LEFs1[i]]
            if self.stalled2[i] == 0:
                if self.LEFtype[i]==0:
                    falloff2 = self.falloff[self.LEFs2[i]]
                elif self.LEFtype[i]==1:
                    falloff2 = self.falloff_typeII[self.LEFs2[i]]
                else:
                    falloff2 = self.falloff_typeIII[self.LEFs2[i]]
            else:
                if self.LEFtype[i]==0:
                    falloff2 = self.stallFalloff[self.LEFs2[i]]
                elif self.LEFtype[i]==1:
                    falloff2 = self.stallFalloff_typeII[self.LEFs2[i]]
                else:
                    falloff2 = self.stallFalloff_typeIII[self.LEFs2[i]]

            falloff = max(falloff1, falloff2)
            if randnum() < falloff:
                if self.permeable:
                    multi_occ1 = 0
                    multi_occ2 = 0
                    for jj in range(self.M):
                        if (self.LEFs1[jj] == self.LEFs1[i]) and not (jj == i):
                            multi_occ1 = 1
                        elif (self.LEFs2[jj] == self.LEFs1[i]):
                            multi_occ1 = 1
                        if (self.LEFs2[jj] == self.LEFs2[i]) and not (jj==i):
                            multi_occ2 = 1
                        elif (self.LEFs1[jj] == self.LEFs2[i]):
                            multi_occ2 = 1

                if not multi_occ1:
                    self.occupied[self.LEFs1[i]] = 0
                if not multi_occ2:
                    self.occupied[self.LEFs2[i]] = 0
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.int_)
            self.ssarray = foundArray
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
        

    cdef step(self):
        cdef int i,jj
        cdef double pause
        cdef double stall1, stall2
        cdef double r
        cdef int cur1
        cdef int cur2 
        cdef int multi_occ = 0

        for i in range(self.M):
            if not self.active[i]:
                continue
            
            if self.LEFtype[i]==2:#only cohesin stalls at CTCF
                stall1 = self.stallLeft[self.LEFs1[i]]
                stall2 = self.stallRight[self.LEFs2[i]]
                                    
                if randnum() < stall1: 
                    self.stalled1[i] = 1
                if randnum() < stall2: 
                    self.stalled2[i] = 1
            else: # cohesin stalled at ends by stallprob=1
                if self.LEFs1[i]==0: 
                    self.stalled1[i] = 1
                if self.LEFs2[i]==self.N-1:
                    self.stalled2[i] = 1
                         
            cur1 = self.LEFs1[i]
            cur2 = self.LEFs2[i]
            
            if self.stalled1[i] == 0: 
                if self.occupied[cur1-1] == 0:
                    if self.LEFtype[i] == 0:
                        pause1 = self.pause[self.LEFs1[i]]
                    elif self.LEFtype[i] == 1:
                        pause1 = self.pause_typeII[self.LEFs1[i]]
                    else:
                        pause1 = self.pause_typeIII[self.LEFs1[i]]
                    if randnum() > pause1: 
                        self.occupied[cur1 - 1] = 1
                        self.occupied[cur1] = 0
                        self.LEFs1[i] = cur1 - 1
                elif (self.permeable==1):# and (self.occupied[cur1-1] == 1): #by virtue of the else, cur-1 is occupied. 
                    if self.LEFtype[i] == 0:
                        pause1 = self.pause[self.LEFs1[i]]
                    elif self.LEFtype[i] == 1:
                        pause1 = self.pause_typeII[self.LEFs1[i]]
                    else:
                        pause1 = self.pause_typeIII[self.LEFs1[i]]
                    if randnum() > pause1:
                        r=randnum()
                        if r < self.permeability[i]: # look at LEF i's permeability to determine its bypassing ability (i.e., 'permeability' param is more like 'bypassing capability')
                            #if not self.use_lef_permeability:
                            self.LEFs1[i] = cur1 - 1
                            self.occupied[cur1-1] = 1
                            #only implement use_lef_permeability; alternative is to use permeability of LEF(s) occupying the target site. would need to add a check here.       
                            multi_occ=0 
                            for jj in range(self.M):
                                if (self.LEFs1[jj]==cur1) and not (jj==i):
                                    multi_occ=1
                                    break
                                if (self.LEFs2[jj]==cur1):
                                    multi_occ=1
                                    break
                            if not multi_occ:
                                self.occupied[cur1] = 0

            if self.stalled2[i] == 0:                
                if self.occupied[cur2 + 1] == 0:                    
                    if self.LEFtype[i] == 0:
                        pause2 = self.pause[self.LEFs2[i]]
                    elif self.LEFtype[i] == 1:
                        pause2 = self.pause_typeII[self.LEFs2[i]]
                    else:
                        pause2 = self.pause_typeIII[self.LEFs2[i]]
                    if randnum() > pause2: 
                        self.occupied[cur2 + 1] = 1
                        self.occupied[cur2] = 0
                        self.LEFs2[i] = cur2 + 1
                elif (self.permeable == 1):# and (self.occupied[cur2+1]==1):
                    if self.LEFtype[i] == 0:
                        pause2 = self.pause[self.LEFs2[i]]
                    elif self.LEFtype[i] == 1:
                        pause2 = self.pause_typeII[self.LEFs2[i]]
                    else:
                        pause2 = self.pause_typeIII[self.LEFs2[i]]
                    if randnum() > pause2:
                        r=randnum()
                        if r < self.permeability[i]:
                            self.LEFs2[i] = cur2 + 1
                            self.occupied[cur2+1] = 1
                            multi_occ=0
                            for jj in range(self.M):
                                if (self.LEFs2[jj]==cur2) and not (jj==i):
                                    multi_occ=1
                                    break
                                if self.LEFs1[jj]==cur2:
                                    multi_occ=1
                                    break
                            if not multi_occ: 
                                self.occupied[cur2] = 0

    def removal(self, cython.long LEF_id):
        #WARNING: need to add a check in the lines below if I allow LEF bypassing
        self.occupied[self.LEFs1[LEF_id]] = 0
        self.occupied[self.LEFs2[LEF_id]] = 0
        self.LEFs1[LEF_id] = -1
        self.LEFs2[LEF_id] = -1

    def activate(self, cython.long LEF_id):
        self.active[LEF_id] = 1
        self.birth(LEF_id)

    def deactivate(self, cython.long LEF_id):
        self.active[LEF_id] = 0
        self.removal(LEF_id)


    def steps(self,N,activateLEFs=[],deactivateLEFs=[]):
        cdef int i,j

        for i in xrange(N):
            for j in range(len(activateLEFs)):
                self.activate(activateLEFs[j])
            for j in range(len(deactivateLEFs)):
                self.deactivate(deactivateLEFs[j])
            self.death()
            self.step()

    def change_falloff(self, deathprob):
        #deathprob should be a float/double
        self.falloff=deathprob*np.ones(self.N, dtype=np.double)

    def getOccupied(self):
        return np.array(self.occupied)
    
    def getLEFs(self):
        return np.array(self.LEFs1), np.array(self.LEFs2)
        
        
    def updateMap(self, cmap):
        cmap[self.LEFs1, self.LEFs2] += 1
        cmap[self.LEFs2, self.LEFs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.LEFs1] = 1
        pos[ind, self.LEFs2] = 1



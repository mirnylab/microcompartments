# Simulations for compartments and extrusion using polychrom

import os, sys
import numpy as np
import time, pickle

import openmm

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter
from polychrom.starting_conformations import grow_cubic, create_random_walk

from smcBondUpdater import smcBondUpdater
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
from LEF_Dynamics_variable import LEFTranslocatorDirectional

import simUtils, tools


#initialize paramsDict filled with params we can feed in via commandline
paramsDict={
            "gpu":str(0),
            "dt":40,
            "thermostat":0.01,
            "numpolyblocks":29400, # 245 min, total sim duration after prometaphase is equilibrated
            "saveevery":2, # save once every second
            "initpolyblocks":36000, #prometaphase equilibration time, 5x condensin II lifetime
            # Boundary Conditions
            "pbc":False,
            "kbc":1,
            "aspect":4, # initial cylinder aspect ratio
            "cylinderFinalSizeFactor":0.5, # cylinder height to be reduced by 1/2
            "cylinderShortenStart": 3000, #25 min, time at which cylinder shortening to commences
            "cylinderShortenEnd": 3600,# and ends
            "pinEnds":False,
            "BC_change_start": 3600, #time to start crossover from cyl to sph
            "BC_change_end": 4200, #time to end crossover from cyl to sph (leaving only sph)
            "t_sphere_inflate_start":999999999, # in a case where we only use spherical BC, use these to change density mid-sim
            "t_sphere_inflate_end":999999999,
            ###
            "outpath":"local_data",
            "npoly":61600, # 500 bp per mono, 1.925 Mb region
            "nchr":1,
            "density":0.6, #initial prometaphase density
            "densityInter":0.3, # density in interphase
            "densityInflate":0.3, # density to target if we only use spherical BC
            "repel":3.0,
             #compartment params
            "epsA":0.,
            "epsB":0.15,
            "epsC":1.5,
            "epsAB":0.,
            "epsAC":0.,
            "epsBC":0.,
            "epsAll":0.,
            "alen":500,
            "blen":500,
            "clen":0,
            "cspace":500,
            "t_comps_on":3600,
            "t_poor_solvent_off":999999999,# after this time, zero out epsAll
            #extrusion params
            "lifeC1":360,# 3 min with 0.5 s per extrusion step
            "sepC1_start":70,
            "sepC1_max":53.8,#30% increase
            "vlefC1":0.5, # -->1 kb/s total speed of loop growth accounting for both sides
            "permC1":0.,
            "lifeC2":7200, # 1 hr
            "sepC2":280,
            "vlefC2":0.5, # 1 kb/s total
            "permC2":0.,
            "lifeCoh":1200, # 10 min with 0.5 s per extrusion step
            "sepCoh":200,
            "vlefCoh": 0.5, # 1 kb/s total
            "permCoh":0.,
            "stall":0.5, #stall is site-based. so we'll make only cohesin pay attention to stalls
            "stallall":False,
            "t_C1_incStart": 2040, #default is 17 min after steady state prometa; "typical" in ms is 1800
            "t_C1_incEnd": 2400, #20 min after steady state prometa; typical is 2100
            "t_C1_decStart": 2400, #20 min after steady state prometa; typical is 2100
            "t_C1_decEnd": 3600, #30 min; typical is 2400 (20min)
            "t_C2_decStart": 3600, #30 min
            "t_C2_decEnd": 3600, #30 min, CII unload simultaneously by default
            "t_Coh_incStart": 3600, #30 min
            "t_Coh_incEnd": 29400, #245 min
            "t_C1_changeLife":2400,
            "C1_lifeFactor":1, # factor change in C1 life at the time above
            ####
            "polysteps":540, # each extrusion step will be 0.5 s 
            "ignore":True,
            "integrator":"langevin",
            "comppath":"",
            "ctcfpath":"",
            "microcomppath":"",
            "flag":""
           }
#info about select variables
helpDict={
          "initskip":"num polymer equilibraiton blocks - default 80",
          "polysteps":"num polymer timesteps between LEF timesteps - default 255",
          "ignore":"whether to ignore adjacent particles for nonbond particle potentials - default True",
          "flag":"label to add to end of output folder - default ''"
         }

#get command line values and options
paramsInput= tools.argsList(pdict=paramsDict.keys(), hdict=helpDict)
for p in paramsInput.arg_dict:
    print(p, paramsInput.arg_dict[p])

for pname in paramsDict.keys():
    if pname in paramsInput.arg_dict:
        paramsDict[pname] = paramsInput.arg_dict[pname]


####basic sim parameters
gpu=str(paramsDict['gpu'])
dt=int(paramsDict['dt'])
THERMOSTAT=float(paramsDict['thermostat'])
INTEGRATOR=str(paramsDict['integrator'])

prev_step=0

polyBlockSteps=int(paramsDict['polysteps'])

smcStepsPerBlock=1
saveEveryBlocks = int(paramsDict['saveevery'])#100  # save every 100 blocks 
skipSavedBlocksBeginning = int(paramsDict['initpolyblocks'])  # how many blocks (saved) to skip after you restart LEF positions
totalSavedBlocks = int(paramsDict['numpolyblocks'])# how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartUpdaterEveryBlocks = 60 # once per 30 s... allows smoother transitions of potentials (and SMC params) when needed.


####parameters for polymer
LENGTH=int(paramsDict['npoly'])
numChr=int(paramsDict['nchr'])
chrSizes=[LENGTH//numChr]*numChr
density=float(paramsDict['density'])
densityInter=float(paramsDict['densityInter'])
densityInflate=float(paramsDict['densityInflate'])
PBC=int(paramsDict['pbc'])
kConfine=float(paramsDict['kbc'])
PIN_ENDS=int(paramsDict['pinEnds'])
Rsph= 0.5*(LENGTH/densityInter)**(1./3.) # assuming radius of a monomer = 0.5 (polynomial repulsive force has range of 1 = 2r)
AR=float(paramsDict['aspect']) # cyl aspect ratio
cylinderFinalSizeFactor=float(paramsDict['cylinderFinalSizeFactor'])
Hcyl=((2./3.)*AR*AR*LENGTH/density)**(1./3.) # initial cylinder height for desired density, with Rcyl defined as below
Rcyl= Hcyl / (2.*AR)
PBCbox_len= 0.5*(4./3.*np.pi*LENGTH/density)**(1./3.)
pbc_param = [PBCbox_len]*3 if PBC else False  # feed this into simulation initialization
REPEL=float(paramsDict['repel']) #soft repulsion -- gives some excl vol, but chain can still pass itself.

polyBondWiggleDist= 0.1 

ignoreAdjacent=int(paramsDict['ignore']) # tells simulation whether or not to ignore non-bonded potentials between monomers adjacent in chain

# parameters for smc bonds
smcBondWiggleDist = 0.1
smcBondDist = 0.5

#compartment interactions
EPSILON_ALL=float(paramsDict['epsAll'])
EPSILON_A=float(paramsDict['epsA'])
EPSILON_B=float(paramsDict['epsB'])
EPSILON_C=float(paramsDict['epsC'])
EPSILON_AB=float(paramsDict['epsAB'])
EPSILON_AC=float(paramsDict['epsAC'])
EPSILON_BC=float(paramsDict['epsBC'])

B_length=int(paramsDict['blen'])
A_length=int(paramsDict['alen'])
C_length=int(paramsDict['clen'])
C_spacing=int(paramsDict['cspace'])
comp_list_str=str(paramsDict['comppath'])
microcomp_list_str=str(paramsDict['microcomppath'])

if not LENGTH % (B_length+A_length) == 0:
    print("Warning: total poly length not divisible by (len B + len A)")
#periodic list of blocks
B_list=[ii*(A_length+B_length) + jj for ii in range(int(LENGTH // (A_length+B_length))+1) for jj in range(B_length)
        if ii*(A_length+B_length) + jj < LENGTH]
if len(comp_list_str)>0:    
    #####
    #FILE FORMAT: for each line is start end for a B compartment, except last line, which dictates how many A monomers follow
    #####
    with open(comp_list_str,"r") as myfile:
        lines=myfile.readlines()
    basePattern=[]
    for comp in lines[:-1]:
        cc= comp.rstrip().split() # read in B compartment start and ends
        basePattern=basePattern+list(np.arange(int(cc[0]),int(cc[1])))
    lastA=int(lines[-1].rstrip())
    B_list=list(basePattern)
    while max(B_list) < LENGTH - 1:
        B_list = B_list + list(basePattern+max(B_list)+1+lastA)
    while B_list[-1] >= LENGTH:
        B_list.pop()


if (EPSILON_C>0.) and (C_length>0):
    C_list=[C_spacing*ii + jj for ii in range(int(LENGTH // C_spacing)+1) for jj in range(C_length)
            if C_spacing*ii + jj < LENGTH]
else:
    C_list=[]
if len(microcomp_list_str)>0:
    #####
    #FILE FORMAT: for each line is start end for a B compartment, except last line, which dictates how many A monomers follow
    #####
    basePattern=[]
    with open(microcomp_list_str,"r") as myfile:
        lines=myfile.readlines()
    for comp in lines[:-1]:
        comp= comp.rstrip().split() # read in micro-compartment start and ends
        basePattern=basePattern+list(np.arange(int(comp[0]),int(comp[1])))
    lastA=int(lines[-1].rstrip())
    C_list=list(basePattern)
    while max(C_list) < LENGTH - 1:
        C_list = C_list + list(basePattern+max(C_list)+1+lastA)
    while C_list[-1] >= LENGTH:
        C_list.pop()


####extrusion params
LIFETIME_C1=float(paramsDict['lifeC1'])
LIFETIME_C2=float(paramsDict['lifeC2'])
LIFETIME_Coh=float(paramsDict['lifeCoh'])
FACTOR_CHANGE_C1=float(paramsDict['C1_lifeFactor'])
SEPARATION_C1_start=float(paramsDict['sepC1_start'])
SEPARATION_C1_max=float(paramsDict['sepC1_max'])
SEPARATION_C2=float(paramsDict['sepC2'])
SEPARATION_Coh=float(paramsDict['sepCoh'])
EXTR_SPEED_C1=float(paramsDict['vlefC1'])
EXTR_SPEED_C2=float(paramsDict['vlefC2'])
EXTR_SPEED_Coh=float(paramsDict['vlefCoh'])
N_C1_start = int(LENGTH // SEPARATION_C1_start)
N_C1_max = int(LENGTH // SEPARATION_C1_max)
N_C2 = int(LENGTH // SEPARATION_C2)
N_Coh = int(LENGTH // SEPARATION_Coh)
Nlefs = N_C1_max + N_C2 + N_Coh
PERMEABILITY_C1 = float(paramsDict['permC1'])
PERMEABILITY_C2 = float(paramsDict['permC2'])
PERMEABILITY_Coh = float(paramsDict['permCoh'])
STALL_RATE= float(paramsDict['stall'])
STALL_ALL= int(paramsDict['stallall'])
stall_path_str=str(paramsDict['ctcfpath'])


######times for transitions####
cylinderShortenStart=int(float(paramsDict['cylinderShortenStart']))
cylinderShortenEnd=int(float(paramsDict['cylinderShortenEnd']))
BC_change_start=int(float(paramsDict['BC_change_start']))
BC_change_end=int(float(paramsDict['BC_change_end']))
t_comps_on=int(float(paramsDict['t_comps_on']))
t_poor_solvent_off=int(float(paramsDict['t_poor_solvent_off']))
t_C1_incStart=int(float(paramsDict['t_C1_incStart']))
t_C1_incEnd=int(float(paramsDict['t_C1_incEnd']))
t_C1_decStart=int(float(paramsDict['t_C1_decStart']))
t_C1_decEnd=int(float(paramsDict['t_C1_decEnd']))
t_C2_decStart=int(float(paramsDict['t_C2_decStart']))
t_C2_decEnd=int(float(paramsDict['t_C2_decEnd']))
t_Coh_incStart=int(float(paramsDict['t_Coh_incStart']))
t_Coh_incEnd=int(float(paramsDict['t_Coh_incEnd']))
t_C1_changeLife=int(float(paramsDict['t_C1_changeLife']))
t_sphere_inflate_start=int(float(paramsDict['t_sphere_inflate_start']))
t_sphere_inflate_end=int(float(paramsDict['t_sphere_inflate_end']))

#assertions for condensin times
assert t_C1_incEnd % restartUpdaterEveryBlocks == 0
assert t_C1_decEnd % restartUpdaterEveryBlocks == 0
assert t_C2_decEnd % restartUpdaterEveryBlocks == 0
assert t_Coh_incEnd % restartUpdaterEveryBlocks == 0
assert t_C1_incEnd <= t_C1_decStart
assert t_C1_decStart <= t_C1_decEnd
assert t_C2_decStart <= t_C2_decEnd

#assertions for BC and density changes
assert BC_change_start <= BC_change_end
assert cylinderShortenStart <= cylinderShortenEnd
# cylinderShortenStart can be earlier than BC_change_start. shortening can happen as long as there is still a cylinder present! 
# note that cylinderShortenEnd should be <= BC_change_end, but this is not controlled by assertion since when BC_change_start occurs long before prometaphase data saving, we don't care about transition from cyl -> sph.
assert t_sphere_inflate_start <= t_sphere_inflate_end
assert t_sphere_inflate_start >= BC_change_end # sphere inflation is designed to transition from one spherical density to another & sim isn't sphere until after BC_change end


######times to take data#####
END_PROMETA=1200
BEGIN_ANATELO=2400
END_ANATELO=3600
BEGIN_EARLYG1=6600
END_EARLYG1=7800
BEGIN_MIDG1=13800
END_MIDG1=15000
BEGIN_LATEG1=28200
END_LATEG1=29400



#### where the data goes
folder_ind=1
folder = paramsDict['outpath']+"/traj0001"
for pname in paramsDict:
    if pname not in ['gpu','outpath','initpolyblocks','numpolyblocks','saveevery','kbc','pbc','ignore','pinEnds',
                     'dt', 'thermostat','nchr','repel','aspect',
                     'alen','blen',
                     'epsAll','epsC','epsAB', 'epsAC', 'epsBC','clen','cspace','epsA',
                     'stall','stallall',
                     'C1_lifeFactor',
                     "comppath","ctcfpath","microcomppath",
                     'integrator',
                     'flag']:
        if ('t_C1' not in pname) and ('t_Coh' not in pname) and ('BC_change' not in pname) and ('cylinder' not in pname) and ('t_C2' not in pname) and ('t_comps_on' not in pname) and ('nflate' not in pname) and ('t_poor_solvent_off' not in pname):
            if 'perm' not in pname:
                if 'C1' in pname:
                    folder = folder+"_"+pname.replace('C1','CI')+str(paramsDict[pname])
                elif 'C2' in pname:
                    folder = folder+"_"+pname.replace('C2','CII')+str(paramsDict[pname])
                else:
                    folder = folder+"_"+pname+str(paramsDict[pname])
if not THERMOSTAT==0.01:
    folder=folder+"_th"+str(THERMOSTAT)
if not dt==40:
    folder=folder+"_dt"+str(dt)
if not numChr==1:
    folder=folder+"_nchr"+str(nchr)
if not REPEL==3.:
    folder=folder+"_repel"+str(REPEL)
if not AR==4.:#5.:
    folder=folder+"_ar"+str(AR)
if EPSILON_A>0.:
    folder=folder+"_epsA"+str(EPSILON_A)
if EPSILON_ALL>0.:
    folder = folder+"_epsAll"+str(EPSILON_ALL)
if (EPSILON_C>0.) or (C_length>0):
    if len(microcomp_list_str)>0:
        folder=folder+"_epsC"+str(EPSILON_C)+"_epsBC"+str(EPSILON_BC)
    else:
        print("hi",microcomp_list_str)
        folder = folder+"_clen"+str(C_length)+"_cspace"+str(C_spacing)+"_epsC"+str(EPSILON_C)+"_epsBC"+str(EPSILON_BC)
if EPSILON_AB>0.:
    folder = folder+"_epsAB"+str(EPSILON_AB)
if EPSILON_AC>0.:
    folder = folder+"_epsAC"+str(EPSILON_AC)
if STALL_RATE>0.:
    if STALL_ALL:
        folder = folder+"_stAll"+str(STALL_RATE)
    else:
        folder = folder+"_stSites"+str(STALL_RATE)
if not (FACTOR_CHANGE_C1 == 1.):
    folder=folder+"_lifeC1fac"+str(FACTOR_CHANGE_C1)
if PBC:
    folder = folder+"_"+"PBC"
if PIN_ENDS:
    folder = folder+"_"+"PIN"
if not INTEGRATOR=="langevin":
    folder = folder+"_"+str(paramsDict["integrator"])
if len(paramsDict["flag"])>0:
    folder = folder+"_"+paramsDict["flag"]

while os.path.exists(folder):
    folder=folder.replace("traj{:04}".format(folder_ind), "traj{:04}".format(folder_ind+1))
    folder_ind+=1 

print(folder)
reporter = HDF5Reporter(folder=folder,
                        max_data_length=100, overwrite=True)
pickle.dump(paramsDict,open(folder+"/paramsDict.pkl","wb"))



# assertions for easy managing code below
assert restartUpdaterEveryBlocks % saveEveryBlocks == 0
assert skipSavedBlocksBeginning % restartUpdaterEveryBlocks == 0
assert totalSavedBlocks % restartUpdaterEveryBlocks == 0

savesPerUpdater = restartUpdaterEveryBlocks // saveEveryBlocks
updaterInitsSkip = skipSavedBlocksBeginning // restartUpdaterEveryBlocks
updaterInitsTotal = (totalSavedBlocks + skipSavedBlocksBeginning) // restartUpdaterEveryBlocks
print("Bond updater will be initialized {0} times, first {1} will be skipped".format(updaterInitsTotal, updaterInitsSkip))


def initModel():
    # this just inits the simulation model. Put your previous init code here
    birthArray = np.zeros(LENGTH, dtype=np.double) + 0.1
    deathArray_C1 = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_C1
    deathArray_C2 = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_C2
    deathArray_Coh = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_Coh
    stallDeathArray_C1 = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_C1
    stallDeathArray_C2 = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_C2
    stallDeathArray_Coh = np.zeros(LENGTH, dtype=np.double) + 1.0 / LIFETIME_Coh
    pauseArray_C1 = np.ones(LENGTH, dtype=np.double) * (1.-EXTR_SPEED_C1)
    pauseArray_C2 = np.ones(LENGTH, dtype=np.double) * (1.-EXTR_SPEED_C2)
    pauseArray_Coh = np.ones(LENGTH, dtype=np.double) * (1.-EXTR_SPEED_Coh)

    stallLeftArray = np.zeros(LENGTH, dtype=np.double)
    stallRightArray = np.zeros(LENGTH, dtype=np.double)

    if len(stall_path_str)>0:
        with open(stall_path_str,"r") as myfile:
            lines=myfile.readlines()

        plus=True
        basePatternLeft=[]
        basePatternRight=[]
        for line in lines[:-1]:
            if line.rstrip()=="-":
                plus=False
                continue
            if plus:
                basePatternLeft.append(int(line.rstrip().split()[0]))
            else:
                basePatternRight.append(int(line.rstrip().split()[0]))
        repeat_interval=int(lines[-1].rstrip())
        stallLeftList=list(basePatternLeft)
        repeats=1
        while max(stallLeftList)<LENGTH:
            stallLeftList=stallLeftList + list(np.array(basePatternLeft)+repeat_interval*repeats)
            repeats+=1
        while stallLeftList[-1]>=LENGTH:
            stallLeftList.pop()
        stallRightList=list(basePatternRight)
        repeats=1
        while max(stallRightList)<LENGTH:
            stallRightList=stallRightList + list(np.array(basePatternRight)+repeat_interval*repeats)
            repeats+=1
        while stallRightList[-1]>=LENGTH:
            stallRightList.pop()
        for ii in stallLeftList:
            stallLeftArray[ii] = STALL_RATE
        for ii in stallRightList:
            stallRightArray[ii] = STALL_RATE
    else:
        if not STALL_ALL:
            stallList = []# put locations of CTCFs here
        else:
            stallList = np.arange(LENGTH)
        for i in stallList:
            #put in correct stall prob
            stallLeftArray[i] = STALL_RATE
            stallRightArray[i] = STALL_RATE
    
    LEFtypeArray= np.array([0]*N_C1_max + [1]*N_C2 + [2]*N_Coh,dtype=int)
    activeArray= np.array([1]*N_C1_start + [0]*(N_C1_max-N_C1_start) + [1]*N_C2 + [0]*N_Coh,dtype=int)

    permArray = np.array([PERMEABILITY_C1]*N_C1_max + [PERMEABILITY_C2]*N_C2 + [PERMEABILITY_Coh]*N_Coh,dtype=np.double)

    SMCTran = LEFTranslocatorDirectional(
        birthArray,
        deathArray_C1,
        stallLeftArray,
        stallRightArray,
        pauseArray_C1,
        stallDeathArray_C1,
        permArray,
        Nlefs,
        deathProbII=deathArray_C2,
        stallFalloffProbII=stallDeathArray_C2,
        pauseProbII=pauseArray_C2,
        deathProbIII=deathArray_Coh,
        stallFalloffProbIII=stallDeathArray_Coh,
        pauseProbIII=pauseArray_Coh,
        LEFtype=LEFtypeArray,
        activeStatus=activeArray
    )
    return SMCTran





#polymer object
polymer = np.array([]).reshape(0,3)

if True:
    #initialize polymer position 
    for ii in range(numChr):
        Rpoly= Rsph*(chrSizes[ii]/numChr)**(1./3.)

        def rw_constraint_func(start_pos, new_pos):
            """Return True if new_pos < Rpoly away from start_pos and new_pos is within Rsph of origin"""
            if PBC:
                if ((np.sum((np.array(start_pos)-np.array(new_pos))**2)<Rpoly**2)
                    and (np.sum(np.array(new_pos)**2)<Rsph**2)):
                    return True
                else:
                    return False
            else:
                if (np.sum(np.array(new_pos[:2]))**2 < Rcyl**2) and (abs(new_pos[2]) < 0.5*Hcyl):
                    return True
                else:
                    return False

        if (ii==0):# and numChr>1:
            starting_pt=(0,0,-0.5*Hcyl) 
        # with cyl BC, might have problems initializing Nchr>1
        else:#elif ii>1:
            while True:
                starting_pt = Rsph*np.random.uniform(-1,1,3)
                if np.sum(starting_pt**2)<Rsph**2:
                    break
    
        polymer=np.concatenate((polymer, simUtils.create_constrained_random_walk(chrSizes[ii], 
                                                                                 rw_constraint_func, 
                                                                                 starting_point=starting_pt)))
    #center the polymer
    polymer = polymer - np.mean(polymer,axis=0)

    

# ------------feed smcTran to the bond updater---
SMCTran = initModel() 
init_num_steps=36000 # 5 CII lifetimes
SMCTran.steps(init_num_steps)  # first steps to "equilibrate" SMC dynamics. If desired of course.
bondUpdater = smcBondUpdater(SMCTran)  # now feed this thing to bond updater (do it once!)


#current BC variables
Hcyl_t=Hcyl
Rcyl_t=Rcyl
density_t=density
kConfine_t=kConfine
kConfineSph_t=0.

#counters of transitions in numbrers of LEFs
num_C1_activated = 0
num_C1_deactivated = 0
num_C2_deactivated = 0
num_Coh_activated = 0

#set order of LEFs for C1 deactivation
ordering_array= [ele for ele in zip(np.arange(N_C1_max), np.random.uniform(0,1,N_C1_max))]
ordering_array.sort(key=lambda x:x[1])
deactivation_order = [ordering_array[ii][0] for ii in range(len(ordering_array))]
#and C2
ordering_array2= [ele for ele in zip(np.arange(N_C1_max,N_C1_max+N_C2), np.random.uniform(0,1,N_C2))]
ordering_array2.sort(key=lambda x:x[1])
deactivation_order_C2 = [ordering_array2[ii][0] for ii in range(len(ordering_array2))]


# now polymer simulation code starts
for updaterCount in range(updaterInitsTotal):
    relative_time = (updaterCount - updaterInitsSkip) * restartUpdaterEveryBlocks
    print("updater init", updaterCount, "/", updaterInitsTotal, "texpt=", relative_time/2/60,"min")
    doSave=False
    if updaterCount >= updaterInitsSkip:
        if relative_time <= END_PROMETA:
            doSave=True
        elif (relative_time >= BEGIN_ANATELO) and (relative_time <= END_ANATELO):
            doSave=True
        elif (relative_time >= BEGIN_EARLYG1) and (relative_time <= END_EARLYG1):
            doSave=True
        elif (relative_time >= BEGIN_MIDG1) and (relative_time <= END_MIDG1):
            doSave=True
        elif (relative_time >= BEGIN_LATEG1) and (relative_time <= END_LATEG1):
            doSave=True


    THERMOSTAT_TO_USE=THERMOSTAT
    max_Ek=20.

    reporterList=[reporter]
    sim = simulation.Simulation(platform="CUDA",
            GPU=gpu,
            integrator=INTEGRATOR,
            collision_rate=THERMOSTAT_TO_USE,
            timestep=dt,#40 is ok here
            max_Ek=max_Ek,
            N=LENGTH,
            PBCbox=pbc_param,
            save_decimals=3, 
            reporters=reporterList)# set up GPU here

    sim.set_data(polymer, center=False)
    
    if not PBC:
        #confining cylinder
        #would be nice to decouple density change and BC change. but BC change is accompanied by axial shortening. 

        #idea: usually, t_densityMid=t_densityEnd. However, if we want to change from density to densityInter smoothly, set t_densityEnd to be t_densityStart + 2*(t_densityMid-t_densityStart). t_sphere_inflate is a totally different density time, changing density from densityInter to densityInflate. So in the end, we maintain obsolete names related to BCs, but the density is independently controlled

        if relative_time <= BC_change_end:
            if (relative_time >= cylinderShortenStart) and (relative_time <= cylinderShortenEnd):
                #linearly decrease height with time
                Hcyl_t = Hcyl - cylinderFinalSizeFactor*Hcyl*(relative_time-cylinderShortenStart+restartUpdaterEveryBlocks)/(cylinderShortenEnd-cylinderShortenStart+restartUpdaterEveryBlocks)
                #linearly decrease density w/ time
                density_t = density - 0.5*(density - densityInter)*(relative_time-cylinderShortenStart+restartUpdaterEveryBlocks)/(cylinderShortenEnd-cylinderShortenStart+restartUpdaterEveryBlocks)
                Rcyl_t =  np.sqrt(LENGTH / (6 * Hcyl_t * density_t))
            if relative_time >= BC_change_start:
                kConfine_t= kConfine * (1.- (relative_time-BC_change_start+restartUpdaterEveryBlocks) / (BC_change_end-BC_change_start+restartUpdaterEveryBlocks) )  # e.g., changing from t=25 to t=30 once per minute is 6 steps.
            if kConfine_t>0.:
                sim.add_force(forces.cylindrical_confinement(sim, r=Rcyl_t, k=kConfine_t, bottom= -0.5*Hcyl_t, top=0.5*Hcyl_t))
                if PIN_ENDS and (relative_time <= BC_change_start):
                    sim.add_force(forces.tether_particles(sim, particles=[0,LENGTH-1], k=[kConfine_t,kConfine_t,kConfine_t],
                                                          positions=[[0.,0.,0.5*Hcyl_t],[0.,0.,-0.5*Hcyl_t]]))
        #confining sphere
        if relative_time >= BC_change_start:
            #here is another option allowing inflation of the sphere, in case we want to have only spherical BC, but still have a density change.
            if (relative_time >= t_sphere_inflate_start) and (relative_time <= t_sphere_inflate_end):
                density_t= densityInter + (densityInflate-densityInter)*(relative_time-t_sphere_inflate_start+restartUpdaterEveryBlocks)/(t_sphere_inflate_end-t_sphere_inflate_start+restartUpdaterEveryBlocks)
                Rsph_t=0.5*((LENGTH/density_t)**(1./3.))
            elif relative_time <= t_sphere_inflate_start:
                Rsph_t=Rsph
            else:
                Rsph_t=0.5*((LENGTH/densityInflate)**(1./3.))
            kConfineSph_t= kConfine - kConfine_t
            if kConfineSph_t>0.:
                sim.add_force(forces.spherical_confinement(sim, r=Rsph_t, k=kConfineSph_t))

    if relative_time >= t_comps_on:
        COMPS_ON=1.
    else:
        COMPS_ON=0.
    if relative_time >= t_poor_solvent_off :
        POOR_SOLVENT=0.
    else:
        POOR_SOLVENT=1.


    extraHardList=[]
    if not ignoreAdjacent:
        extraHardList=B_list
    if (EPSILON_C > 0.) or (EPSILON_A > 0.):
        if not ignoreAdjacent:
            extraHardList= extraHardList + C_list
        nonbondedForce=forces.heteropolymer_SSW
        monoTypeList=np.zeros(LENGTH,dtype=int)
        monoTypeList[B_list] = 1
        monoTypeList[C_list] = 2
        extraHardList=np.unique(extraHardList)
        nonbondedKeys={
                       "interactionMatrix":np.array([[COMPS_ON*EPSILON_A, COMPS_ON*EPSILON_AB, COMPS_ON*EPSILON_AC],
                                                     [COMPS_ON*EPSILON_AB, COMPS_ON*EPSILON_B, COMPS_ON*EPSILON_BC],
                                                     [COMPS_ON*EPSILON_AC, COMPS_ON*EPSILON_BC, EPSILON_C]]),
                       "monomerTypes":monoTypeList,
                       "extraHardParticlesIdxs":extraHardList,
                       "repulsionEnergy": REPEL,
                       "attractionEnergy": EPSILON_ALL*POOR_SOLVENT,
                      }
    else:
        nonbondedForce=forces.selective_SSW
        extraHardList=np.unique(extraHardList)
        nonbondedKeys={
                "repulsionEnergy": REPEL,
                "stickyParticlesIdxs": B_list,
                "extraHardParticlesIdxs": extraHardList, # empty unless ignoreAdjacent is False
                "attractionEnergy": EPSILON_ALL*POOR_SOLVENT,
                "selectiveAttractionEnergy": COMPS_ON*EPSILON_B
                }

    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(int(np.sum(chrSizes[:ii])), int(np.sum(chrSizes[:ii])+chrSizes[ii]), False) for ii in range(numChr)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": polyBondWiggleDist,  # Bond distance will fluctuate +-  on average
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5,
            },
            nonbonded_force_func=nonbondedForce, # force & keys chosen above
            nonbonded_force_kwargs=nonbondedKeys,
            except_bonds= ignoreAdjacent, #if True, don't calculate nonbonded forces for adjacent monos
        ))
    
    
    # ------------ initializing bondUpdater; adding bonds ---------
    # copied from addBond
    kbond = sim.kbondScalingFactor / (smcBondWiggleDist**2)
    bondDist = smcBondDist * sim.length_scale

    activeParams = {"length": bondDist, "k": kbond}
    inactiveParams = {"length": bondDist, "k": 0}
    bondUpdater.setParams(activeParams, inactiveParams)

    # deal with changes in LEF numbers
    lefs_to_activate=[]
    lefs_to_deactivate=[]

    if (relative_time >= t_C1_incStart) and (relative_time <= t_C1_incEnd):
        #activate some number of LEFs. use a target number to ensure we stay on track & don't fall victim to rounding errors
        target_num_activated = (N_C1_max-N_C1_start) * (relative_time-t_C1_incStart+1) / (t_C1_incEnd-t_C1_incStart+1)
        Nactivate= int(np.round(target_num_activated - num_C1_activated,0))
        lefs_to_activate.extend([nn for nn in range(N_C1_start+num_C1_activated,N_C1_start+num_C1_activated+Nactivate)])
        num_C1_activated+=Nactivate
        print("activate C1:",Nactivate,"list:",lefs_to_activate)

    elif (relative_time > t_C1_decStart) and (relative_time <= t_C1_decEnd): #use > t_C1_decStart instead of >= here since we typically take t_C1_incEnd==t_C1_decStart, and we have an elif statement
        if t_C1_decEnd>t_C1_decStart:
            target_num_deactivated = N_C1_max * (relative_time-t_C1_decStart) / (t_C1_decEnd-t_C1_decStart) #corresponding to the above remark, we remove the +1 in the numerator and denominator
            Ndeactivate=int(np.round(target_num_deactivated-num_C1_deactivated))
        else:
            Ndeactivate=N_C1_max
        lefs_to_deactivate.extend(deactivation_order[num_C1_deactivated:num_C1_deactivated+Ndeactivate])
        num_C1_deactivated+=Ndeactivate
        print("deactivate C1:",Ndeactivate,"list:",lefs_to_deactivate)

    if (relative_time > t_C2_decStart) and (relative_time <= t_C2_decEnd):
        if t_C2_decEnd>t_C2_decStart:
            target_num_deactivated= N_C2 * (relative_time-t_C2_decEnd) / (t_C2_decEnd - t_C2_decStart)
            Ndeactivate=int(np.round(target_num_deactivated-num_C2_deactivated))
        else:
            Ndeactivate=N_C2
        lefs_to_deactivate.extend(deactivation_order_C2[num_C2_deactivated:num_C2_deactivated+Ndeactivate])
        num_C2_deactivated+=Ndeactivate
        print("deactivate C2:",Ndeactivate,"list:",lefs_to_deactivate)

    newDeathRate=[]
    if (relative_time== t_C1_changeLife):
        newDeathRate.append(1./(FACTOR_CHANGE_C1*LIFETIME_C1))

    if (relative_time >= t_Coh_incStart) and (relative_time <= t_Coh_incEnd):
        target_num_activated = N_Coh * (relative_time-t_Coh_incStart+1) / (t_Coh_incEnd-t_Coh_incStart+1)
        Nactivate=int(np.round(target_num_activated - num_Coh_activated,0)) 
        lefs_to_activate.extend([nn for nn in range(N_C1_max+N_C2+num_Coh_activated, N_C1_max+N_C2+num_Coh_activated+Nactivate)])
        num_Coh_activated+=Nactivate
        print("activate Coh:",Nactivate,"list:",lefs_to_activate)

    # this step actually puts all bonds in and sets first bonds to be what they should be
    bondUpdater.setup(
        bondForce=sim.force_dict['harmonic_bonds'],
        blocks=restartUpdaterEveryBlocks,
        smcStepsPerBlock=smcStepsPerBlock,
        LEFsToActivate=lefs_to_activate,
        LEFsToDeactivate=lefs_to_deactivate,
        newfalloff1=newDeathRate
    )  # now only one step of SMC per step
    

    # If your simulation does not start, consider using energy minimization below
    if (updaterCount==0) or (updaterCount==updaterInitsSkip):
        sim.local_energy_minimization() 
    else:
        sim._apply_forces()
    
    sim.step=prev_step 
    for i in range(restartUpdaterEveryBlocks):
        if (i % saveEveryBlocks == (saveEveryBlocks - 1)): 
            sim.do_block(steps=polyBlockSteps,save=doSave)
            if doSave and i < saveEveryBlocks:
                pickle.dump(
                    curBonds,
                    open(os.path.join(folder, "SMC{0}.dat".format(sim.step)), "wb"),
                    )
        else:
            sim.integrator.step(polyBlockSteps)  # do polyBlockSteps without getting the positions from the GPU (faster)
            sim.step+=polyBlockSteps # integrator doesn't increment steps
        if i < restartUpdaterEveryBlocks - 1: 
            curBonds, pastBonds, numBondsAdded = bondUpdater.step(sim.context, countSteps=True)  # this updates bonds. You can do something with bonds here
    polymer = sim.get_data()  # save data and step, and delete the simulation
    prev_step = sim.step
    del sim
    
    reporter.blocks_only = True  # Write output hdf5-files only for blocks
    
    time.sleep(0.1)  
   
     
reporter.dump_data()
       
 

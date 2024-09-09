import numpy as np

class smcBondUpdater(object):
    def __init__(self, smcTransObject):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.smcObject = smcTransObject
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict

    def setup(self, bondForce, blocks=100, smcStepsPerBlock=1, LEFsToActivate=[], LEFsToDeactivate=[], newfalloff1=[]):
        """
        A method that extracts info from smcTranslocator object
        to create a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """

        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        # precalculating all bonds
        allBonds = []
        
        #randomize activation/deactivation
        if len(LEFsToActivate)>0:
            activationTimes=np.random.randint(0,blocks,len(LEFsToActivate))
            activationTimes=np.sort(activationTimes)
            #activate lowest indexed LEFs first 
        if len(LEFsToDeactivate)>0:
            deactivationTimes=np.random.randint(0,blocks,len(LEFsToDeactivate))
            #want random deactivation order

        #check if falloff times change
        if len(newfalloff1)>0:
            self.smcObject.change_falloff(newfalloff1[0])

        for dummy in range(blocks):
            LEFsA=[]
            LEFsD=[]
            for nn in range(len(LEFsToActivate)):#consider if len(activationTimes)>0: while activationTimes[0]==i: activate(activateLEFs.pop(0)) activationTimes.pop(0
                if activationTimes[nn] == dummy:
                    LEFsA.append(LEFsToActivate[nn])
            for nn in range(len(LEFsToDeactivate)):#while could be used w/ deactivation if I sort deactive list with deactivationTimes
                if deactivationTimes[nn] == dummy:
                    LEFsD.append(LEFsToDeactivate[nn])            

            self.smcObject.steps(smcStepsPerBlock, LEFsA, LEFsD)
            left, right = self.smcObject.getLEFs()
            left_sites = left[left>=0]
            right_sites = right[right>=0]
            bonds = [(int(i), int(j)) for i, j in zip(left_sites, right_sites)]
            allBonds.append(bonds)

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))


        # adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)
        self.bondToInd = {i: j for i, j in zip(self.uniqueBonds, self.bondInds)}
        return self.curBonds, []

    def step(self, context, verbose=False, countSteps=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        countSteps option returns number of new bonds added (not included by default to maintain backward compatibility)
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if countSteps:
            numNewBonds = len(np.unique(bondsAdd,axis=0))
        if verbose:
            print(
                "{0} bonds stay, {1} new bonds, {2} bonds removed".format(
                    len(bondsStay), len(bondsAdd), len(bondsRemove)
                )
            )
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        if countSteps:
            return self.curBonds, pastBonds, numNewBonds
        else:
            return self.curBonds, pastBonds



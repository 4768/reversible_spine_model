"""
netParams.py 

High-level specifications for M1 network model using NetPyNE

Contributors: salvadordura@gmail.com
"""

from netpyne import specs, sim
from neuron import h
import pickle, json
import random
import numpy as np
import math
import os

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

netParams.version = 56

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfg import cfg

num_PT = 20
num_PV = 10
num_SOM = 10
num_dict = {'PT': num_PT, 'PV': num_PV, 'SOM': num_SOM}
seclist = {'PT': {'dend':[], 'all':[]}, 'PV': [], 'SOM': []} #'all' except axon




#------------------------------------------------------------------------------
#
#  Random sequence generation
#
#------------------------------------------------------------------------------
class RandomSequenceNetParams:
    def __init__(self):
        self.randomlist = None
    
    def initializeRandomSequence(self, simConfig):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # Set random seeds consistently across nodes
        sd = int(os.environ.get('NEURON_SEED', '1234'))
        random.seed(sd)
        if rank == 0:
            print(f'Using {sd} -- random seed for connection generation')
        # Initialize the randomlist structure based on population parameters
        self.randomlist = {
            'EI': {k: [[[] for _ in range(num_PT)] 
                for _ in range(num_dict[k])] 
                for k in ['PV', 'SOM']},
            'PVE': [[{'soma': []} for _ in range(num_dict['PV'])]
                    for _ in range(num_PT)],
            'SOME': [[{s: [] for s in ['Adend1', 'Adend2', 'Adend3', 'Bdend']}
                    for _ in range(num_dict['SOM'])]
                    for _ in range(num_PT)],
            'II': {posttype: { pretype: [[[] for _ in range(num_dict[pretype] - 1 if pretype == posttype 
                    else num_dict[pretype])]
                    for _ in range(num_dict[posttype])]
                    for pretype in ['PV', 'SOM']}
                for posttype in ['PV', 'SOM']},
            'LongE': {k: {sec: [[[] for _ in range(numCells_long)] 
                    for _ in range(num_PT)] 
                    for sec in ['Adend1', 'Adend2', 'Adend3', 'Bdend']}
                    for k in ['S1_4K', 'S1_12K', 'OC']}
        }

        if rank == 0:  # Only generate on master node
            # Generate weights and locations
            weights, seglocs = self._generate_random_values()
            # Process connections
            self._process_all_connections(weights, seglocs)
            
            
        # Broadcast fram rank 0 to all other ranks
        self.randomlist = comm.bcast(self.randomlist, root=0)

    def _generate_random_values(self):
        weights = {}
        seglocs = {}

        # Generate weights for each postsynaptic neuron (ipost)
        for celltype in ['PT', 'PV', 'SOM']:
            weights[celltype] = {}
            seglocs[celltype] = self._create_segment_locations(celltype)

            if celltype == 'PT':
                # For PT cells, generate weights for each compartment
                weights[celltype]['soma'] = []
                for ipost in range(num_PT):
                    # Sample weights for this ipost
                    weights[celltype]['soma'].append(
                        self.sample_custom_distribution(
                            self._calculate_total_synapses(celltype)['axosomatic'] // num_PT,
                            0.4, 3, 15 
                        )
                    )
                for sec in seclist['PT']['dend']:
                    weights[celltype][sec] = []
                    for ipost in range(num_PT):
                        weights[celltype][sec].append(
                            self.sample_custom_distribution(
                                self._calculate_total_synapses(celltype)['axodendritic'][sec] // num_PT,
                                0.4, 3, 15 
                            )
                        )
            else:
                weights[celltype] = []
                for ipost in range(num_dict[celltype]):
                    weights[celltype].append(
                        self.sample_custom_distribution(
                            self._calculate_total_synapses(celltype)['axosomatic'] // num_dict[celltype],
                            0.4, 3, 15  
                        )
                    )

        return weights, seglocs

    def sample_custom_distribution(self, n_samples, prob, weight_threshold, maxweight=1.0):

        # Generate random numbers from a uniform distribution
        u = [random.uniform(0, 1) for _ in range(n_samples)]

        # Allocate samples as a list of zeros
        samples = [0.0] * n_samples
        for i, ui in enumerate(u):
            if ui <= prob:  # prob in [0, weight]
                samples[i] = weight_threshold * (ui / prob)  # Scale uniformly in [0, weight]
            else:  # 1-prob in (weight, 1]
                samples[i] = weight_threshold + (ui - prob) * (maxweight-weight_threshold) / (1-prob)  # Scale uniformly in (weight, 1]
        return samples

    def _calculate_total_synapses(self, celltype):
        # Calculate based on connection rules and population sizes
        total = {'axosomatic':0, 'axodendritic':{k: 0 for k in seclist['PT']['dend']}}
        if celltype == 'PT':
            total['axosomatic'] = (num_PT * 
                     num_PV)  # PVE connections
            for sec in seclist['PT']['dend']:
                total['axodendritic'][sec] += math.ceil(nseg['PT'][sec]/numCells_long) * numCells_long * num_PT * 3  # LongE connections (3 types)
                total['axodendritic'][sec] += (num_PT *
                        num_SOM)  # SOME connections
        else:  # PV or SOM
            total['axosomatic'] += (num_dict[celltype] * 
                     num_PT)  # EI connections
            total['axosomatic'] += (num_dict[celltype] * 
                     (num_PV + 
                      num_SOM - 1))  # II connections
        return total

    def _create_segment_locations(self, celltype):
        seglocs = {}

        if celltype == 'PT':
            # For PT cells, generate segment locations for each compartment
            for sec in seclist['PT']['all']:
                seglocs[sec] = []
                for ipost in range(num_PT):
                    # Generate segment locations for this ipost
                    seglocs[sec].append(
                        [1 / (2 * nseg[celltype][sec]) + i * 1 / nseg[celltype][sec]
                        for i in range(math.ceil(nseg[celltype][sec]))]
                    )
                    if sec == 'soma':
                        k = self._calculate_total_synapses(celltype)['axosomatic'] // num_PT - math.ceil(nseg[celltype][sec])
                    else:
                        k = self._calculate_total_synapses(celltype)['axodendritic'][sec] // num_PT - math.ceil(nseg[celltype][sec]) 
                    seglocs[sec][ipost].extend(random.choices(seglocs[sec][ipost], k = k))
                    random.shuffle(seglocs[sec][ipost])
        else:
            # For PV and SOM cells, generate segment locations for each ipost
            seglocs = []
            for ipost in range(num_dict[celltype]):
                seglocs.append(
                    [1 / (2 * nseg[celltype]['soma']) + i * 1 / nseg[celltype]['soma']
                    for i in range(math.ceil(nseg[celltype]['soma']))]
                )
                seglocs[ipost].extend(
                    random.choices(seglocs[ipost],
                                k=self._calculate_total_synapses(celltype)['axosomatic'] // num_dict[celltype] - math.ceil(nseg[celltype]['soma']))
                )
                random.shuffle(seglocs[ipost])

        return seglocs
    
    

    def _process_all_connections(self, weights, seglocs):
        for conntype in self.randomlist.keys():
            self._process_connections(conntype, weights, seglocs)

    def _process_connections(self, conntype, weights, seglocs):
        if conntype == 'LongE':
            synsperconn = {}
            for sec in seclist['PT']['all']:
                synsperconn[sec] = math.ceil(nseg['PT'][sec]/numCells_long)
            for ipost in range(num_PT):
                for pretype in ['S1_4K', 'S1_12K', 'OC']:
                    for ipre in range(numCells_long):
                        self._process_single_connection(conntype, 'PT', pretype, 
                                                    ipost, ipre, weights, seglocs, synsperconn)
        
        elif conntype in ['PVE', 'SOME']:
            if conntype == 'PVE':
                pretype = 'PV'
            else:
                pretype = 'SOM'
            for ipost in range(num_PT):
                for ipre in range(num_dict[pretype]):
                    self._process_single_connection(conntype, 'PT', pretype, 
                                                     ipost, ipre, weights, seglocs, 1)
        
        elif conntype == 'EI':
            for posttype in ['PV', 'SOM']:
                for ipost in range(num_dict[posttype]):
                    for ipre in range(num_PT):
                        self._process_single_connection(conntype, posttype, None, 
                                                     ipost, ipre, weights, seglocs, 1)
        
        elif conntype == 'II':
            for posttype in ['PV', 'SOM']:
                for pretype in ['PV', 'SOM']:
                    num_cells = num_dict[pretype]
                    for ipost in range(num_dict[posttype]):
                        for ipre in range(num_cells - (1 if posttype == pretype else 0)):
                            self._process_single_connection(conntype, posttype, pretype, 
                                                         ipost, ipre, weights, seglocs, 1)

    def _process_single_connection(self, conntype, posttype, pretype, ipost, ipre, weights, seglocs, synsperconn):
        if posttype == 'PT':
            selected_weights = {}
            selected_locs = {}
            if conntype == 'LongE':
                for sec in seclist['PT']['dend']:
                    selected_weights[sec] = [weights[posttype][sec][ipost].pop() for _ in range(synsperconn[sec])]
                    selected_locs[sec] = [seglocs[posttype][sec][ipost].pop() for _ in range(synsperconn[sec])]
            else:
                if pretype == 'PV':
                    selected_weights['soma'] = [weights[posttype]['soma'][ipost].pop() for _ in range(synsperconn)]
                    selected_locs['soma'] = [seglocs[posttype]['soma'][ipost].pop() for _ in range(synsperconn)]
                else:
                    for sec in seclist['PT']['dend']:
                        selected_weights[sec] = [weights[posttype][sec][ipost].pop() for _ in range(synsperconn)]
                        selected_locs[sec] = [seglocs[posttype][sec][ipost].pop() for _ in range(synsperconn)]
        else:
            selected_weights = [weights[posttype][ipost].pop() for _ in range(synsperconn)]
            selected_locs = [seglocs[posttype][ipost].pop() for _ in range(synsperconn)]

        # Store the weights and locations in randomlist
        if conntype == 'II':
            self.randomlist[conntype][posttype][pretype][ipost][ipre] = list(zip(selected_locs, selected_weights))
        elif conntype == 'EI':
            self.randomlist[conntype][posttype][ipost][ipre] = list(zip(selected_locs, selected_weights))
        elif conntype == 'LongE':
            for sec in seclist['PT']['dend']:
                self.randomlist[conntype][pretype][sec][ipost][ipre] = list(zip(selected_locs[sec], selected_weights[sec]))
        else:
            if pretype == 'SOM':
                for sec in seclist['PT']['dend']:
                    self.randomlist[conntype][ipost][ipre][sec] = list(zip(selected_locs[sec], selected_weights[sec]))
            else:
                self.randomlist[conntype][ipost][ipre]['soma'] = list(zip(selected_locs['soma'], selected_weights['soma']))




#------------------------------------------------------------------------------
#
# NETWORK PARAMETERS
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------
netParams.scale = cfg.scale # Scale factor for number of cells
netParams.sizeX = cfg.sizeX # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ # z-dimension (horizontal depth) size in um
netParams.shape = 'cylinder' # cylindrical (column-like) volume

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0 # Connection weight scale factor (default if no model specified)
#netParams.scaleConnWeightModels = {'HH_full': 1.0} #scale conn weight factor for each cell model
#netParams.scaleConnWeightNetStims = 1.0 #0.5  # scale conn weight factor for NetStims
netParams.defaultThreshold = -5.0 # spike threshold, 10 mV is NetCon default, lower it for all cells
netParams.defaultDelay = 2.0 # default conn delay (ms)
netParams.propVelocity = 500.0 # propagation velocity (um/ms)
netParams.probLambda = 100.0  # length constant (lambda) for connection probability decay (um)
netParams.defineCellShapes = True  # convert stylized geoms to 3d points


# special condition to change Kgbar together with ih when running batch
# note min Kgbar is assumed to be 0.5, so this is set here as an offset 
#if cfg.makeKgbarFactorEqualToNewFactor:
#    cfg.KgbarFactor = 0.5 + cfg.modifyMechs['newFactor']

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
#cellModels = ['HH_full']

cellModels = ['HH_reduced']
layer = {'1':[0.0, 0.1],'5B': [0.47,0.8],'longS1_4K': [2.2,2.3],'longS1_12K': [2.2,2.3], 'longOC': [2.2,2.3]}  # normalized layer boundaries

netParams.correctBorder = {'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
                        'yborders': [layer['1'][0], layer['5B'][0], layer['5B'][1]]}  # correct conn border effect



#------------------------------------------------------------------------------
## Load cell rules previously saved using netpyne format
cellParamLabels = ['PV_simple', 'SOM_simple', 'PT5B_reduced']# ['VIP_reduced', 'NGF_simple','PT5B_full'] #  # list of cell rules to load from file
loadCellParams = cellParamLabels
saveCellParams = False #True

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/'+ruleLabel+'_cellParams.pkl')
    
    # Adapt K gbar
    if ruleLabel in ['IT2_reduced', 'IT4_reduced', 'IT5A_reduced', 'IT5B_reduced', 'IT6_reduced', 'CT6_reduced', 'IT5A_full']:
        cellRule = netParams.cellParams[ruleLabel]
        for secName in cellRule['secs']:
            for kmech in [k for k in cellRule['secs'][secName]['mechs'].keys() if k.startswith('k') and k!='kBK']:
                cellRule['secs'][secName]['mechs'][kmech]['gbar'] *= cfg.KgbarFactor 
                
    if ruleLabel in ['PT5B_reduced']:
        cellRule = netParams.cellParams[ruleLabel]
        seclist['PT']['dend'] = cellRule['secLists']['alldend']
        seclist['PT']['all'] = [sec for sec in cellRule['secs'] if sec not in ['axon']]
        secL_PT = {k: 0.0 for k in cellRule['secs']}
        for secName in cellRule['secs']:
            #cellRule['secs'][secName]['geom']['L'] = 5
            secL_PT[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None
    elif ruleLabel in ['PV_simple']:
        cellRule = netParams.cellParams[ruleLabel]
        secL_PV = {'soma': 0.0, 'dend': 0.0, 'axon': 0.0}
        for secName in cellRule['secs']:
            #cellRule['secs'][secName]['geom']['L'] = 5
            secL_PV[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None
    elif ruleLabel in ['SOM_simple']:
        cellRule = netParams.cellParams[ruleLabel]
        secL_SOM = {'soma': 0.0, 'dend': 0.0, 'axon': 0.0}
        for secName in cellRule['secs']:
            #cellRule['secs'][secName]['geom']['L'] = 5
            secL_SOM[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None
  
print('sec length:', secL_PT, secL_PV, secL_SOM)
#------------------------------------------------------------------------------
## PT5B full cell model params (700+ comps)
if 'PT5B_full' not in loadCellParams:
    ihMod2str = {'harnett': 1, 'kole': 2, 'migliore': 3}
    cellRule = netParams.importCellParams(label='PT5B_full', conds={'cellType': 'PT', 'cellModel': 'HH_full'},
      fileName='cells/PTcell.hoc', cellName='PTcell', cellArgs=[ihMod2str[cfg.ihModel], cfg.ihSlope], somaAtOrigin=True)
    nonSpiny = ['apic_0', 'apic_1']
    netParams.addCellParamsSecList(label='PT5B_full', secListName='perisom', somaDist=[0, 50])  # sections within 50 um of soma
    netParams.addCellParamsSecList(label='PT5B_full', secListName='below_soma', somaDistY=[-600, 0])  # sections within 0-300 um of soma
    for sec in nonSpiny: cellRule['secLists']['perisom'].remove(sec)
    cellRule['secLists']['alldend'] = [sec for sec in cellRule.secs if ('dend' in sec or 'apic' in sec)] # basal+apical
    cellRule['secLists']['apicdend'] = [sec for sec in cellRule.secs if ('apic' in sec)] # apical
    cellRule['secLists']['spiny'] = [sec for sec in cellRule['secLists']['alldend'] if sec not in nonSpiny]
    # Adapt ih params based on cfg param
    for secName in cellRule['secs']:
        for mechName,mech in cellRule['secs'][secName]['mechs'].items():
            if mechName in ['ih','h','h15', 'hd']: 
                mech['gbar'] = [g*cfg.ihGbar for g in mech['gbar']] if isinstance(mech['gbar'],list) else mech['gbar']*cfg.ihGbar
                if cfg.ihModel == 'migliore':   
                    mech['clk'] = cfg.ihlkc  # migliore's shunt current factor
                    mech['elk'] = cfg.ihlke  # migliore's shunt current reversal potential
                if secName.startswith('dend'): 
                    mech['gbar'] *= cfg.ihGbarBasal  # modify ih conductance in soma+basal dendrites
                    mech['clk'] *= cfg.ihlkcBasal  # modify ih conductance in soma+basal dendrites
                if secName in cellRule['secLists']['below_soma']: #secName.startswith('dend'): 
                    mech['clk'] *= cfg.ihlkcBelowSoma  # modify ih conductance in soma+basal dendrites

        # Adapt K gbar
        for kmech in [k for k in cellRule['secs'][secName]['mechs'].keys() if k.startswith('k') and k!='kBK']:
            cellRule['secs'][secName]['mechs'][kmech]['gbar'] *= cfg.KgbarFactor 

    # NOT Reduce dend Na to HAVE dend spikes 
    for secName in cellRule['secLists']['alldend']:
        cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0153130368342 * cfg.dendNa # 1
    
                
    cellRule['secs']['soma']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.somaNa # 5 
    cellRule['secs']['axon']['mechs']['nax']['gbar'] = 0.0153130368342  * cfg.axonNa # 5 
    cellRule['secs']['axon']['geom']['Ra'] = 137.494564931 * cfg.axonRa # 0.005
    # Remove Na (TTX)
    if cfg.removeNa:
        for secName in cellRule['secs']: cellRule['secs'][secName]['mechs']['nax']['gbar'] = 0.0
    #netParams.addCellParamsWeightNorm('PT5B_full', 'conn/PT5B_full_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # load weight norm
    if saveCellParams: netParams.saveCellParamsRule(label='PT5B_full', fileName='cells/PT5B_full_cellParams.pkl')


#------------------------------------------------------------------------------
# Reduced cell model params (6-comp) 
reducedCells = {  # layer and cell type for reduced cell models
    #'PT5B_reduced': {'layer': '5B', 'cname': 'SPI6',  'carg':  None}
    }

reducedSecList = {  # section Lists for reduced cell model
    'alldend':  ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'spiny':    ['Adend1', 'Adend2', 'Adend3', 'Bdend'],
    'apicdend': ['Adend1', 'Adend2', 'Adend3'],
    'perisom':  ['soma']}
 

for label, p in reducedCells.items():  # create cell rules that were not loaded 
    if label not in loadCellParams:
        cellRule = netParams.importCellParams(label=label, conds={'cellType': label[0], 'cellModel': 'HH_reduced', 'ynorm': layer[p['layer']]},
        fileName='cells/'+p['cname']+'.py', cellName=p['cname'], cellArgs={'params': p['carg']} if p['carg'] else None)
        dendL = (layer[p['layer']][0]+(layer[p['layer']][1]-layer[p['layer']][0])/2.0) * cfg.sizeY  # adapt dend L based on layer
        for secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']: 
            cellRule['secs'][secName]['geom']['L'] = dendL / 3.0  
        for k,v in reducedSecList.items(): cellRule['secLists'][k] = v  # add secLists
        netParams.addCellParamsWeightNorm('PT5B_reduced', 'conn/'+'PT5B_reduced'+'_weightNorm.pkl', threshold=cfg.weightNormThreshold)  # add weightNorm
        if saveCellParams: netParams.saveCellParamsRule(label='PT5B_reduced', fileName='cells/'+'PT5B_reduced'+'_cellParams.pkl')
        

        # set 3d points
        offset, prevL = 0, 0
        somaL = netParams.cellParams[label]['secs']['soma']['geom']['L']
        for secName, sec in netParams.cellParams[label]['secs'].items():
            sec['geom']['pt3d'] = []
            if secName in ['soma', 'Adend1', 'Adend2', 'Adend3']:  # set 3d geom of soma and Adends
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
                prevL = float(prevL + sec['geom']['L'])
                sec['geom']['pt3d'].append([offset+0, prevL, 0, sec['geom']['diam']])
            if secName in ['Bdend']:  # set 3d geom of Bdend
                sec['geom']['pt3d'].append([offset+0, somaL, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+sec['geom']['L'], somaL, 0, sec['geom']['diam']])        
            if secName in ['axon']:  # set 3d geom of axon
                sec['geom']['pt3d'].append([offset+0, 0, 0, sec['geom']['diam']])
                sec['geom']['pt3d'].append([offset+0, -sec['geom']['L'], 0, sec['geom']['diam']])   



#------------------------------------------------------------------------------
## PV cell params (3-comp)
if 'PV_simple' not in loadCellParams:
    cellRule = netParams.importCellParams(label='PV_simple', conds={'cellType':'PV', 'cellModel':'HH_simple'}, 
        fileName='cells/FS3.hoc', cellName='FScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    #netParams.addCellParamsWeightNorm('PV_simple', 'conn/PV_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='PV_simple', fileName='cells/PV_simple_cellParams.pkl')


#------------------------------------------------------------------------------
## SOM cell params (3-comp)
if 'SOM_simple' not in loadCellParams:
    cellRule = netParams.importCellParams(label='SOM_simple', conds={'cellType':'SOM', 'cellModel':'HH_simple'}, 
        fileName='cells/LTS3.hoc', cellName='LTScell1', cellInstance = True)
    cellRule['secLists']['spiny'] = ['soma', 'dend']
    #netParams.addCellParamsWeightNorm('SOM_simple', 'conn/SOM_simple_weightNorm.pkl', threshold=cfg.weightNormThreshold)
    if saveCellParams: netParams.saveCellParamsRule(label='SOM_simple', fileName='cells/SOM_simple_cellParams.pkl')



#------------------------------------------------------------------------------
# Population parameters
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
## load densities
with open('cells/cellDensity.pkl', 'rb') as fileObj: density = pickle.load(fileObj)['density']


## Local populations
#netParams.popParams['PT5B'] =   {'cellModel': cfg.cellmod['PT5B'], 'cellType': 'PT', 'ynormRange': layer['5B'], 'numCells': num_PT}
netParams.popParams['PT5B_reduced'] =   {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynormRange': layer['5B'], 'numCells': num_PT}
netParams.popParams['SOM5B'] =  {'cellModel': 'HH_simple',         'cellType': 'SOM','ynormRange': layer['5B'], 'numCells': num_SOM}
netParams.popParams['PV5B']  =  {'cellModel': 'HH_simple',         'cellType': 'PV', 'ynormRange': layer['5B'], 'numCells': num_PV}
#netParams.popParams['rnd']  =  {'cellModel': 'Randomizer', 'numCells': 1}


if cfg.singleCellPops:
    for pop in netParams.popParams.values(): pop['numCells'] = 1

#------------------------------------------------------------------------------
## Long-range input populations (VecStims)

numCells_long = 20

''' spkTimes_4K = np.concatenate((np.arange(0, 20, 1), np.arange(cfg.t_HSP + 5*1000, cfg.t_HSP + 15*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D1train'], ts['D1train'] + 10*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D1train'] + 11*1000, ts['D1train'] + 21*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D1train'] + 22*1000, ts['D1train'] + 32*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D1test'] + 5*1000, ts['D1test'] + 15*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D3test'] + 5*1000, ts['D3test'] + 15*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4extinct'], ts['D4extinct'] + 40*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4extinct'] + 41*1000, ts['D4extinct'] + 81*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4extinct'] + 82*1000, ts['D4extinct'] + 122*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4extinct'] + 123*1000, ts['D4extinct'] + 163*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4extinct'] + 164*1000, ts['D4extinct'] + 204*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D4test'] + 5*1000, ts['D4test'] + 15*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D5extinct'], ts['D5extinct'] + 40*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D5extinct'] + 41*1000, ts['D5extinct'] + 81*1000, 1/rate['4K']*1000),
                np.arange(ts['D5extinct'] + 123*1000, ts['D5extinct'] + 163*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D5extinct'] + 164*1000, ts['D5extinct'] + 204*1000, 1/rate['4K']*1000) ,
                np.arange(ts['D5test'] + 5*1000, ts['D5test'] + 15*1000, 1/rate['4K']*1000)))
spkTimes_12K = np.concatenate((np.arange(cfg.t_HSP + 35*1000, cfg.t_HSP + 45*1000, 1/rate['12K']*1000),
            np.arange(ts['D1test'] + 35*1000, ts['D1test'] + 45*1000, 1/rate['12K']*1000) ,
            np.arange(ts['D3test'] + 35*1000, ts['D3test'] + 45*1000, 1/rate['12K']*1000),
            np.arange(ts['D4test'] + 35*1000, ts['D4test'] + 45*1000, 1/rate['12K']*1000),
            np.arange(ts['D5test'] + 35*1000, ts['D5test'] + 45*1000, 1/rate['12K']*1000)))
 spkTimes_shock = np.concatenate((np.arange(ts['D1train'] + 8*1000, ts['D1train'] + 10*1000, 1/rate['shock']*1000),
                    np.arange(ts['D1train'] + 19*1000, ts['D1train'] + 21*1000, 1/rate['shock']*1000),
                    np.arange(ts['D1train'] + 30*1000, ts['D1train'] + 32*1000, 1/rate['shock']*1000)))
    '''

ts = cfg.tstart
if cfg.addLongConn:
    # VecStims
    longPops = ['S1_4K','S1_12K', 'OC']
    # create list of pulses (each item is a dict with pulse params) #noise(0 = deterministic; 1 = completely random)
    rate = {'4K': 50, '12K': 50, 'shock': 100} #Hz
    hspRate = 5
    pulses_4K = [{'start': ts['HSPset0'], 'end': ts['HSPset0']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset1'], 'end': ts['HSPset1']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP0'], 'end': ts['HSP0']+cfg.t_HSP, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset2'], 'end': ts['HSPset2']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP1'], 'end': ts['HSP1']+cfg.t_HSP , 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['Baseline'] + 5*1000, 'end': ts['Baseline'] + 15*1000, 'rate': rate['4K'], 'noise': 0}, #Baseline
                 {'start': ts['D1train'], 'end': ts['D1train'] + 10*1000, 'rate': rate['4K'], 'noise': 0}, #D1 FC
                 {'start': ts['D1train'] + 11*1000, 'end': ts['D1train'] + 21*1000, 'rate': rate['4K'], 'noise': 0}, #D1 FC
                 {'start': ts['D1train'] + 22*1000, 'end': ts['D1train'] + 32*1000, 'rate': rate['4K'], 'noise': 0}, #D1 FC
                 {'start': ts['D1test'] + 5*1000, 'end': ts['D1test'] + 15*1000, 'rate': rate['4K'], 'noise': 0}, #D1 test
                 {'start': ts['D3test'] + 5*1000, 'end': ts['D3test'] + 15*1000, 'rate': rate['4K'], 'noise': 0}, #D3 test
                 {'start': ts['D4extinct'], 'end': ts['D4extinct'] + 40*1000, 'rate': rate['4K'], 'noise': 0}, #D4 FC
                 {'start': ts['D4extinct'] + 41*1000, 'end': ts['D4extinct'] + 81*1000, 'rate': rate['4K'], 'noise': 0}, #D4 FC
                 {'start': ts['D4extinct'] + 82*1000, 'end': ts['D4extinct'] + 122*1000, 'rate': rate['4K'], 'noise': 0}, #D4 FC
                 {'start': ts['D4extinct'] + 123*1000, 'end': ts['D4extinct'] + 163*1000, 'rate': rate['4K'], 'noise': 0}, #D4 FC
                 {'start': ts['D4extinct'] + 164*1000, 'end': ts['D4extinct'] + 204*1000, 'rate': rate['4K'], 'noise': 0}, #D4 FC
                 {'start': ts['D4test'] + 5*1000, 'end': ts['D4test'] + 15*1000, 'rate': rate['4K'], 'noise': 0}, #D4 test
                 {'start': ts['D5extinct'], 'end': ts['D5extinct'] + 40*1000, 'rate': rate['4K'], 'noise': 0}, #D5 FC
                 {'start': ts['D5extinct'] + 41*1000, 'end': ts['D5extinct'] + 81*1000, 'rate': rate['4K'], 'noise': 0}, #D5 FC
                 {'start': ts['D5extinct'] + 82*1000, 'end': ts['D5extinct'] + 122*1000, 'rate': rate['4K'], 'noise': 0}, #D5 FC
                 {'start': ts['D5extinct'] + 123*1000, 'end': ts['D5extinct'] + 163*1000, 'rate': rate['4K'], 'noise': 0}, #D5 FC
                 {'start': ts['D5extinct'] + 164*1000, 'end': ts['D5extinct'] + 204*1000, 'rate': rate['4K'], 'noise': 0}, #D5 FC
                 {'start': ts['D5test'] + 5*1000, 'end': ts['D5test'] + 15*1000, 'rate': rate['4K'], 'noise': 0} #D5 test
                 ]
   
    pulses_12K = [
                 {'start': ts['HSPset0'], 'end': ts['HSPset0']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset1'], 'end': ts['HSPset1']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP0'], 'end': ts['HSP0']+cfg.t_HSP, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset2'], 'end': ts['HSPset2']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP1'], 'end': ts['HSP1']+cfg.t_HSP , 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['Baseline'] + 35*1000, 'end': ts['Baseline'] + 45*1000, 'rate': rate['12K'], 'noise': 0}, #Baseline
                  {'start': ts['D1test'] + 35*1000, 'end': ts['D1test'] + 45*1000, 'rate': rate['12K'], 'noise': 0}, #D1 test
                  {'start': ts['D3test'] + 35*1000, 'end': ts['D3test'] + 45*1000, 'rate': rate['12K'], 'noise': 0}, #D3 test
                  {'start': ts['D4test'] + 35*1000, 'end': ts['D4test'] + 45*1000, 'rate': rate['12K'], 'noise': 0}, #D4 test
                  {'start': ts['D5test'] + 35*1000, 'end': ts['D5test'] + 45*1000, 'rate': rate['12K'], 'noise': 0} #D5 test
                ]  
    
    
    pulses_shock = [
                 {'start': ts['HSPset0'], 'end': ts['HSPset0']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset1'], 'end': ts['HSPset1']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP0'], 'end': ts['HSP0']+cfg.t_HSP, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSPset2'], 'end': ts['HSPset2']+cfg.t_hspvtrg, 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['HSP1'], 'end': ts['HSP1']+cfg.t_HSP , 'rate': hspRate, 'noise': 0.25}, #HSP
                 {'start': ts['D1train'] + 8*1000, 'end': ts['D1train'] + 10*1000, 'rate': rate['shock'], 'noise': 0}, #D1 FC
                    {'start': ts['D1train'] + 19*1000, 'end': ts['D1train'] + 21*1000, 'rate': rate['shock'], 'noise': 0}, #D1 FC
                    {'start': ts['D1train'] + 30*1000, 'end': ts['D1train'] + 32*1000, 'rate': rate['shock'], 'noise': 0} #D1 FC
                    ]

    netParams.popParams['S1_4K'] = {'cellModel': 'VecStim', 'numCells': numCells_long,
                                     'ynormRange': layer['longS1_4K'], 'spkTimes': [cfg.duration], 'pulses': pulses_4K}                                                  
    

    netParams.popParams['S1_12K'] = {'cellModel': 'VecStim', 'numCells': numCells_long, 
                                     'ynormRange': layer['longS1_12K'], 'spkTimes': [cfg.duration], 'pulses': pulses_12K}

    netParams.popParams['OC'] = {'cellModel': 'VecStim', 'numCells': numCells_long,
                                'ynormRange': layer['longOC'], 'spkTimes': [cfg.duration], 'pulses': pulses_shock}

    '''netParams.popParams['HSP1'] = {'cellModel': 'VecStim', 'numCells': 1,
                                   'spikePattern': {'type': 'poisson', 'start': 0, 'stop': cfg.t_hspvtrg, 'frequency': 10}}
    netParams.popParams['HSP2'] = {'cellModel': 'VecStim', 'numCells': 1,
                                   'spikePattern': {'type': 'poisson', 'start': ts['D1test'] + cfg.t_test +cfg.t_cool-20, 'stop': ts['D3test'], 'frequency': 10}}
    netParams.popParams['HSP3'] = {'cellModel': 'VecStim', 'numCells': 1,
                                   'spikePattern': {'type': 'poisson', 'start': ts['D4test'] + cfg.t_test +cfg.t_cool-20, 'stop': ts['D5extinct'], 'frequency': 10}}
 '''
                                

    #netParams.popParams['HSP'] = {'cellModel': 'VecStim', 'numCells': 1,
    #                            'spkTimes': [ts['D1test'] + cfg.t_test +cfg.t_cool], 'pulses': pulses_HSP}

#------------------------------------------------------------------------------
# Synaptic mechanism parameters
#------------------------------------------------------------------------------
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod':'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod':'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod':'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB','tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}





ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow','GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']



#------------------------------------------------------------------------------
# Local connectivity parameters
#------------------------------------------------------------------------------

# Store the number of segments in a dictionary
nseg = {'PV': {key: math.floor(value) for key, value in secL_PV.items()},
        'SOM': {key: math.floor(value) for key, value in secL_SOM.items()}, 
        'PT': {key: math.floor(value) for key, value in secL_PT.items()}} #highest possible spine density = 1/um


# plasticity parameters
STDPparams = {'hebbwt': .5, 'antiwt':-.5, 'wmax': 15, 'RLon': 0 , 'RLhebbwt': 0.1, 'RLantiwt': -0.100, \
    'tauhebb': 10, 'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 1, 'verbose':0}

randomlistgen = RandomSequenceNetParams()
randomlistgen.initializeRandomSequence(cfg)

with open(f'data/randomlistgen{cfg.rdseed}.txt', 'w') as f:
    f.write(json.dumps(randomlistgen.randomlist, indent=4))
'''cell_types = ['PT', 'PV', 'SOM']
total_synapses_all_cells = 0

for celltype in cell_types:
    synapse_counts = randomlistgen._calculate_total_synapses(celltype)
    axosomatic_synapses = synapse_counts['axosomatic']
    axodendritic_synapses = sum(synapse_counts['axodendritic'].values())

    total_synapses = axosomatic_synapses + axodendritic_synapses
    total_synapses_all_cells += total_synapses

print(total_synapses_all_cells)'''
#------------------------------------------------------------------------------
## E -> I
if cfg.EIGain: # Use IEGain if value set
    cfg.EPVGain = cfg.EIGain
    cfg.ESOMGain = cfg.EIGain
else: 
    cfg.EIGain = (cfg.EPVGain+cfg.ESOMGain)/2.0

if cfg.addConn and (cfg.EPVGain > 0.0 or cfg.ESOMGain > 0.0):
    preTypes = ['PT']
    postTypes = ['PV', 'SOM']
    ESynMech = ['AMPA','NMDA']
    lGain = [cfg.EPVGain, cfg.ESOMGain] # E -> PV or E -> SOM
    for ipost, postType in enumerate(postTypes):
        weights = [
            randomlistgen.randomlist['EI'][postType][j][i][0][1]
            for i in range(num_PT)
            for j in range(num_dict[postType])
        ]
        weights_syns = [[x, x] for x in weights]
        locs = [
            randomlistgen.randomlist['EI'][postType][j][i][0][0]
            for i in range(num_PT)
            for j in range(num_dict[postType])
        ]
        locs_syns = [[x, x] for x in locs]
        ruleLabel = 'EI'+'_'+ postType
        netParams.connParams[ruleLabel] = {
            'preConds': {'cellType': preTypes},#, 'ynorm': list(preBin)},
            'postConds': {'cellType': postType},#, 'ynorm': list(postBin)},
            'synMech': ESynMech,
            'connList': [(i, j) for i in range(num_PT) for j in range(num_dict[postType])],
            'weight': weights_syns,#lambda pre, post: randomlistgen.randomlist['EI'][postType][post.index][pre.index][0][1],
            'loc': locs_syns, #lambda pre, post: randomlistgen.randomlist['EI'][postType][post.index][pre.index][0][0],
            #'delay': 'defaultDelay+dist_3D/propVelocity',
            'sec': 'soma',
            'synsPerConn': 1,
            'plast': {'mech': 'STDP', 'params': STDPparams}} # simple I cells used right now only have soma
            #debug: line 695 in compartCell.py, conn['plast'] -> conn


#------------------------------------------------------------------------------
## I -> all
if cfg.addConn:
    preCellTypes = ['SOM', 'PV']
    ynorms = [[0,1]]*2 # <----not local, interneuron can connect to all layers
    postCellTypes = ['PT', 'PV', 'SOM']
    disynapticBias = None  # default, used for I->I

    for i,(preCellType, ynorm) in enumerate(zip(preCellTypes, ynorms)):
        for ipost, postCellType in enumerate(postCellTypes):
            if postCellType == 'PV':    # postsynaptic I cell
                sec = 'soma'
                synWeightFraction = [1]
                if preCellType == 'PV':             # PV->PV
                    #  weight = IIweight * cfg.PVPVGain
                    synMech = PVSynMech
                else:                           # SOM->PV
                    #  weight = IIweight * cfg.SOMPVGain
                    synMech = SOMISynMech
            elif postCellType == 'SOM': # postsynaptic I cell
                sec = 'soma'
                synWeightFraction = [1]
                if preCellType == 'PV':             # PV->SOM
                    #  weight = IIweight * cfg.PVSOMGain
                    synMech = PVSynMech
                else:                           # SOM->SOM
                    #  weight = IIweight * cfg.SOMSOMGain
                    synMech = SOMISynMech
            elif postCellType == 'PT': # postsynaptic PT cell
                #disynapticBias = IEdisynBias
                if preCellType == 'PV':             # PV->E
                    #weight = IEweight * cfg.IPTGain * cfg.PVEGain
                    synMech = PVSynMech
                    sec = 'perisom'
                else:                           # SOM->E
                    #weight = IEweight * cfg.IPTGain * cfg.SOMEGain
                    synMech = SOMESynMech
                    sec = 'spiny'
                    synWeightFraction = cfg.synWeightFractionSOME

            if postCellType == 'PT':
                numCell = num_PT
            else:
                numCell = num_dict[postCellType]
            if postCellType == 'PT':
                if sec in ['soma', 'perisom']:
                    sec = 'soma'
                    weights = [
                        randomlistgen.randomlist[preCellType+'E'][j][i][sec][0][1]
                        for i in range(num_dict[preCellType])
                        for j in range(num_PT)
                    ]
                    locs = [
                        randomlistgen.randomlist[preCellType+'E'][j][i][sec][0][0]
                        for i in range(num_dict[preCellType])
                        for j in range(num_PT)
                    ]

                    ruleLabel = preCellType +'_'+ postCellType
                    netParams.connParams[ruleLabel] = {
                        'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                        'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                        'synMech': synMech,
                        'connList': [(i, j) for i in range(num_dict[preCellType]) for j in range(numCell)],
                        'weight': weights, #lambda pre, post: randomlistgen.randomlist[preCellType + 'E'][postCellType][post.index][pre.index][sec][0][1],
                        'loc': locs, #lambda pre, post: randomlistgen.randomlist[preCellType+'E'][postCellType][post.index][pre.index][sec][0][0],
                        #'delay': 'defaultDelay+dist_3D/propVelocity',
                        'synsPerConn': 1,
                        'sec': sec,
                        #'disynapticBias': disynapticBias,
                        'plast': {'mech': 'STDP', 'params': STDPparams}}
                else:
                    
                    for s in seclist['PT']['dend']:
                        weights = [
                            randomlistgen.randomlist[preCellType+'E'][j][i][s][0][1]
                            for i in range(num_dict[preCellType])
                            for j in range(num_PT)
                        ]
                        locs = [
                            randomlistgen.randomlist[preCellType+'E'][j][i][s][0][0]
                            for i in range(num_dict[preCellType])
                            for j in range(num_PT)
                        ]
                        ruleLabel = preCellType +'_'+ postCellType + '_' + s
                        netParams.connParams[ruleLabel] = {
                            'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                            'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                            'synMech': synMech,
                            'connList': [(i, j) for i in range(num_dict[preCellType]) for j in range(numCell)],
                            'weight': weights, #lambda pre, post: randomlistgen.randomlist[preCellType + 'E'][postCellType][post.index][pre.index][s][0][1],
                            'loc': locs, #lambda pre, post: randomlistgen.randomlist[preCellType + 'E'][postCellType][post.index][pre.index][s][0][0],
                            #'delay': 'defaultDelay+dist_3D/propVelocity',
                            'synsPerConn': 1,
                            'sec': s,
                            #'disynapticBias': disynapticBias,
                            'plast': {'mech': 'STDP', 'params': STDPparams}}
            else:
                if preCellType == postCellType:
                    weights = [
                        randomlistgen.randomlist['II'][postCellType][preCellType][j][i][0][1]
                        for i in range(num_dict[preCellType]-1)
                        for j in range(num_dict[postCellType])
                    ]
                    locs = [
                        randomlistgen.randomlist['II'][postCellType][preCellType][j][i][0][0]
                        for i in range(num_dict[preCellType]-1)
                        for j in range(num_dict[postCellType])
                    ]
                else:
                    weights = [
                        randomlistgen.randomlist['II'][postCellType][preCellType][j][i][0][1]
                        for i in range(num_dict[preCellType])
                        for j in range(num_dict[postCellType])
                    ]
                    locs = [
                        randomlistgen.randomlist['II'][postCellType][preCellType][j][i][0][0]
                        for i in range(num_dict[preCellType])
                        for j in range(num_dict[postCellType])
                    ]
                ruleLabel = preCellType + '_' + postCellType
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                    'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                    'synMech': synMech,
                    'connList': [(i, j) for i in range(num_dict[preCellType]-1) for j in range(numCell)],
                    'weight': weights, #lambda pre, post: randomlistgen.randomlist[preCellType + 'E'][postCellType][post.index][pre.index][sec][0][1],
                    'loc': locs, #lambda pre, post: randomlistgen.randomlist[preCellType+'E'][postCellType][post.index][pre.index][sec][0][0],
                    #'delay': 'defaultDelay+dist_3D/propVelocity',
                    'synsPerConn': 1,
                    'sec': sec,
                    #'disynapticBias': disynapticBias,
                    'plast': {'mech': 'STDP', 'params': STDPparams}}



#------------------------------------------------------------------------------
# Long-range connectivity parameters
#------------------------------------------------------------------------------
if cfg.addLongConn:

    longPops = ['S1_4K','S1_12K', 'OC']
    cellTypes = ['PT']
    for longPop in longPops:
        for secName in seclist['PT']['dend']:
            weights = [
                [syn[1] for syn in randomlistgen.randomlist['LongE'][longPop][secName][j][i]]
                for i in range(numCells_long)
                for j in range(num_PT)
            ]
            weights_syns = [[x, x] for x in weights]
            locs = [
                [syn[0] for syn in randomlistgen.randomlist['LongE'][longPop][secName][j][i]]
                for i in range(numCells_long)
                for j in range(num_PT)
            ]
            locs_syns = [[x, x] for x in locs]
            ruleLabel = longPop+'_'+secName
            netParams.connParams[ruleLabel] = { 
                'preConds': {'pop': longPop}, 
                'postConds': {'cellType': 'PT'},
                'synMech': ESynMech,
                'connList': [(i, j) for i in range(numCells_long) for j in range(num_PT)],
                'weight': weights_syns,
                'loc': locs_syns, 
                'synsPerConn': math.ceil(nseg['PT'][secName]/numCells_long),
                'delay': 'defaultDelay+dist_3D/propVelocity',
                'sec': secName,
                'plast': {'mech': 'STDP', 'params': STDPparams}} 


#------------------------------------------------------------------------------
# Description
#------------------------------------------------------------------------------
netParams.description = """ 
- M1 net, 6 layers, 7 cell types 
- NCD-based connectivity from  Weiler et al. 2008; Anderson et al. 2010; Kiritani et al. 2012; 
  Yamawaki & Shepherd 2015; Apicella et al. 2012
- Parametrized version based on Sam's code
- Updated cell models and mod files
- Added parametrized current inputs
- Fixed bug: prev was using cell models in /usr/site/nrniv/local/python/ instead of cells 
- Use 5 synsperconn for 5-comp cells (HH_reduced); and 1 for 1-comp cells (HH_simple)
- Fixed bug: made global h params separate for each cell model
- Fixed v_init for different cell models
- New IT cell with same geom as PT
- Cleaned cfg and moved background inputs here
- Set EIGain and IEGain for each inh cell type
- Added secLists for PT full
- Fixed reduced CT (wrong vinit and file)
- Added subcellular conn rules to distribute synapses
- PT full model soma centered at 0,0,0 
- Set cfg seeds here to ensure they get updated
- Added PVSOMGain and SOMPVGain
- PT subcellular distribution as a cfg param
- Cylindrical volume
- DefaultDelay (for local conns) = 2ms
- Added long range connections based on Yamawaki 2015a,b; Suter 2015; Hooks 2013; Meyer 2011
- Updated cell densities based on Tsai 2009; Lefort 2009; Katz 2011; Wall 2016; 
- Separated PV and SOM of L5A vs L5B
- Fixed bugs in local conn (PT, PV5, SOM5, L6)
- Added perisom secList including all sections 50um from soma
- Added subcellular conn rules (for both full and reduced models)
- Improved cell models, including PV and SOM fI curves
- Improved subcell conn rules based on data from Suter15, Hooks13 and others
- Adapted Bdend L of reduced cell models
- Made long pop rates a cfg param
- Set threshold to 0.0 mV
- Parametrized I->E/I layer weights
- Added missing subconn rules (IT6->PT; S1,S2,cM1->IT/CT; long->SOM/PV)
- Added threshold to weightNorm (PT threshold=10x)
- weightNorm threshold as a cfg parameter
- Separate PV->SOM, SOM->PV, SOM->SOM, PV->PV gains 
- Conn changes: reduced IT2->IT4, IT5B->CT6, IT5B,6->IT2,4,5A, IT2,4,5A,6->IT5B; increased CT->PV6+SOM6
- Parametrized PT ih gbar
- Added IFullGain parameter: I->E gain for full detailed cell models
- Replace PT ih with Migliore 2012
- Parametrized ihGbar, ihGbarBasal, dendNa, axonNa, axonRa, removeNa
- Replaced cfg list params with dicts
- Parametrized ihLkcBasal and AMPATau2Factor
- Fixed synMechWeightFactor
- Parametrized PT ih slope
- Added disynapticBias to I->E (Yamawaki&Shepherd,2015)
- Fixed E->CT bin 0.9-1.0
- Replaced GABAB with exp2syn and adapted synMech ratios
- Parametrized somaNa
- Added ynorm condition to NetStims
- Added option to play back recorded spikes into long-range inputs
- Fixed Bdend pt3d y location
- Added netParams.convertCellShapes = True to convert stylized geoms to 3d points
- New layer boundaries, cell densities, conn, FS+SOM L4 grouped with L2/3, low cortical input to L4
- Increased exc->L4 based on Yamawaki 2015 fig 5
- v54: Moved from NetPyNE v0.7.9 to v0.9.1 (v54_batch1-6)
- v54: Moved to NetPyNE v0.9.1 and py3 (v54_batch7 onwards)
- v56: Reduced dt from 0.05 to 0.025 (note this version follows from v54, i.e. without new cell types; branch 'paper2019_py3')
- v56: (included in prev version): Added cfg.KgbarFactor
"""

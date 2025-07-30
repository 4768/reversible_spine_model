"""
exp1_netParams.py

Network parameters for Experiment 1: Spine elimination effect on D1 test.
This keeps the exact same structure as the original netParams.py to ensure
that the only difference is the weights of eliminated spines.

Based on realsim_new/netParams.py
"""

from netpyne import specs, sim
from neuron import h
import pickle, json
import random
import numpy as np
import math
import os
import ast  
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size = comm.Get_size()

netParams = specs.NetParams()   # object of class NetParams to store the network parameters

netParams.version = 56

try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from exp1_cfg_D1 import cfg

num_PT = 20
num_PV = 10
num_SOM = 10
num_dict = {'PT': num_PT, 'PV': num_PV, 'SOM': num_SOM}
seclist = {'PT': {'dend':[], 'all':[]}, 'PV': [], 'SOM': []} #'all' except axon

#------------------------------------------------------------------------------
# General network parameters
#------------------------------------------------------------------------------
netParams.scale = cfg.scale 
netParams.sizeX = cfg.sizeX 
netParams.sizeY = cfg.sizeY 
netParams.sizeZ = cfg.sizeZ 
netParams.shape = 'cylinder'

#------------------------------------------------------------------------------
# General connectivity parameters
#------------------------------------------------------------------------------
netParams.scaleConnWeight = 1.0
netParams.defaultThreshold = -5.0
netParams.defaultDelay = 2.0
netParams.propVelocity = 500.0
netParams.probLambda = 100.0
netParams.defineCellShapes = True

#------------------------------------------------------------------------------
# Cell parameters
#------------------------------------------------------------------------------
cellModels = ['HH_reduced']
layer = {'1':[0.0, 0.1],'5B': [0.47,0.8],'longS1_4K': [2.2,2.3],'longS1_12K': [2.2,2.3], 'longOC': [2.2,2.3]}

netParams.correctBorder = {
    'threshold': [cfg.correctBorderThreshold, cfg.correctBorderThreshold, cfg.correctBorderThreshold], 
    'yborders': [layer['1'][0], layer['5B'][0], layer['5B'][1]]
}

#------------------------------------------------------------------------------
## Load cell rules from template files
#------------------------------------------------------------------------------
cellParamLabels = ['PV_simple', 'SOM_simple', 'PT5B_reduced']
loadCellParams = cellParamLabels
saveCellParams = False 

for ruleLabel in loadCellParams:
    netParams.loadCellParamsRule(label=ruleLabel, fileName='cells/'+ruleLabel+'_cellParams.pkl')
    
    if ruleLabel in ['PT5B_reduced']:
        cellRule = netParams.cellParams[ruleLabel]
        seclist['PT']['dend'] = cellRule['secLists']['alldend']
        seclist['PT']['all'] = [sec for sec in cellRule['secs'] if sec not in ['axon']]
        secL_PT = {k: 0.0 for k in cellRule['secs']}
        for secName in cellRule['secs']:
            secL_PT[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None
    elif ruleLabel in ['PV_simple']:
        cellRule = netParams.cellParams[ruleLabel]
        secL_PV = {'soma': 0.0, 'dend': 0.0, 'axon': 0.0}
        for secName in cellRule['secs']:
            secL_PV[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None
    elif ruleLabel in ['SOM_simple']:
        cellRule = netParams.cellParams[ruleLabel]
        secL_SOM = {'soma': 0.0, 'dend': 0.0, 'axon': 0.0}
        for secName in cellRule['secs']:
            secL_SOM[secName] = cellRule['secs'][secName]['geom']['L']
            cellRule['secs'][secName]['geom']['nseg'] = math.floor(cellRule['secs'][secName]['geom']['L'])
            cellRule['secs'][secName]['weightNorm'] = None

#------------------------------------------------------------------------------
## Population parameters
#------------------------------------------------------------------------------
# Store the number of segments in a dictionary
nseg = {'PV': {key: math.floor(value) for key, value in secL_PV.items()},
        'SOM': {key: math.floor(value) for key, value in secL_SOM.items()}, 
        'PT': {key: math.floor(value) for key, value in secL_PT.items()}}

# Define local populations
netParams.popParams['PT5B_reduced'] = {'cellModel': 'HH_reduced', 'cellType': 'PT', 'ynormRange': layer['5B'], 'numCells': num_PT}
netParams.popParams['SOM5B'] = {'cellModel': 'HH_simple', 'cellType': 'SOM', 'ynormRange': layer['5B'], 'numCells': num_SOM}
netParams.popParams['PV5B'] = {'cellModel': 'HH_simple', 'cellType': 'PV', 'ynormRange': layer['5B'], 'numCells': num_PV}
numCells_long = 20


#------------------------------------------------------------------------------
# Synaptic mechanism parameters - Keep identical to original netParams
#------------------------------------------------------------------------------
netParams.synMechParams['NMDA'] = {'mod': 'MyExp2SynNMDABB', 'tau1NMDA': 15, 'tau2NMDA': 150, 'e': 0}
netParams.synMechParams['AMPA'] = {'mod': 'MyExp2SynBB', 'tau1': 0.05, 'tau2': 5.3*cfg.AMPATau2Factor, 'e': 0}
netParams.synMechParams['GABAB'] = {'mod': 'MyExp2SynBB', 'tau1': 3.5, 'tau2': 260.9, 'e': -93} 
netParams.synMechParams['GABAA'] = {'mod': 'MyExp2SynBB', 'tau1': 0.07, 'tau2': 18.2, 'e': -80}
netParams.synMechParams['GABAASlow'] = {'mod': 'MyExp2SynBB', 'tau1': 2, 'tau2': 100, 'e': -80}
netParams.synMechParams['GABAASlowSlow'] = {'mod': 'MyExp2SynBB', 'tau1': 200, 'tau2': 400, 'e': -80}

# Define synapse mechanism combinations
ESynMech = ['AMPA', 'NMDA']
SOMESynMech = ['GABAASlow', 'GABAB']
SOMISynMech = ['GABAASlow']
PVSynMech = ['GABAA']


# Store the number of segments in a dictionary
nseg = {'PV': {key: math.floor(value) for key, value in secL_PV.items()},
        'SOM': {key: math.floor(value) for key, value in secL_SOM.items()}, 
        'PT': {key: math.floor(value) for key, value in secL_PT.items()}} #highest possible spine density = 1/um


# STDP parameters - needed for applying loaded weights
STDPparams = {
    'hebbwt': .5, 'antiwt': -.5, 'wmax': 15, 'RLon': 0, 
    'RLhebbwt': 0.1, 'RLantiwt': -0.100, 'tauhebb': 10,
    'RLwindhebb': 50, 'useRLexp': 0, 'softthresh': 1, 'verbose': 0
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
seed = 255
# Use configurable data directory
data_dir = os.environ.get('SIM_DATA_DIR', os.path.join(BASE_DIR, 'data'))
randomlist_path = os.path.join(data_dir, f"seed{seed}", f"randomlistgen{seed}.txt")
with open(randomlist_path, 'r') as f:
    content = f.read()
    randomlist = ast.literal_eval(content)

#------------------------------------------------------------------------------
# Connectivity - same pattern as original but simplified for Exp1
#------------------------------------------------------------------------------

num_PT = 20
num_PV = 10
num_SOM = 10
num_dict = {'PT': num_PT, 'PV': num_PV, 'SOM': num_SOM}


#------------------------------------------------------------------------------
## E -> I

preTypes = ['PT']
postTypes = ['PV', 'SOM']
ESynMech = ['AMPA','NMDA']
for ipost, postType in enumerate(postTypes):
    weights = [0]*num_PT*num_dict[postType]
    weights_syns = [[x, x] for x in weights]
    locs = [
            randomlist['EI'][postType][j][i][0][0]
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
        'weight': weights_syns,
        'loc': locs_syns,
        'sec': 'soma',
        'synsPerConn': 1,
        'plast': {'mech': 'STDP', 'params': STDPparams}}


#------------------------------------------------------------------------------
## I -> all
if cfg.addConn:
    preCellTypes = ['SOM', 'PV']
    ynorms = [[0,1]]*2 # <----not local, interneuron can connect to all layers
    postCellTypes = ['PT', 'PV', 'SOM']

    for i,(preCellType, ynorm) in enumerate(zip(preCellTypes, ynorms)):
        for ipost, postCellType in enumerate(postCellTypes):
            if postCellType == 'PV':    # postsynaptic I cell
                sec = 'soma'
                if preCellType == 'PV':             # PV->PV
                    synMech = PVSynMech
                else:                           # SOM->PV
                    synMech = SOMISynMech
            elif postCellType == 'SOM': # postsynaptic I cell
                sec = 'soma'
                if preCellType == 'PV':             # PV->SOM
                    synMech = PVSynMech
                else:                           # SOM->SOM
                    synMech = SOMISynMech
            elif postCellType == 'PT': # postsynaptic PT cell
                if preCellType == 'PV':             # PV->E
                    synMech = PVSynMech
                    sec = 'perisom'
                else:                           # SOM->E
                    synMech = SOMESynMech
                    sec = 'spiny'

            if postCellType == 'PT':
                numCell = num_PT
            else:
                numCell = num_dict[postCellType]
            if postCellType == 'PT':
                if sec in ['soma', 'perisom']:
                    sec = 'soma'
                    weights = [0]*num_PT*num_dict[preCellType]
                    locs = [
                        randomlist[preCellType+'E'][j][i][sec][0][0]
                        for i in range(num_dict[preCellType])
                        for j in range(num_PT)
                    ]


                    ruleLabel = preCellType +'_'+ postCellType
                    netParams.connParams[ruleLabel] = {
                        'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                        'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                        'synMech': synMech,
                        'connList': [(i, j) for i in range(num_dict[preCellType]) for j in range(numCell)],
                        'weight': weights, 
                        'loc': locs, 
                        'synsPerConn': 1,
                        'sec': sec,
                        'plast': {'mech': 'STDP', 'params': STDPparams}}
                else:
                    
                    for s in seclist['PT']['dend']:
                        weights = [0]*num_PT*num_dict[preCellType]
                        locs = [
                            randomlist[preCellType+'E'][j][i][s][0][0]
                            for i in range(num_dict[preCellType])
                            for j in range(num_PT)
                        ]
                        ruleLabel = preCellType +'_'+ postCellType + '_' + s
                        netParams.connParams[ruleLabel] = {
                            'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                            'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                            'synMech': synMech,
                            'connList': [(i, j) for i in range(num_dict[preCellType]) for j in range(numCell)],
                            'weight': weights, 
                            'loc': locs, 
                            'synsPerConn': 1,
                            'sec': s,
                            'plast': {'mech': 'STDP', 'params': STDPparams}}
            else:
                if preCellType == postCellType:
                    weights = [0]*(num_dict[preCellType]-1)*num_dict[postCellType]
                    locs = [
                        randomlist['II'][postCellType][preCellType][j][i][0][0]
                        for i in range(num_dict[preCellType]-1)
                        for j in range(num_dict[postCellType])
                    ]
                else:
                    weights = [0]*num_dict[preCellType]*num_dict[postCellType]
                    locs = [
                        randomlist['II'][postCellType][preCellType][j][i][0][0]
                        for i in range(num_dict[preCellType])
                        for j in range(num_dict[postCellType])
                    ]
                ruleLabel = preCellType + '_' + postCellType
                netParams.connParams[ruleLabel] = {
                    'preConds': {'cellType': preCellType, 'ynorm': ynorm},
                    'postConds': {'cellType': postCellType, 'ynorm': ynorm},
                    'synMech': synMech,
                    'connList': [(i, j) for i in range(num_dict[preCellType]-1) for j in range(numCell)],
                    'weight': weights, 
                    'loc': locs, 
                    'synsPerConn': 1,
                    'sec': sec,
                    'plast': {'mech': 'STDP', 'params': STDPparams}}

#setup_stimulations() :TODO


print(f"Experiment 1 Network Parameters Loaded (preserving original structure).") 
"""
cfg.py 

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import pickle
from neuron import h
import os
cfg = specs.SimConfig() 


# ------------------------------------------------------------------------------
# Experiment 2 Specific Parameters
# ------------------------------------------------------------------------------
cfg.experiment = 'exp2'

# Spine elimination parameters
cfg.spineEliminationThreshold = 3

# Data saving folder structure
cfg.baseSaveFolder = f'data/{cfg.experiment}' # Base directory for Exp2 results

# ------------------------------------------------------------------------------
# Timing Parameters (Focus on D3 Test Phase)
# ------------------------------------------------------------------------------
# Durations from base cfg.py needed for reference (adjust if necessary)
cfg.t_test = (15 + 15 + 15) * 1000 # Standard test duration: 15s stim + 15s buffer + 15s stim
cfg.dt = 0.01 # Simulation time step

# D3 Test phase timing (relative to the start of this specific simulation)
cfg.d3test_start_time = 0 # Simulation starts directly with D3 test

# Define specific timing for each stimulus within the D3 test
# 4K stimulation (conditioning stimulus)
cfg.d3test_4k_record_start = 0  # Starting recording immediately
cfg.d3test_4k_stim_start = 5000  # Start stimulus after 5 seconds
cfg.d3test_4k_stim_end = cfg.d3test_4k_stim_start + 10000  # 10 seconds of stimulation

# 12K stimulation (follows 4K)
cfg.d3test_12k_record_start = cfg.d3test_4k_stim_end + 15000
cfg.d3test_12k_stim_start = cfg.d3test_12k_record_start + 5000  # Start stimulus after 5 seconds
cfg.d3test_12k_stim_end = cfg.d3test_12k_stim_start + 10000  # 10 seconds of stimulation 

# Make sure duration covers both stimulation periods
cfg.duration = cfg.d3test_12k_stim_end

#------------------------------------------------------------------------------
#
# SIMULATION CONFIGURATION
#
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Run parameters
#------------------------------------------------------------------------------
cfg.dt = 0.01
cfg.seeds = {'conn': 4321, 'stim': 1234, 'loc': 4321, 'mech': 2341} 
cfg.rdseed = 255
cfg.hParams = {'celsius': 34, 'v_init': -80}  
cfg.verbose = 0
cfg.loadBalancing = True
cfg.createNEURONObj = True
cfg.createPyStruct = 1
cfg.connRandomSecFromList = False  # set to false for reproducibility 
cfg.cvode_active = False
cfg.cvode_atol = 1e-6
cfg.cache_efficient = True
cfg.printRunTime = 0.1
cfg.oneSynPerNetcon = False  # only affects conns not in subconnParams; produces identical results

cfg.includeParamsLabel = True #True # needed for modify synMech False
cfg.printPopAvgRates = [0, cfg.duration]

cfg.checkErrors = False

cfg.saveInterval = 100 # define how often the data is saved, this can be used with interval run if you want to update the weights more often than you save
cfg.intervalFolder = 'interval_saving'


#------------------------------------------------------------------------------
# Recording 
#------------------------------------------------------------------------------
#allpops = ['PT5B','PV5B','SOM5B']
allpops = ['PT5B_reduced','PV5B','SOM5B']
cfg.recordCells = ['all']
count = 0

cfg.saveLFPPops =  False # allpops 

cfg.recordDipoles = False # {'L2': ['IT2'], 'L4': ['IT4'], 'L5': ['IT5A', 'IT5B', 'PT5B']}
#cfg.recordSpikesGids = ['PT5B_reduced']
cfg.recordStim = False
cfg.recordTime = False  
cfg.recordStep = 0.5 #0.025
cfg.timing = True

#------------------------------------------------------------------------------
# Saving
#------------------------------------------------------------------------------
cfg.simLabel = 'todd_test'
cfg.saveFolder = 'data/todd_test/seed1265'
cfg.savePickle = True
cfg.saveJson = True
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams']#, 'net']
cfg.backupCfgFile = None #['cfg.py', 'backupcfg/'] 
cfg.gatherOnlySimData = False
cfg.saveCellSecs = True
cfg.saveCellConns = True
cfg.compactConnFormat = 0


#------------------------------------------------------------------------------
# Cells
#------------------------------------------------------------------------------
cfg.cellmod =  {'PT5B': 'HH_full'}
#cfg.cellmod =  {'PT5B_reduced': 'HH_reduced'}

cfg.ihModel = 'migliore'  # ih model
cfg.ihGbar = 1.0  # multiplicative factor for ih gbar in PT cells
cfg.ihGbarZD = None # multiplicative factor for ih gbar in PT cells
cfg.ihGbarBasal = 1.0 # 0.1 # multiplicative factor for ih gbar in PT cells
cfg.ihlkc = 0.2 # ih leak param (used in Migliore)
cfg.ihlkcBasal = 1.0
cfg.ihlkcBelowSoma = 0.01
cfg.ihlke = -86  # ih leak param (used in Migliore)
cfg.ihSlope = 14*2

cfg.removeNa = False  # simulate TTX; set gnabar=0s
cfg.somaNa = 5
cfg.dendNa = 1 # do not change the NA level in dend
cfg.axonNa = 5
cfg.axonRa = 0.005

cfg.gpas = 0.5  # multiplicative factor for pas g in PT cells
cfg.epas = 0.9  # multiplicative factor for pas e in PT cells

cfg.KgbarFactor = 1.0 # multiplicative factor for K channels gbar in all E cells
cfg.makeKgbarFactorEqualToNewFactor = False

cfg.modifyMechs = {'startTime': 0, 'endTime': 1000, 'cellType':'PT', 'mech': 'hd', 'property': 'gbar', 'newFactor': 1.00, 'origFactor': 0.75}



#------------------------------------------------------------------------------
# Synapses
#------------------------------------------------------------------------------
cfg.synWeightFractionEE = [0.5, 0.5] # E->E AMPA to NMDA ratio
cfg.synWeightFractionEI = [0.5, 0.5] # E->I AMPA to NMDA ratio
cfg.synWeightFractionSOME = [0.9, 0.1] # SOM -> E GABAASlow to GABAB ratio

cfg.synsperconn = {'HH_full': 1, 'HH_reduced': 1, 'HH_simple': 1}
cfg.AMPATau2Factor = 1.0


#------------------------------------------------------------------------------
# Network 
#------------------------------------------------------------------------------
cfg.singleCellPops = 0  # Create pops with 1 single cell (to debug)
#cfg.weightNorm = 0  # use weight normalization
#cfg.weightNormThreshold = 4.0  # weight normalization factor threshold

cfg.addConn = 1
cfg.scale = 1.0
cfg.sizeY = 1350.0
cfg.sizeX = 300.0
cfg.sizeZ = 300.0
cfg.correctBorderThreshold = 150.0

cfg.L5BrecurrentFactor = 1.0
cfg.ITinterFactor = 1.0
cfg.strengthFactor = 1.0

cfg.EEGain = 1.0
cfg.EIGain = 1.0
cfg.IEGain = 1.0
cfg.IIGain = 1.0

cfg.IEdisynapticBias = None  # increase prob of I->Ey conns if Ex->I and Ex->Ey exist 

#------------------------------------------------------------------------------
## E->I gains
cfg.EPVGain = 1.0
cfg.ESOMGain = 1.0

#------------------------------------------------------------------------------
## I->E gains
cfg.PVEGain = 1.0
cfg.SOMEGain = 1.0

#------------------------------------------------------------------------------
## I->I gains
cfg.PVSOMGain = 1.0
cfg.SOMPVGain = 1.0
cfg.PVPVGain = 1.0
cfg.SOMSOMGain = 1.0

#------------------------------------------------------------------------------
## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = 1.0
cfg.IIweights = 1.0

cfg.IPTGain = 1.0
cfg.IFullGain = 1.0


#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = True


print(f"Experiment 2 Configuration Loaded.") 
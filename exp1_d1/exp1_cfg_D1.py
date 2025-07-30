"""
cfg.py 

Simulation configuration for M1 model (using NetPyNE)

Contributors: salvadordura@gmail.com
"""

from netpyne import specs
import pickle
from neuron import h
import os
import math
cfg = specs.SimConfig() 
# ------------------------------------------------------------------------------
# Experiment 1 Specific Parameters
# ------------------------------------------------------------------------------
cfg.experiment = 'exp1'

# Spine elimination parameters
cfg.spineEliminationThreshold = 3

# Data saving folder structure
cfg.baseSaveFolder = f'data/{cfg.experiment}' # Base directory for Exp1 results

cfg.max_concurrent_trials = 20  # Maximum number of trials that can run simultaneously

# Get base directory
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))

# Set base directories for output using relative paths
cfg.base_dir = current_dir

# Exp1-specific parameters
cfg.exp1 = {
    'targetPhase': 'D1test',
    'weightPeriod': 'After D1 train',  # Weight loading period for D1test
    'activityTimepoint': 'D1 test 4k'  # Activity data timepoint
}





#------------------------------------------------------------------------------
# TRIAL STRUCTURE - Following 6-second cycle
#------------------------------------------------------------------------------
# Duration of each trial component (ms)
cfg.pre_stimuli_duration = 2000  # 2 seconds pre-stimulus period
cfg.trial_duration = 2000        # 2 seconds stimulation period
cfg.reset_duration = 2000        # 2 seconds cooling/reset period

# Complete trial cycle for one stimulus type (coherent or incoherent)
cfg.trial_cycle_duration = cfg.pre_stimuli_duration + cfg.trial_duration + cfg.reset_duration  # 6 seconds total

# For each reference spine, we need two complete cycles:
# 1. First cycle: coherent stimulation (pre-stim, stim, reset)
# 2. Second cycle: incoherent stimulation (pre-stim, stim, reset)
cfg.full_trial_duration = 2 * cfg.trial_cycle_duration  # 12 seconds total for one complete trial (coherent + incoherent)

# Check if TEST_MODE environment variable is set
test_mode = os.environ.get('TEST_MODE', 'false').lower() in ['true', '1', 't', 'yes', 'y']

# Stabilization period (ms)
cfg.t_stab = 2000  # Default: 2000 ms

# Number of trials per phase (D1test or D3test)
if test_mode:
    cfg.num_trials = 5  # TEST MODE: Only run 5 trials per phase
    cfg.t_stab = 100  # Reduced stabilization time for testing
    print("\n*** RUNNING IN TEST MODE: Reduced trials and timing ***\n")
else:
    cfg.num_trials = 100  # FULL RUN: Original 100 trials per phase

# Calculate duration of one full trial cycle
full_trial_cycle_ms = cfg.full_trial_duration

# Calculate the number of concurrent blocks needed
num_concurrent_blocks = math.ceil(cfg.num_trials / cfg.max_concurrent_trials)

# Calculate total duration for the D1test phase based on concurrent blocks
phase_duration_ms = num_concurrent_blocks * full_trial_cycle_ms

# Set timing for D1test only
cfg.tstart = cfg.t_stab
cfg.duration = cfg.tstart + phase_duration_ms 
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
cfg.printPopAvgRates = [0, 426e4]

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
# Initialize the cfg.recordTraces dictionary
#cfg.recordTraces = {'V_soma_ih': {'sec':'soma', 'loc':0.5, 'var':'gbar', 'mech':'hd', 'conds':{'pop': 'PT5B'}}}

						
						
						# 'V_apic_26': {'sec':'apic_26', 'loc':0.5, 'var':'v', 'conds':{'pop': 'PT5B'}},
						# 'V_dend_5': {'sec':'dend_5', 'loc':0.5, 'var':'v', 'conds':{'pop': 'PT5B'}}}
						

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
cfg.saveFolder = os.path.join(current_dir, 'results_fixed_max', f'seed{cfg.rdseed}')
cfg.savePickle = True
cfg.saveJson = True
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams']#, 'net']
cfg.backupCfgFile = None #['cfg.py', 'backupcfg/'] 
cfg.gatherOnlySimData = False
cfg.saveCellSecs = True
cfg.saveCellConns = True
cfg.compactConnFormat = 0

#------------------------------------------------------------------------------
# Analysis and plotting 
#------------------------------------------------------------------------------
#with open('cells/popColors.pkl', 'rb') as fileObj: popColors = pickle.load(fileObj)['popColors']

#cfg.analysis['plotRaster'] = {'include': allpops, 'orderBy': ['pop', 'y'], 'timeRange': [120*1000, 120*1000+10*1000], 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'popColors': popColors, 'figSize': (12,10), 'lw': 0.3, 'markerSize':3, 'marker': '.', 'dpi': 300} 
#cfg.analysis['plotRaster'] = {'include': allpops, 'orderBy': ['pop', 'y'], 'timeRange': [120*1000+5*60*1000, 120*1000+5*60*1000+10*1000], 'saveFig': True, 'showFig': False, 'popRates': True, 'orderInverse': True, 'popColors': popColors, 'figSize': (12,10), 'lw': 0.3, 'markerSize':3, 'marker': '.', 'dpi': 300} 

#cfg.analysis['plotTraces'] = {'include': [], 'timeRange': [0, cfg.duration], 'oneFigPer': 'trace', 'figSize': (10,4), 'saveFig': True, 'showFig': False} 
 

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
cfg.weightNorm = 0  # use weight normalization
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

cfg.EEGain = 0.5
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
cfg.PVSOMGain = 0.25
cfg.SOMPVGain = 0.25
cfg.PVPVGain = 0.25
cfg.SOMSOMGain = 0.25

#------------------------------------------------------------------------------
## I->E/I layer weights (L2/3+4, L5, L6)
cfg.IEweights = 0.8
cfg.IIweights = 1.0

cfg.IPTGain = 1.0
cfg.IFullGain = 1.0


#------------------------------------------------------------------------------
# Long range inputs
#------------------------------------------------------------------------------
cfg.addLongConn = 1 



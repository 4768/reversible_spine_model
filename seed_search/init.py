"""
init.py

Starting script to run NetPyNE-based M1 model.

Usage:
    python init.py # Run simulation, optionally plot a raster

MPI usage:
    mpiexec -n 4 nrniv -python -mpi init.py
    mpiexec -n 10 nrniv -python -mpi init_test.py

Contributors: salvadordura@gmail.com
"""

import matplotlib; #matplotlib.use('Agg')  # to avoid graphics error in servers
from netpyne import sim
from neuron import h
import os # Add os import
weights_dict = {}
find_vtrg = {}
v_cnt = 0
stp = 0


#------------------------------------------------------------------------------
def recordspike(ti, te):
    spktsAll = sim.simData['spkt'] 
    spkidsAll = sim.simData['spkid'] 
    print('spktsAll: ', spktsAll)
    print('spkidsAll: ', spkidsAll)
    filtered_spikes = [
        (spkid, spkt)
        for spkid, spkt in zip(spkidsAll, spktsAll)
        if ti <= spkt <= te
    ]
    print('spikes: ', filtered_spikes)
    spikes_dict = {}
    for spkid, spkt in filtered_spikes:
        if spkid in spikes_dict:
            spikes_dict[spkid].append(spkt)
        else:
            spikes_dict[spkid] = [spkt]
    
    save_path = os.path.join(cfg.saveFolder, f'spkt_{sim.pc.id()}.txt')
    with open(save_path, 'a') as f:
        for neuron_id, spkt_list in spikes_dict.items():
            f.write(f'{neuron_id}: {spkt_list}\n')
            
def setHSPEenable(cells, onoff, tp = "", Vtrg_list = []):
    vtrg_dict = {}

    for cell in cells:
        for conn in cell.conns:
            STDPmech = conn.get('hSTDP')
            if STDPmech:
                setattr(STDPmech, 'Eenable', onoff)
                if getattr(STDPmech, 'Eenable'):
                    secName = conn['sec']
                    seg = f'{conn['loc']:.6f}'
                    v = Vtrg_list[tp][cell][secName][seg][2]
                    v0 = Vtrg_list['init'][cell][secName][seg][2]
                    vtrg = v - 1/2 * (v - v0)
                    setattr(STDPmech, 'Vtrg', vtrg)
                    key = (cell.gid, secName, seg)
                    vtrg_dict[key] = (cell.gid, secName, seg, v0, v, vtrg)

    vtrg_data = list(vtrg_dict.values())
    vtrg_data.sort(key=lambda x: (x[0], x[1], x[2]))

    #with open(f"Vtrg_{sim.pc.id()}.txt", 'a') as f:
    #    for gid, secName, seg, v0, v, vtrg in vtrg_data:
    #        f.write(f"{tp}: {gid} {secName} {seg} {v0} {v} {vtrg}\n")
    #    f.flush()


def setSTDPon(cells, onoff):
    for cell in cells:
        if 'cellType' in cell.tags:
            for conn in cell.conns:
                STDPmech = conn.get('hSTDP')
                if STDPmech:
                    setattr(STDPmech, 'STDPon', onoff) 

def modifyMechsFunc(simTime):
    from netpyne import sim
    global weights_dict
    global find_vtrg
    global v_cnt
    t = simTime
    ts = cfg.tstart

    # Print period start messages (within 20ms of start)
    if any(abs(t - ts[period]) <= 20 for period in ['HSPset0', 'Baseline', 'D1train', 'HSPset1', 'D1test', 'HSP0', 'D3test', 'D4extinct', 'HSPset2', 'D4test', 'HSP1', 'D5extinct', 'D5test']):
        if abs(t - ts['HSPset0']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting HSP voltage target sampling 0 at t={t}")
        if abs(t - ts['Baseline']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting Baseline period at t={t}")
        elif abs(t - ts['D1train']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D1 training period at t={t}")
        elif abs(t - ts['HSPset1']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting HSP voltage target sampling 1 at t={t}")
        elif abs(t - ts['D1test']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D1 test period at t={t}")
        elif abs(t - ts['HSP0']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting HSP phase 0 at t={t}")
        elif abs(t - ts['D3test']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D3 test period at t={t}")
        elif abs(t - ts['D4extinct']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D4 extinction period at t={t}")
        elif abs(t - ts['HSPset2']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting HSP voltage target sampling 2 at t={t}")
        elif abs(t - ts['D4test']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D4 test period at t={t}")
        elif abs(t - ts['HSP1']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting HSP phase 1 at t={t}")
        elif abs(t - ts['D5extinct']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D5 extinction period at t={t}")
        elif abs(t - ts['D5test']) <= 20:
            print(f"Rank {sim.pc.id()}: Starting D5 test period at t={t}")

    # Print period end messages (20ms before next period)
    if any(abs(t - (ts[next_period] - 20)) <= 20 for next_period in ['Baseline', 'D1train', 'HSPset1', 'D1test', 'HSP0', 'D3test', 'D4extinct', 'HSPset2', 'D4test', 'HSP1', 'D5extinct', 'D5test']):
        if abs(t - (ts['Baseline'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed HSP voltage target sampling 0 at t={t}")
        if abs(t - (ts['D1train'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed Baseline period at t={t}")
        elif abs(t - (ts['HSPset1'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D1 training period at t={t}")
        elif abs(t - (ts['D1test'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed HSP voltage target sampling 1 at t={t}")
        elif abs(t - (ts['HSP0'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D1 test period at t={t}")
        elif abs(t - (ts['D3test'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed HSP phase 0 at t={t}")
        elif abs(t - (ts['D4extinct'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D3 test period at t={t}")
        elif abs(t - (ts['HSPset2'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed HSP voltage target sampling 2 at t={t}")
        elif abs(t - (ts['D4test'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D4 test period at t={t}")
        elif abs(t - (ts['HSP1'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed HSP phase 1 at t={t}")
        elif abs(t - (ts['D5extinct'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D5 extinction period at t={t}")
        elif abs(t - (ts['D5test'] - 20)) <= 20:
            print(f"Rank {sim.pc.id()}: Completed D5 test period at t={t}")

    #if 40 < t < 60:
    #    sim.saveData(include=['netParams', 'net'], filename='out_netParams_netInstance')
    #find time-averaged EPSP init, after D1 train, after D4 extinct when stable
    if ts['HSPset0'] + 2000 <= t < ts['HSPset0'] + cfg.t_hspvtrg or \
    ts['HSPset1'] + 2000 <= t < ts['HSPset1']+ cfg.t_hspvtrg or \
    ts['HSPset2'] + 2000 <= t < ts['HSPset2'] + cfg.t_hspvtrg:
        if ts['HSPset0'] + 2000 <= t < ts['HSPset0'] + cfg.t_hspvtrg:
            timepoint = 'init'
        elif ts['HSPset1'] + 2000 <= t < ts['HSPset1']+ cfg.t_hspvtrg:
            timepoint = 'After D1 train'
        else:
            timepoint = 'After D4 extinct'
        
        if timepoint not in find_vtrg:
            find_vtrg[timepoint] = {}
        for cell in sim.net.cells:
            if 'cellType' in cell.tags:
                if cell not in find_vtrg[timepoint]:
                    find_vtrg[timepoint][cell] = {}
                for secName, sec in cell.secs.items():
                    if secName not in find_vtrg[timepoint][cell]:
                        find_vtrg[timepoint][cell][secName] = {}
                    for seg in sec['hObj']:
                        loc = f'{seg.x:.6f}'
                        if loc not in find_vtrg[timepoint][cell][secName]:
                            find_vtrg[timepoint][cell][secName][loc] = [0, 0, 0]  # [voltage_sum, sample_count, avg]
                        
                        # Accumulate voltage and increment count
                        find_vtrg[timepoint][cell][secName][loc][0] += sec['hObj'](seg.x).v
                        find_vtrg[timepoint][cell][secName][loc][1] += 1

                        # Calculate average at end of time window
                        if (timepoint == 'init' and t >= ts['HSPset0'] + cfg.t_hspvtrg - 20) or \
                        (timepoint == 'After D1 train' and t >= ts['HSPset1'] + cfg.t_hspvtrg - 20) or \
                        (timepoint == 'After D4 extinct' and t >= ts['HSPset2'] + cfg.t_hspvtrg - 20):
                                voltage_sum, sample_count = find_vtrg[timepoint][cell][secName][loc][:2]
                                find_vtrg[timepoint][cell][secName][loc][2] = voltage_sum / sample_count if sample_count > 0 else 0


    if ts['HSP0'] + 2000 <= t < ts['HSP0'] + 2020 or \
        ts['HSP1'] + 2000 <= t < ts['HSP1'] + 2020:
        if ts['HSP0'] + 2000 <= t < ts['HSP0'] + 2020:
            vtrg_timepoint = 'After D1 train'
        else:
            vtrg_timepoint = 'After D4 extinct'
        setHSPEenable(sim.net.cells, 1, vtrg_timepoint, find_vtrg)
        if sim.pc.id() == 0:
            print(f"HSP changed from 0 to 1 at t={t}")
    elif ts['HSP0'] + cfg.t_HSP - 20 <= t < ts['HSP0'] + cfg.t_HSP or\
        ts['HSP1'] + cfg.t_HSP - 20 <= t < ts['HSP1'] + cfg.t_HSP:
        setHSPEenable(sim.net.cells, 0)
        if sim.pc.id() == 0:
            print(f"HSP changed from 1 to 0 at t={t}")

    # Enable STDP at the exact beginning of training and extinction phases
    if ts['D1train'] <= t < ts['D1train'] + 20 or \
     ts['D4extinct'] <= t < ts['D4extinct'] + 20 or \
     ts['D5extinct'] <= t < ts['D5extinct'] + 20:
        setSTDPon(sim.net.cells, 1)
        if sim.pc.id() == 0:
            print(f"STDP changed from 0 to 1 at t={t}")
    # Disable STDP at the exact beginning of test phases
    elif ts['D1test'] <= t < ts['D1test'] + 20 or \
        ts['D4test'] <= t < ts['D4test'] + 20 or \
        ts['D5test'] <= t < ts['D5test'] + 20:
        setSTDPon(sim.net.cells, 0)
        if sim.pc.id() == 0:
            print(f"STDP changed from 1 to 0 at t={t}")

    try:
        # Record weights at consistent times across all phases
        if ts['Baseline'] <= t < ts['Baseline'] + 21 or\
        ts['D1test'] <= t < ts['D1test'] + 21 or \
        ts['D3test'] <= t < ts['D3test'] + 21 or \
        ts['D4test'] <= t < ts['D4test'] + 21 or \
        ts['D5test'] <= t < ts['D5test'] + 21:


            if ts['Baseline'] <= t < ts['Baseline'] + 21:
                period = 'Baseline'
            elif ts['D1test'] <= t < ts['D1test'] + 21:
                period = 'After D1 train'
            elif ts['D3test'] <= t < ts['D3test'] + 21:
                period = 'Before D3 test'
            elif ts['D4test'] <= t < ts['D4test'] + 21:
                period = 'After D4 extinct' 
            elif ts['D5test'] <= t < ts['D5test'] + 21:
                period = 'After D5 extinct'
            from collections import defaultdict

            save_path = os.path.join(cfg.saveFolder, f'weight_node_{sim.pc.id()}_{cfg.rdseed}.txt')

            weightss = defaultdict(list)
            
            for cell in sim.net.cells:
                # Skip VecStim cells
                if 'cellType' in cell.tags and cell.tags['cellType'] == 'VecStim':
                    continue
                    
                # Get cell type, default to 'Unknown' if not specified
                cell_type = cell.tags.get('cellType', 'Unknown')
                
                for conn in cell.conns:
                    STDPmech = conn.get('hSTDP')
                    if STDPmech:
                        # Record weights for all sections, not just Adend1-3
                        spineid = f"{cell_type}_{cell.gid}_{conn['preGid']}.{conn['sec']}.{conn['loc']:.6f}"
                        weight = conn['hObj'].weight[0]
                        weightss[spineid].append(weight)
                
            avgweights = {spineid: sum(weights) / len(weights) for spineid, weights in weightss.items()}
            
            # Store averages for this period
            for spineid, avg_weight in avgweights.items():
                if spineid not in weights_dict:
                    weights_dict[spineid] = {}
                weights_dict[spineid][period] = avg_weight
            
            weights_dict = {spineid: weights_dict[spineid] for spineid in sorted(weights_dict)}
            
                # Only write the current period's weight values to file
            if period == 'Before D3 test':
                with open(save_path, 'a') as f:
                    f.write(f"Time: {t}; Period: {period}\n")
                    for spineid, weights in sorted(avgweights.items()):
                        f.write(f"{spineid}: {weights}\n")
                    
                    f.flush()

        # Record branch and spine activity at consistent time windows during stimulation periods
        #if 0 <= t < 22 or \
            #ts['HSPset1'] <= t < ts['HSPset1'] + cfg.t_hspvtrg or\
            #ts['HSPset2'] <= t < ts['HSPset2'] + cfg.t_hspvtrg or\
        if  ts['Baseline'] <= t < ts['Baseline'] + 15*1000 or \
            ts['Baseline'] + 30*1000 <= t < ts['Baseline'] + 45*1000 or \
            ts['D3test'] <= t < ts['D3test'] + 15*1000 or \
            ts['D3test'] + 30*1000 <= t < ts['D3test'] + 45*1000 or\
            ts['D4test'] <= t < ts['D4test'] + 15*1000 or \
            ts['D4test'] + 30*1000 <= t < ts['D4test'] + 45*1000 or \
            ts['D5test'] <= t < ts['D5test'] + 15*1000 or \
            ts['D5test'] + 30*1000 <= t < ts['D5test'] + 45*1000:
            #ts['D1test'] <= t < ts['D1test'] + 15*1000 or \
            #ts['D1test'] + 30*1000 <= t < ts['D1test'] + 45*1000 or \
            #ts['HSPset0'] <= t < ts['HSPset0'] + cfg.t_hspvtrg or\


            if 0 <= t < 22:
               timepoint = 'debug'
            elif ts['HSPset0'] <= t < ts['HSPset0'] + cfg.t_hspvtrg:
                timepoint = 'HSPset0'
            elif ts['HSPset1'] <= t < ts['HSPset1'] + cfg.t_hspvtrg:
                timepoint = 'HSPset1'
            elif ts['HSPset2'] <= t < ts['HSPset2'] + cfg.t_hspvtrg:
                timepoint = 'HSPset2'
            elif ts['Baseline'] <= t < ts['Baseline'] + 15*1000:
                timepoint = 'baseline 4k'
            elif ts['Baseline'] + 30*1000 <= t < ts['Baseline'] + 45*1000:
                timepoint = 'baseline 12k'
            elif ts['D1test'] <= t < ts['D1test'] + 15*1000: 
                timepoint = 'D1 test 4k'
            elif ts['D1test'] + 30*1000 <= t < ts['D1test'] + 45*1000: 
                timepoint = 'D1 test 12k'
            elif ts['D3test'] <= t < ts['D3test'] + 15*1000: 
                timepoint = 'D3 test 4k'
            elif ts['D3test'] + 30*1000 <= t < ts['D3test'] + 45*1000: 
                timepoint = 'D3 test 12k'
            elif ts['D4test'] <= t < ts['D4test'] + 15*1000:
                timepoint = 'D4 test 4k'
            elif ts['D4test'] + 30*1000 <= t < ts['D4test'] + 45*1000:
                timepoint = 'D4 test 12k'
            elif ts['D5test'] <= t < ts['D5test'] + 15*1000: 
                timepoint = 'D5 test 4k'
            elif ts['D5test'] + 30*1000 <= t < ts['D5test'] + 45*1000: 
                timepoint = 'D5 test 12k'
            save_path = os.path.join(cfg.saveFolder, f'Branch_activity_node_{sim.pc.id()}_{cfg.rdseed}.txt')
            with open(save_path, 'a') as f:
                Bactivity = []
                Spine_act = []
                for cell in sim.net.cells:
                    if 'cellType' in cell.tags and cell.tags['cellType'] == 'PT':
                        for secName, sec in cell.secs.items():
                            secV = 0
                            n = 0
                            if secName in ['Adend1', 'Adend2', 'Adend3', 'Bdend']:
                                cur = []
                                for seg in sec['hObj']:
                                    n = n + 1
                                    v = sec['hObj'](seg.x).v
                                    secV += v
                                avgBranchV = secV / n
                                activityinfo = f"{cell.gid} {secName}: {avgBranchV:.6f};"
                                Bactivity.append(activityinfo)
#                    for conn in cell.conns:
#                        try:
#                            # Try to get iNMDA attribute safely
#                            s_act = getattr(conn['hObj'].syn(), 'iNMDA', None)
#                            if s_act is not None:
#                                s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: {s_act:.6f};"
#                                Spine_act.append(s_act_info)
#                            else:
#                                # Try to get 'i' attribute as fallback
#                                s_act = getattr(conn['hObj'].syn(), 'i', 0)
#                                s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: {s_act:.6f};"
#                                Spine_act.append(s_act_info)
#                        except Exception as e:
#                            # If any error occurs, record 0 activity for this synapse
#                            print(f"Warning: Failed to get synapse activity for {cell.gid}.{conn['sec']}.{conn['loc']:.6f} - {str(e)}")
#                            s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: 0.000000;"

                Bactivity.sort()
                f.write(f"{t}; {timepoint}\n")
#                f.write(f"\tBranch activity:\n")
                f.write(f"\t{' '.join(Bactivity)}\n")
#                f.write(f"\tSpine activity:\n")
#                f.write(f"\t\t{' '.join(Spine_act)}\n")
                f.flush()

        # Calculate spine elimination/formation ratios at consistent test points
        if  ts['D1test'] + 20 <= t < ts['D1test'] + 40 or \
            ts['D3test'] <= t < ts['D3test'] + 20 or \
            ts['D4test'] <= t < ts['D4test'] + 20 or \
            ts['D5test'] <= t < ts['D5test'] + 20:
            periods = ['Baseline', 'After D1 train', 'Before D3 test', 'After D4 extinct', 'After D5 extinct']

            ipre = 0
            if ts['D1test'] + 20 <= t < ts['D1test'] + 40:
                i = 1
            elif ts['D3test'] <= t < ts['D3test'] + 20:
                i = 2 
                #ipre = 1
            elif ts['D4test'] <= t < ts['D4test'] + 20:
                i = 3
                #ipre = 2
            elif ts['D5test'] <= t < ts['D5test'] + 20:
                i = 4
                #ipre = 3

            n_a = {'total spine': 0, 'elim spine': 0, 'form spine': 0}
            n_t = {'total spine': 0, 'elim spine': 0, 'form spine': 0}
            for spineid in weights_dict:
                if 'PT' in spineid:
                    if 'adend' in spineid.lower():
                        w_pre_a = weights_dict[spineid].get(periods[ipre], 0)
                        w_post_a = weights_dict[spineid].get(periods[i], 0) 
                        
                        if w_pre_a > 3:
                            n_a['total spine'] += 1
                            if w_post_a < 3:
                                n_a['elim spine'] += 1
                        elif w_pre_a < 3 and w_post_a > 3:
                            n_a['form spine'] += 1
                    if 'dend' in spineid.lower():
                        w_pre_t = weights_dict[spineid].get(periods[ipre], 0)
                        w_post_t = weights_dict[spineid].get(periods[i], 0) 
                        
                        if w_pre_t > 3:
                            n_t['total spine'] += 1
                            if w_post_t < 3:
                                n_t['elim spine'] += 1
                        elif w_pre_t < 3 and w_post_t > 3:
                            n_t['form spine'] += 1
            
            if n_a['total spine'] > 0:
                elim_rate_a = n_a['elim spine'] / n_a['total spine'] * 100
                form_rate_a = n_a['form spine'] / n_a['total spine'] * 100
            else:
                elim_rate_a = 0
                form_rate_a = 0
                print("0 apical spine")

            if n_t['total spine'] > 0:
                elim_rate_t = n_t['elim spine'] / n_t['total spine'] * 100
                form_rate_t = n_t['form spine'] / n_t['total spine'] * 100
            else:
                elim_rate_t = 0
                form_rate_t = 0
                print("0 spine in total")

            # Save spine elimination and formation data to file
            with open(os.path.join(cfg.saveFolder, f'spine_elim_form_node_{sim.pc.id()}_{cfg.rdseed}.txt'), 'a') as f:
                f.write(f"Apical: {periods[ipre]} to {periods[i]}: elim {elim_rate_a:.3f}%; form {form_rate_a:.3f}%\n")
                f.write(f"Total: {periods[ipre]} to {periods[i]}: elim {elim_rate_t:.3f}%; form {form_rate_t:.3f}%\n")
                f.flush()
        
    except Exception as e:
        import traceback
        if sim.pc.id() == 0:
            print(f"Error in modifyMechsFunc at t={t}: {e}")
            traceback.print_exc()
'''        # Record spikes at the end of training and extinction phases
        if ts['D1train'] + cfg.t_train - 20 <= t < ts['D1train'] + cfg.t_train:
            recordspike(t - cfg.t_train, t)
            if sim.pc.id() == 0:
                print(f"Recording spikes at end of D1 training phase: {t-cfg.t_train} to {t}")

        # Record spikes during all test phases (first 1000ms)
        if (ts['D1test'] <= t < ts['D1test'] + 1000 or \
            ts['D3test'] <= t < ts['D3test'] + 1000 or \
            ts['D4test'] <= t < ts['D4test'] + 1000 or \
            ts['D5test'] <= t < ts['D5test'] + 1000) and t % 1000 == 0:
            recordspike(t - 1000, t)
            if sim.pc.id() == 0:
                print(f"Recording spikes during test phase: {t-1000} to {t}")

        # Record spikes at the end of extinction phases
        if (ts['D4extinct'] + cfg.t_extinct - 20 <= t < ts['D4extinct'] + cfg.t_extinct or \
            ts['D5extinct'] + cfg.t_extinct - 20 <= t < ts['D5extinct'] + cfg.t_extinct):
            recordspike(t - 1000, t)
            if sim.pc.id() == 0:
                print(f"Recording spikes at end of extinction phase: {t-1000} to {t}")'''


#######current version of recording spine activity
'''                    for conn in cell.conns:
                        try:
                            # Try to get iNMDA attribute safely
                            s_act = getattr(conn['hObj'].syn(), 'iNMDA', None)
                            if s_act is not None:
                                s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: {s_act:.6f};"
                                Spine_act.append(s_act_info)
                            else:
                                # Try to get 'i' attribute as fallback
                                s_act = getattr(conn['hObj'].syn(), 'i', 0)
                                s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: {s_act:.6f};"
                                Spine_act.append(s_act_info)
                        except Exception as e:
                            # If any error occurs, record 0 activity for this synapse
                            print(f"Warning: Failed to get synapse activity for {cell.gid}.{conn['sec']}.{conn['loc']:.6f} - {str(e)}")
                            s_act_info = f"{cell.gid}.{conn['sec']}.{conn['loc']:.6f}.{conn['hObj'].syn()}: 0.000000;"
                            Spine_act.append(s_act_info)'''


# -----------------------------------------------------------
# Main code

from mpi4py import MPI
import os # Add os import again just in case, although once should be enough


cfg, netParams = sim.readCmdLineArgs()
sim.initialize(
    simConfig = cfg, 	
    netParams = netParams)  # create network object and set cfg and net params

# Ensure the save directory exists
if sim.rank == 0:
    os.makedirs(cfg.saveFolder, exist_ok=True)
sim.pc.barrier() # Ensure all ranks wait until directory is created

sim.pc.timeout(0)                          # set nrn_timeout threshold to X sec (max time allowed without increasing simulation time, t; 0 = turn off)
sim.net.createPops()               			# instantiate network populations
sim.net.createCells()              			# instantiate network cells based on defined populations
sim.net.connectCells()            			# create connections between cells based on params

hsp_id = []

'''for cell in sim.net.cells:
    if cell.tags['pop'] in ['HSP1', 'HSP2', 'HSP3']:
        hsp_id.append(cell.gid)

print(hsp_id)'''
# Synchronize again after setting parameters
sim.pc.barrier()

def find_coord(gid_range):
    # Local coord on this node
    local_coord = []
    
    # Get local cells within gid range
    local_cells = {cell.gid: [cell.tags['x'], cell.tags['y'], cell.tags['z']] 
                    for cell in sim.net.cells if gid_range[0] <= cell.gid <= gid_range[1]}
    
    return local_cells

local_coord = find_coord(gid_range=(40, 99))
#print(f"Node {sim.pc.id()}: Local coord = {local_coord}")

all_coord = sim.pc.py_allgather([local_coord])  # Each node sends its list
#print(all_coord)
global_coord = {}
for node_coord in all_coord:
    for k, v in node_coord[0].items():
        global_coord[k] = v
#print(f"Node {sim.pc.id()}: Gathered coord = {global_coord}")


# Point postsynaptic membrane potential to HSP in stdp.mod
n = 0
from netpyne.network.subconn import posFromLoc
from math import sqrt
for cell in sim.net.cells:  # Each rank processes its own cells
    for conn in cell.conns:
        #print(f'{cell.gid}: {conn}')
        STDPmech = conn.get('hSTDP')
        if STDPmech:
            #nc = sim.pc.gid_connect(rnd_id, STDPmech)          
            setattr(STDPmech, 'NEproc', 1)
            setattr(STDPmech, 'Etau', 1e3)
            secName = conn['sec']
            loc = conn['loc']
            h.setpointer(cell.secs[secName]['hObj'](loc)._ref_v, 'v_postsyn', STDPmech)
            h.setpointer(conn['hObj']._ref_weight[0], 'Egmax', conn['hObj'].syn())
            #egmax = getattr(conn['hObj'].syn(), 'Egmax')
            #print(f'egmax{conn['hObj'].syn()}: {egmax}')
            #h.setpointer(conn['hObj']._ref_weight[0], 'Egmax', conn['hObj'].syn())
            #for hsp in [100, 101, 102]:
            #    nc = sim.pc.gid_connect(hsp, conn['hObj'].syn())
            #    nc.weight[0] = 1
            #    print(f'{sim.pc.id()}:{nc}, {hsp}, {conn['hObj'].syn()}')
            if any(char in conn['label'] for char in ['S1_4K', 'S1_12K', 'OC']):
                x, y, z = posFromLoc(cell.secs[conn['sec']], conn['loc'])
                x0, y0, z0 = global_coord[conn['preGid']]
                d = sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
                conn.delay = d/netParams.propVelocity


# Final synchronization
sim.pc.barrier()

                    
sim.net.addStims() 							# add network stimulation
sim.setupRecording()              			# setup variables to record for each cell (spikes, V traces, etc)

# Clear/Create log files in the correct directory
log_files = [
    f'weight_node_{sim.pc.id()}_{cfg.rdseed}.txt',
    f'Branch_activity_node_{sim.pc.id()}_{cfg.rdseed}.txt',
    f'spine_elim_form_node_{sim.pc.id()}_{cfg.rdseed}.txt',
    #f'spkt_{sim.pc.id()}.txt',
    #f'Vtrg_{sim.pc.id()}.txt'
]

for filename in log_files:
    filepath = os.path.join(cfg.saveFolder, filename)
    with open(filepath, 'w') as f:
        pass


sim.runSimWithIntervalFunc(20, modifyMechsFunc)       # run parallel Neuron simulation

sim.saveDataInNodes()
sim.gatherDataFromFiles()

sim.saveData()                    			# save params, cell info and sim output to file (pickle,mat,txt,etc)#
NEURON {
    ARTIFICIAL_CELL Randomizer
	RANGE Eintrvl			: mean excitatory ISI
	RANGE NEproc			: number of processes to generate
	RANGE proc_num		    : current process number (global variable)
                            : used to randomize the receiver side to 1/NEproc
                            : for simplicity, we set the probability of receiver side directly
	RANGE special
    RANGE sd 				:seed
    RANGE period_start, period_stop  : arrays for start and stop times of periods
    RANGE n_periods         : number of periods
}

PARAMETER {
    sd = 0              : seed for random number generator
    Eintrvl             : mean excitatory ISI (ms)
    NEproc              : number of processes to generate
    special = 0         : flag to create same input stream for 1 Hz
    n_periods = 0       : number of active periods
}

ASSIGNED {
    proc_num
    last_event
    tshift
    skip
    count
    period_start[10]    : array of start times (ms), max 10 periods
    period_stop[10]     : array of stop times (ms), max 10 periods
}

INITIAL {
    LOCAL j
    seed(sd)            : set random seed
    skip = 0
    count = 0
    if (special) {
        Eintrvl = Eintrvl / 2.5
    }
    last_event = 0
    tshift = 0
    j = 0
    while (j < NEproc) {
        net_send(exprand(Eintrvl), j)
        j = j + 1
    }
}

NET_RECEIVE(w) { 
    LOCAL in_period, i
    if (t == last_event) {
        tshift = tshift + dt/10
        net_send(tshift, flag)
:       printf("Randomizer, event overlap: time = %g process %g\n", t, flag)
    } else {
        proc_num = flag
        net_send(exprand(Eintrvl), flag)
        : Special case for 1 Hz raster
        if (special) {
            count = count + 1
            if (count == 2 || count == 4) {
                skip = 0
            } else {
                skip = 1
            }
            if (count == 5) {
                count = 0
            }
        } else {
            skip = 0  : Reset skip if special is off
        }

        : Check if current time is within any defined period
        in_period = 0  : Default to not firing unless within a period
        if (n_periods == 0) {
            in_period = 1  : Allow all events if no periods defined
        }
        if (n_periods > 0 && n_periods <= 10) {
            i = 0
            while (i < n_periods) {
                if (t >= period_start[i] && t <= period_stop[i]) {
                    in_period = 1
                }
                i = i + 1
            }
        }
        if (!skip){: && in_period) {
            net_event(t)
        }
        tshift = 0
    }
    last_event = t
}

PROCEDURE seed(x) {
    set_seed(x)
}

: Procedure to set periods from Python
PROCEDURE set_periods(start, stop, n, idx) {
    if (idx == 0) {
        n_periods = n  : Only set n_periods on first call
    }
    if (idx < n && idx < 10) {  
        period_start[idx] = start
        period_stop[idx] = stop
    }
}
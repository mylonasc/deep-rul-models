import numpy as np
# This is code to create a sequence of observations from a random process that have the following properties:
# * There is a latent variable (Z), stochastically evolving, which when it passes a certain threshold the process stops (at time tf).
# * A signal X (some spikes overlapped over background signal) is related in a non-linear manner on the latent variable
# * The dataset is created with the following ML task in mind:
#   - to learn a model for tf-t (where t is the current time) only by sparse observations of "X" in a fully data-driven manner.
#   - learn implicitly some representation of "Z" and how it evolves in time, as well as a model that gives an estimate of current "Z" given previous records of "X". 


def lhs_sample(npoints, range_size = 10,  lranges=None):
    if lranges is None:
        lranges = npoints;

    nchoices = np.random.choice(range_size,npoints);
    ret_vals = [];
    for k in range(lranges) :
        ret_vals.append( k * range_size + nchoices[k])
    return ret_vals

def transform_exp_data_to_random_signal_params(exp_dat_, rr = None):
    """
    Splits the stochastic degradation to segments that are to be the inputs to another
    function that creates samples from 
    rr: how many consecutive samples to use for making a segment.
    """
    ee = exp_dat_[:-(exp_dat_.shape[0]%rr)].reshape([-1,rr]) if (exp_dat_.shape[0]%rr != 0) else exp_dat_.reshape([-1,rr])
    return ee

def add_disturbances_in_signal(ee, speed =50, npoints = 1000):
    # Have disturbances that are proportional to the gamma random variable.
    #  points per segment.
     # a parameter in order to make the problem a bit more difficult.
    t = np.linspace(0,2*np.pi,npoints); # segment time

    v = np.sin(t*speed)*0.1 + np.random.randn(t.shape[0])*0.0
    
    #ndist_samples = 10
    ndisturb_samples = 20
    tseg = np.linspace(0,2*np.pi,ndisturb_samples)

    random_segment_pos = lhs_sample(ee.shape[1],int(np.floor(npoints/ee.shape[1]))) # controls the index where the signal disturbance is added.
    
    #pplot.plot(np.sin(v))
    vals=[]
    for seg in ee:
        vc = v.copy();
        random_segment_pos = lhs_sample(10,int(npoints/10)+1)
        for t_s,s in zip(random_segment_pos, seg):
            vc[t_s:t_s+ndisturb_samples] += np.sin(t[t_s:t_s+ndisturb_samples]*speed*5)*(s/160+1)**2

        vals.append(vc)
            #vals = t_s
            #v[t_] + np.sin(t*)
            #pplot.vlines(tt, ymin = 0 ,ymax = 0.2)
    len(vals)
    return np.vstack(vals)


def get_signal_for_segments(exp_dat_, speed, rr=10) :
    return add_disturbances_in_signal(transform_exp_data_to_random_signal_params(exp_dat_, rr = rr), speed)


def get_dat(d, rr = None):
    """
    Transforms the latent values to a timeseries segment. 
    
    The timeseries segment is used for prediction. The "case" input parameter contains the information of which loading case we are producing datra from.
    Different cases have a different background "noise". The network should learn to exploit this background for better prediction (different cases have different rates of damage accumulation).
    """
    speed_dict = {0:20, 1:25, 2:30};
    a = d['latent_values']
    get_signal_for_segments = lambda exp_dat_, speed : add_disturbances_in_signal(transform_exp_data_to_random_signal_params(exp_dat_, rr =rr), speed)

    l = d['latent_values']
    l2 =d['ttf']
    trunc_l = l[:-(l2.shape[0]%rr)] if (l2.shape[0]%rr !=0)  else l
    y = np.mean(trunc_l.reshape([-1,rr]),1)
    case = d['case']
    X = get_signal_for_segments(l, speed= speed_dict[case])
    eid = d['exp_index']
    exp_data = {"X" : X , "eid" : eid , "y" : y, "case" : case}    
    return exp_data


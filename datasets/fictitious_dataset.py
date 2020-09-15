import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
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

def transform_exp_data_to_random_signal_params(exp_dat_, rr = 10):
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

    v = np.sin(t*speed)*0.1 + np.random.randn(t.shape[0])*0.05
    
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
            vc[t_s:t_s+ndisturb_samples] += np.sin(t[t_s:t_s+ndisturb_samples]*speed*5)*(s/250+np.random.randn()/5)**2

            #vc[t_s:t_s+ndisturb_samples] += np.sin(t[t_s:t_s+ndisturb_samples]*speed*5)*(s/160+1)**2

        vals.append(vc)
            #vals = t_s
            #v[t_] + np.sin(t*)
            #pplot.vlines(tt, ymin = 0 ,ymax = 0.2)
    len(vals)
    return np.vstack(vals)


def get_signal_for_segments(exp_dat_, speed, rr = 10): 
    return  add_disturbances_in_signal(transform_exp_data_to_random_signal_params(exp_dat_, rr =rr), speed)


def get_dat(d, rr = 10):
    """
    NOT USED - Maybe remove?
    Transforms the latent values to a timeseries segment. 
    
    The timeseries segment is used for prediction. The "case" input parameter contains the information of which loading case we are producing data from.
    Different cases have a different background "noise". The network should learn to exploit this background for better prediction (different cases have different rates of damage accumulation).
    """
    speed_dict = {0:20, 1:25, 2:30};
    a = d['latent_values']
    get_signal_for_segment_ = lambda exp_dat_, speed : get_signal_for_segment(exp_dat_, speed, rr=rr)


    l = d['latent_values']
    l2 =d['ttf']
    trunc_l = l[:-(l2.shape[0]%rr)] if (l2.shape[0]%rr !=0)  else l
    y = np.mean(trunc_l.reshape([-1,rr]),1)
    case = d['case']
    X = get_signal_for_segments(l, speed= speed_dict[case])
    eid = d['exp_index']
    exp_data = {"X" : X , "eid" : eid , "y" : y, "case" : case}    
    return exp_data

def make_experiments_3conditions(nexp_per_case = 1 ,base_rate = 1.0, 
                     rates_shift = -0.5 , base_concentration = 3.,
                     concentration_shifts = 0.2):
    """
    This function creates the latent variable for all cases. 

    When the cummulative latent variable passes a threshold failure occurs.
    The returned experiments correspond to 3 damage evolution regimes.
    These parameters were hand-picked so that the evolution is aligned with 
    some assumptions on the evolution of damage, have sufficient 
    variation to make the problem non-trivial, and have sufficient 
    variation between the 3 evolution cases.  In future implementations it may be 
    interesting to combine evolution regimes for representing arbitrary loading.

    """
    rates = [base_rate,base_rate-rates_shift,base_rate+rates_shift]
    concentrations = [base_concentration,
            base_concentration-concentration_shifts,
            base_concentration+concentration_shifts]
    params = [(r,c) for r,c in zip(rates, concentrations)]

    def func(case):
        """
        Create a series of latent variable paths corresponding to experiments in different conditions. 
        Each latent variable is realized 
        """
        r,c = params[case]
        t  = np.linspace(0,1,3000)
        # For every different loading condition, 
        # corresponds a different rate and concentration parameter for a gamma distribution
        # Failure is defined when a threshold on the accumulated gamma steps is reached. 

        v1 = tfd.Gamma(concentration = 0.02+t**(c),rate = r).sample(1).numpy().T
        v1= np.cumsum(v1)
        v1 = v1[v1 < 250]
        return v1

    # Create the latent underlying data:
    all_dat = [];
    exp_index = 0;
    for kk in range(nexp_per_case):    
        for nn in range(3):
            exp_dat = np.diff(func(nn))
            exp_dat = (func(nn))

            df_dat = {"case" : nn, "exp_index" : exp_index, "latent_values" : exp_dat, "ttf" : len(exp_dat)-np.array([rr for rr in range(len(exp_dat))])};
            all_dat.append(df_dat)
            exp_index += 1;
            
    return all_dat

class FictitiousDataset:
    def  __init__(self, n_exp_per_case=10, pct_val_set = 0.3):
        """
        n_exp_per_case : number of experiment for each "loading" case.
                         each "case" represents a different mean evolution of the underlying latent.
        pct_val_set :    percentage of experiments from each load case kept for validation.
        """
        
        latent = make_experiments_3conditions(nexp_per_case=n_exp_per_case)
        self.pct_val_set = pct_val_set 
        
        all_exp_dat = []
        rr = 10

        
        
        def reshape_ttf(v):
            if v.shape[0]%rr == 0:
                return np.mean(v.reshape([-1,rr]),1)
            else:
                return np.mean(v[:-(v.shape[0]%rr)].reshape([-1,rr]),1)

        for d in latent:
            l = d['latent_values']
            ttf =d['ttf']

            y = reshape_ttf(ttf)
            case = d['case']
            speed_dict = {0:20, 1:25, 2:30}; # This is to superimpose a salient, yet irrelevant feature in the time-series.
                                             # Internally the network is expected to exploit this feature 
            if y.shape[0] == 0:
                break
            X = get_signal_for_segments(l, speed= speed_dict[case])
            eid = d['exp_index']

            exp_data = {
                "X" : X ,
                "eid" : (np.ones([X.shape[0],1]) * eid).astype(int),
                "y" : y,
                "case" : np.ones([X.shape[0],1]) * case
            }

            all_exp_dat.append(exp_data)

        X, eid, y, cases = [[],[],[],[]] #['X','eid','y']]
        for expdat in all_exp_dat:
            X.extend(expdat['X'])
            eid.extend(expdat['eid'])
            y.extend(expdat['y'])
            cases.extend(expdat['case'])

        X, eid, y , cases= [np.vstack(a) for a in [X, eid, y, cases]]
        X = X[...,np.newaxis]
        eid_oh = np.zeros([eid.shape[0],int(np.max(eid)) + 1])
        for k in np.unique(eid):
            k = int(k)    
            eid_oh[eid.flatten() == k,k] = 1.;
            
        
        self.X, self.eid, self.eid_oh, self.y ,self.cases= [X, eid, eid_oh, y, cases]
        inds_train = [];
        inds_test = [];
        for c in np.unique(cases): # 3 of them.
            eid_unique = np.unique(self.eid[self.cases.flatten() == c,...])
            ntrain = int((1-pct_val_set)*len(eid_unique))
            inds_train.extend(eid_unique[0:ntrain])
            inds_test.extend(eid_unique[ntrain:])
            
        self.inds_train = inds_train
        self.inds_test = inds_test
        self.inds_exp_source = inds_train
        self.inds_exp_target = inds_test
        print("Created random data for fictitious experiment.")
        print("training experiments: %i , testing %i"%(len(inds_train), len(inds_test)))
        self.normalization_factor_time = np.std(self.y)*10
        self.yrem_norm = self.y / self.normalization_factor_time


import theano.tensor as tt
import pymc3 as pm
from pymc3.distributions import Continuous
import theano
import numpy as np



def calculate_precip_store(S,precip_net,x1):
    """Calculates the amount of rainfall which enters the storage reservoir."""
    n = x1*(1 - (S / x1)**2) * tt.tanh(precip_net/x1)
    d = 1 + (S / x1) * tt.tanh(precip_net / x1)
    return n/d

# Determines the evaporation loss from the production store
def calculate_evap_store(S,evap_net,x1):
    """Calculates the amount of evaporation out of the storage reservoir."""
    n = S * (2 - S / x1) * tt.tanh(evap_net/x1)
    d = 1 + (1- S/x1) * tt.tanh(evap_net / x1)
    return n/d

# Determines how much water percolates out of the production store to streamflow
def calculate_perc(current_store,x1):
    """Calculates the percolation from the storage reservoir into streamflow."""
    return current_store * (1- (1+(4.0/9.0 * current_store / x1)**4)**-0.25)

def hydrograms(x4_limit,x4):
    """Produces a vector which partitions an input of water into streamflow over
    the course of several days.

    This function is intended to be run once as a part of GR4J, before the main
    loop executes.

    Parameters
    ----------
    x4_limit : scalar Theano tensor
        An upper limit on the value of x4, which is real-valued
    x4 : scalar Theano tensor
        A parameter controlling the rapidity of transmission of stream inputs
        into streamflow.

    Returns
    -------
    UH1 : 1D Theano tensor
        Partition vector for the portion of streamflow which is routed with a
        routing store.
    UH2 : 1D Theano tensor
        Partition vector for the portion of streamflow which is NOT routed with
        the routing store.
    """
    timesteps = tt.arange(2*x4_limit)

    SH1  = tt.switch(timesteps <= x4,(timesteps/x4)**2.5,1.0)
    SH2A = tt.switch(timesteps <= x4, 0.5 * (timesteps/x4)**2.5,0)
    SH2B = tt.switch(( x4 < timesteps) & (timesteps <= 2*x4),
                     1 - 0.5 * (2 - timesteps/x4)**2.5,0)

    # The next step requires taking a fractional power and
    # an error will be thrown if SH2B_term is negative.
    # Thus, we use only the positive part of it.
    SH2B_term = tt.maximum((2 - timesteps/x4),0)
    SH2B = tt.switch(( x4 < timesteps) & (timesteps <= 2*x4),
                     1 - 0.5 * SH2B_term**2.5,0)
    SH2C = tt.switch( 2*x4 < timesteps,1,0)

    SH2 = SH2A + SH2B + SH2C
    UH1 = SH1[1::] - SH1[0:-1]
    UH2 = SH2[1::] - SH2[0:-1]
    return UH1,UH2

def streamflow_step(P,E,S,runoff_history,R,x1,x2,x3,x4,UH1,UH2):
    """Logic for simulating a single timestep of streamflow from GR4J within
    Theano.

    This function is usually used as an argument to theano.scan as the inner
    function for a loop.

    Parameters
    ----------
    P : scalar Theano tensor
        Current timestep's value for precipitation input.
    E : scalar Theano tensor
        Current timestep's value for precipitation input.
    S : scalar Theano tensor
        Beginning value of storage in the storage reservoir.
    runoff_history : 1D Theano tensor
        Previous days' levels of streamflow input. Needed for routing streamflow
        over multiple days.
    R : scalar Theano tensor
        Beginning value of storage in the routing reservoir.
    x1 : scalar Theano tensor
        Storage reservoir parameter
    x2 : scalar Theano tensor
        Catchment water exchange parameter
    x3 : scalar Theano tensor
        Routing reservoir parameters
    x4 : scalar Theano tensor
        Routing time parameter
    UH1 : 1D Theano tensor
        Partition vector routing daily stream inputs into multiday streamflow
        for the fraction of water which interacts with the routing reservoir
    UH1 : 1D Theano tensor.
        Partition vector routing daily stream inputs into multiday streamflow
        for the fraction of water which does not interact with the routing
        reservoir.

    Returns
    -------
    Q : scalar Theano tensor
        Resulting streamflow
    S : scalar Theano tensor
        Storage reservoir level at the end of the timestep
    runoff_history : 1D Theano tensor
        Past timesteps' stream input values
    R : scalar Theano tensor
        Routing reservoir level at the end of the timestep
    UH2 : 1D Theano tensor
        Partition vector for the portion of streamflow which is NOT routed with
        the routing store.
    """
    # Calculate net precipitation and evapotranspiration
    precip_difference = P-E
    precip_net    = tt.maximum(precip_difference,0)
    evap_net      =  tt.maximum(-precip_difference,0)

    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S,precip_net,x1)

    # Calculate the amount of evaporation from storage
    evap_store   = calculate_evap_store(S,evap_net,x1)

    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store

    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S,x1)
    S = S  - perc

    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + ( precip_net - precip_store)

    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = tt.roll(runoff_history,1)
    runoff_history = tt.set_subtensor(runoff_history[0],current_runoff)

    Q9 = 0.9* tt.dot(runoff_history,UH1)
    Q1 = 0.1* tt.dot(runoff_history,UH2)

    F = x2*(R/x3)**3.5
    R = tt.maximum(0,R+Q9+F)

    Qr = R * (1-(1+(R/x3)**4)**-0.25)
    R = R-Qr

    Qd = tt.maximum(0,Q1+F)
    Q = Qr+Qd

    # The order of the returned values is important because it must correspond
    # up with the order of the kwarg list argument 'outputs_info' to theano.scan.
    return Q,S,runoff_history,R

def streamflow_step_tv_x1(P,E,x1,S,runoff_history,R,x2,x3,x4,UH1,UH2):

    # Calculate net precipitation and evapotranspiration
    precip_difference = P-E
    precip_net    = tt.maximum(precip_difference,0)
    evap_net      =  tt.maximum(-precip_difference,0)

    # Calculate the fraction of net precipitation that is stored
    precip_store = calculate_precip_store(S,precip_net,x1)

    # Calculate the amount of evaporation from storage
    evap_store   = calculate_evap_store(S,evap_net,x1)

    # Update the storage by adding effective precipitation and
    # removing evaporation
    S = S - evap_store + precip_store

    # Update the storage again to reflect percolation out of the store
    perc = calculate_perc(S,x1)
    S = S  - perc

    # The precip. for routing is the sum of the rainfall which
    # did not make it to storage and the percolation from the store
    current_runoff = perc + ( precip_net - precip_store)

    # runoff_history keeps track of the recent runoff values in a vector
    # that is shifted by 1 element each timestep.
    runoff_history = tt.roll(runoff_history,1)
    runoff_history = tt.set_subtensor(runoff_history[0],current_runoff)

    Q9 = 0.9* tt.dot(runoff_history,UH1)
    Q1 = 0.1* tt.dot(runoff_history,UH2)

    F = x2*(R/x3)**3.5
    R = tt.maximum(0,R+Q9+F)

    Qr = R * (1-(1+(R/x3)**4)**-0.25)
    R = R-Qr

    Qd = tt.maximum(0,Q1+F)
    Q = Qr+Qd

    # The order of the returned values is important because it must correspond
    # up with the order of kwarg list argument 'outputs_info' to theano.scan.
    return Q,S,precip_store,evap_store,perc,runoff_history,R,F,Q9,Q1

def simulate_streamflow(P,E,
                  S0,Pr0,R0,x1,x2,x3,x4,x4_limit,
                  truncate_gradient=-1):
      """Simulates streamflow over time using the model logic from GR4J as
      implemented in Theano.

      This function can be used in PyMC3 or other Theano-based libraries to
      offer up the functionality of GR4J with added gradient information.

      Parameters
      ----------
      P : 1D Theano tensor
          Time series of precipitation
      E : 1D Theano tensor
          Time series of evapotranspiration
      S0 : scalar Theano tensor
          Initial value of storage in the storage reservoir.
      Pr0 : 1D Theano tensor
          Initial levels of streamflow input. Needed for routing streamflow.
          If this is nonzero, then it is implied that there is initially
          some streamflow which must be routed in the first few timesteps.
      R0 : Initial Theano tensor
          Beginning value of storage in the routing reservoir.
      x1 : scalar Theano tensor
          Storage reservoir parameter
      x2 : scalar Theano tensor
          Catchment water exchange parameter
      x3 : scalar Theano tensor
          Routing reservoir parameters
      x4 : scalar Theano tensor
          Routing time parameter


      Returns
      -------
      streamflow : 1D Theano tensor
          Time series of simulated streamflow
      """

    forcings        = [P,E]
    state_variables = [None,S0,Pr0,R0]
    parameters      = [x1,x2,x3,x4,UH1,UH2]
    UH1,UH2 = hydrograms(x4_limit,x4)
    results,out_dict = theano.scan(fn = streamflow_step,
                          sequences = forcings,
                          outputs_info = state_variables,
                          non_sequences = parameters,
                          truncate_gradient=truncate_gradient)
    streamflow = results[0]
    return streamflow


class GR4J(Continuous):
    """ Class for wrapping GR4J hydrology model in PyMC3 with Theano"""

    def __init__(self,x1,x2,x3,x4,x4_limit,S0,R0,Pr0,sd,
                precipitation,evaporation,subsample_index=None,truncate=-1,
                *args,**kwargs):
        super(GR4J,self).__init__(*args,**kwargs)

        self.x1 = tt.as_tensor_variable(x1)
        self.x2 = tt.as_tensor_variable(x2)
        self.x3 = tt.as_tensor_variable(x3)
        self.x4 = tt.as_tensor_variable(x4)
        self.x4_limit = tt.as_tensor_variable(x4_limit)

        self.S0  = tt.as_tensor_variable(S0)
        self.R0  = tt.as_tensor_variable(R0)
        self.Pr0 = tt.as_tensor_variable(Pr0)
        self.sd  = tt.as_tensor_variable(sd)

        self.precipitation = tt.as_tensor_variable(precipitation)
        self.evaporation   = tt.as_tensor_variable(evaporation)

        # If we want the autodiff to stop calculating the gradient after
        # some number of chain rule applications, we pass an integer besides
        # -1 here.
        self.truncate   = truncate

        # If we only want to evaluate the likelihood at a few points, we can use
        # this argument to restrict the calculations.
        self.subsample_index = subsample_index

    def logp(self,observed):
        """Calculated the log likelihood of the observed streamflow given
        simulated streamflow from GR4J"""

        simulated = simulate_streamflow(self.precipitation,self.evaporation,
                                       self.S0,self.Pr0,self.R0,
                                       self.x1,self.x2,self.x3,self.x4,
                                       self.x4_limit,
                                       truncate_gradient=self.truncate)

        if self.subsample_index is not None:
            observed   = observed[subsample_index]
            simulated  = simulated[subsample_index]

        density = pm.Normal.dist(mu = simulated,sd = self.sd)
        return = tt.sum(density.logp(observed))


class GR4J_tv_x1(Continuous):
    def __init__(self,x1,x2,x3,x4,x4_limit,S0,R0,Pr0,sd,precipitation,
    evaporation,truncate,
                *args,**kwargs):
        super(GR4J_tv_x1,self).__init__(*args,**kwargs)

        self.x1 = tt.as_tensor_variable(x1)
        self.x2 = tt.as_tensor_variable(x2)
        self.x3 = tt.as_tensor_variable(x3)
        self.x4 = tt.as_tensor_variable(x4)
        self.x4_limit   = tt.as_tensor_variable(x4_limit)

        self.S0 = tt.as_tensor_variable(S0)
        self.R0 = tt.as_tensor_variable(R0)
        self.Pr0 = tt.as_tensor_variable(np.zeros(2*x4_limit-1))
        self.sd = tt.as_tensor_variable(sd)

        self.precipitation = tt.as_tensor_variable(precipitation)
        self.evaporation   = tt.as_tensor_variable(evaporation)

        # If we want the autodiff to stop calculating the gradient after
        # some number of chain rule applications, we pass an integer besides
        # -1 here.
        self.truncate   = truncate

    def get_streamflow(self,precipitation,evaporation,
                      S0,Pr0,R0,x1,x2,x3,x4,x4_limit):

        UH1,UH2 = hydrograms(x4_limit,x4)
        results,out_dict = theano.scan(fn = streamflow_step_tv_x1,
                              sequences = [precipitation,evaporation,x1],
                              outputs_info = [None,S0,None,None,None,
                                              Pr0,R0,None,None,None],
                              non_sequences = [x2,x3,x4,UH1,UH2])
        streamflow = results[0]
        return streamflow

    def logp(self,observed):
        simulated = self.get_streamflow(self.precipitation,self.evaporation,
                                       self.S0,self.Pr0,self.R0,
                                       self.x1,self.x2,self.x3,
                                       self.x4,self.x4_limit)
        sd = self.sd

        return tt.sum(pm.Normal.dist(mu = simulated,sd = sd).logp(observed))

def estimate_GR4J_tv(P,ET,Q,trace_name,truncate = 5,
    fixed_sd = None,jump_sd = 3.0):
    with pm.Model() as model:
        x1_initial = pm.Uniform('x1_initial',lower = 50, upper = 2000)
        x1_change  = pm.GaussianRandomWalk('x1_change',
        sd = jump_sd,shape = Q.shape[0])
        x1 = pm.Deterministic('x1',x1_initial+x1_change)
        x2 = pm.Uniform('x2',lower = 0.5, upper = 10)
        x3 = pm.Uniform('x3',lower = 5, upper = 100)
        x4 = pm.Uniform('x4',lower = 0.5, upper = 5)
        x4_limit = 5

        S0  = pm.Uniform('S0',lower = 100, upper = 2000)
        R0  = pm.Uniform('R0',lower = 5, upper = 100)
        Pr0 = np.zeros(x4_limit)

        if fixed_sd is not None:
            sd = fixed_sd
        else:
            sd = pm.HalfNormal('sd',sd = 0.1)

        streamflow = GR4J_tv_x1('streamflow',x1=x1,x2=x2,x3=x3,x4=x4,
                                   x4_limit=x4_limit,S0=S0,R0=R0,Pr0=Pr0,sd=sd,
                                   precipitation = P,
                                   evaporation = ET,
                                   observed = Q,truncate = truncate)
        trace = pm.backends.Text(trace_name)
        trace = pm.sample(trace = trace)
    return trace

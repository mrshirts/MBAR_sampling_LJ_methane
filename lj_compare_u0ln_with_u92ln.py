import numpy as np 
import os.path
from pymbar import MBAR
import matplotlib.pyplot as plt
from pymbar import timeseries
try:
    #Load the current installed simtk/openmm version
    import simtk.unit as units
except:
    #Use the one included in the folder
    import unitcopy as units


#################### OPTIONS ####################
savedata = True #Save/load dhdl data

Natoms = 300 #Number of atoms in system
massatom = 16.04 * (units.grams/units.mole)

################## END OPTIONS ##################

NA = units.AVOGADRO_CONSTANT_NA
kB = units.BOLTZMANN_CONSTANT_kB * NA
masstotal = Natoms*massatom

#### Reference Pressure and Temp for simulations
refP = 45 * units.atmosphere
refT = 150 * units.kelvin
refkT = kB*refT
refBeta = 1.0/refkT
refkjpermolTokT = units.kilojoules_per_mole/refkT

#dBeta for finite differences
deltarefBeta = refBeta*0.001
refTplus = 1.0/((refBeta + deltarefBeta)*kB)
refTminus = 1.0/((refBeta - deltarefBeta)*kB)
refBetaplus = refBeta + deltarefBeta
refBetaminus = refBeta - deltarefBeta

#dP for finite differences
deltarefP = refP * 0.001
refPplus = refP + deltarefP
refPminus = refP - deltarefP

#Test numpy's savez_compressed routine
try:
    savez = np.savez_compressed
except:
    savez = np.savez


################ SUBROUTINES ##################

def printFreeEnergy(DeltaF_ij, dDeltaF_ij):
    '''
    Helper function to print the free energy and its error in human readable format
 
    DeltaF_ij and dDeltaF_ij : ndarrays shape [K,K]
    '''

    nstates = DeltaF_ij.shape[0]
    print "%s DeltaF_ij:" % "complex"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % DeltaF_ij[i,j],
        print ""
    print "%s dDeltaF_ij:" % "complex"
    for i in range(nstates):
        for j in range(nstates):
            print "%8.3f" % dDeltaF_ij[i,j],
        print ""

def subsample_series(series, g_t=None, return_g_t=False):
    '''
    Subsample the timeseries either with a provided g_t or compute it directly

    series : ndarray
    '''
    if g_t is None:
        g_t = timeseries.statisticalInefficiency(series)
    state_indices = timeseries.subsampleCorrelatedData(series, g = g_t, conservative=True)
    N_k = len(state_indices)
    transfer_series = series[state_indices]
    if return_g_t:
        return state_indices, transfer_series, g_t
    else:
        return state_indices, transfer_series

class consts(object): #Class to house all constant information
    '''
    Basis function energy housing class. Keeps all basis function matricies and operations in one object
    '''

    #Converts built in units from kT to kJ/mol and back
    def _convertunits(self, converter):
        self.const_unaffected_matrix *= converter
        self.const_pv_matrix *= converter
        self.const_R_matrix *= converter
        self.const_A_matrix *= converter
        self.u_kln *= converter
        try:
            self.const_A0_matrix *= converter
            self.const_A1_matrix *= converter
            self.const_R0_matrix *= converter
            self.const_R1_matrix *= converter
            self.const_Un_matrix *= converter
        except:
            pass
        return

    #Convert kJ per mol to kT
    def dimless(self):
        if self.units:
            self._convertunits(refkjpermolTokT)
            self.units = False

    #Convert from kT to kJ/mol
    def kjunits(self):
        if not self.units:
            self._convertunits(1.0/refkjpermolTokT)
            self.units = True

    def save_consts(self, filename):
        '''
        Save energy matrix to compressed numpy .npz file reload from file later

        filename : string
            Name of file to save to disk. If no extention is passed, .npz is appended. Can use paths such as '../myfile.npz' or just filenames 'myfile.npz'
        '''
        savez(filename, u_kln=self.u_kln, const_R_matrix=self.const_R_matrix, const_A_matrix=self.const_A_matrix, const_unaffected_matrix=self.const_unaffected_matrix, const_pv_matrix=self.const_pv_matrix)
    
    def determine_N_k(self, series):
        '''
        Invoked by determine_all_N_k per series to figure out how many non-zero entries there are from the right
        '''
        npoints = len(series)
        #Go backwards to speed up process
        N_k = npoints
        for i in xrange(npoints,0,-1):
            if not np.allclose(series[N_k-1:], np.zeros(len(series[N_k-1:]))):
                break
            else:
                N_k += -1
        return N_k

    def determine_all_N_k(self, force=False):
        '''
        Determine how many non-zero samples there are in a multidimentional timeseries since different series may have a different number of samples, but the ndarray must be the same size in a given dimention
        '''
        if self.Nkset and not force:
            print "N_k is already set! Use the 'force' flag to manually set it"
            return
        self.N_k = np.zeros(self.nstates, dtype=np.int32)
        for k in xrange(self.nstates):
            self.N_k[k] = self.determine_N_k(self.u_kln[k,k,:])
        self.Nkset = True
        return

    #Set the maximum iterations and automatically cast all the energy matricies to support this new number
    #NOTE: Cannot reshape down
    def updateiter(self, iter):
        if iter > self.itermax:
            self.itermax = iter
    @property #Set the property of itermax which will also update the matricies in place
    def itermax(self):
        return self._itermax
    @itermax.setter #Whenever itermax is updated, the resize should be cast
    def itermax(self, iter):
        if iter > self.itermax:
            ukln_xfer = np.zeros([self.nstates, self.nstates, iter])
            unaffected_xfer = np.zeros([self.nstates, iter])
            pv_xfer = np.zeros([self.nstates, iter])
            un_xfer = np.zeros([self.nstates, iter])
            r0_xfer = np.zeros([self.nstates, iter])
            r1_xfer = np.zeros([self.nstates, iter])
            r_xfer = np.zeros([self.nstates, iter])
            a0_xfer = np.zeros([self.nstates, iter])
            a1_xfer = np.zeros([self.nstates, iter])
            a_xfer = np.zeros([self.nstates, iter])
            #Transfer data
            unaffected_xfer[:,:self.itermax] = self.const_unaffected_matrix
            pv_xfer[:,:self.itermax] = self.const_pv_matrix
            un_xfer[:,:self.itermax] = self.const_Un_matrix
            a_xfer[:,:self.itermax] = self.const_A_matrix
            r_xfer[:,:self.itermax] = self.const_R_matrix
            ukln_xfer[:,:,:self.itermax] = self.u_kln
            self.const_unaffected_matrix = unaffected_xfer
            self.const_pv_matrix = pv_xfer
            self.const_R_matrix = r_xfer
            self.const_A_matrix = a_xfer
            self.const_Un_matrix = un_xfer
            self.u_kln = ukln_xfer
            try:
                a0_xfer[:,:self.itermax] = self.const_A0_matrix
                a1_xfer[:,:self.itermax] = self.const_A1_matrix
                r0_xfer[:,:self.itermax] = self.const_R0_matrix
                r1_xfer[:,:self.itermax] = self.const_R1_matrix
                self.const_A0_matrix = a0_xfer
                self.const_A1_matrix = a1_xfer
                self.const_R0_matrix = r0_xfer
                self.const_R1_matrix = r1_xfer
            except:
                pass
            self.shape = self.u_kln.shape
        self._itermax = iter

    def __init__(self, nstates, file=None, itermax=1):
        '''
        Create the basis function housing object, either an empty one of size [nstates,itermax] or load from file
 
        nstates : int
            number of states to consider, static count
        file : string
            file name of the saved compressed numpy .npz object to load
        itermax : int
            maximum number of iterations, dynamicly updates all arrays when updateiter is called
        '''
        loaded = False
        self._itermax=itermax
        self.nstates=nstates
        self.Nkset = False
        if file is not None:
            try:
                ukln_file = np.load(file)
                self.u_kln = ukln_file['u_kln']
                self.const_R_matrix = ukln_file['const_R_matrix'] 
                self.const_A_matrix = ukln_file['const_A_matrix']
                self.const_unaffected_matrix = ukln_file['const_unaffected_matrix']
                self.const_pv_matrix = ukln_file['const_pv_matrix']
                self._itermax = self.u_kln.shape[2]
                self.determine_all_N_k()
                self.Nkset = True
                loaded = True
            except: 
                pass
        if not loaded:
            self.const_unaffected_matrix = np.zeros([self.nstates,self.itermax])
            self.const_pv_matrix = np.zeros([self.nstates,self.itermax])
            self.const_Un_matrix = np.zeros([self.nstates,self.itermax])
            self.const_R0_matrix = np.zeros([self.nstates,self.itermax])
            self.const_R1_matrix = np.zeros([self.nstates,self.itermax])
            self.const_R_matrix = np.zeros([self.nstates,self.itermax])
            self.const_A0_matrix = np.zeros([self.nstates,self.itermax])
            self.const_A1_matrix = np.zeros([self.nstates,self.itermax])
            self.const_A_matrix = np.zeros([self.nstates,self.itermax])
            self.u_kln = np.zeros([self.nstates,self.nstates,self.itermax])
            self.itermax = itermax
            self.N_k = np.ones(self.nstates) * self.itermax
        self.shape = self.u_kln.shape
        self.units = True

def epsi_sig_from_TPstar(Tstar,Pstar, T=refT, P=refP):
    '''
    Compute the epsilon and sigma from a Tstar and Pstar at T and P
   
    Tstar : float
        Dimensionless temperature of LJ Fluid
    Pstar : float
        Dimensionless pressure of LJ Fluid
    T     : simtk.unit Quantity in temperature (prefered kelvin), default refT
        Temperature at which to compute epsilon and sigma
    P     : simtk.unit Quantity in pressure (prefered atm), default refP
        Pressure at which to compute epsilon and sigma
    '''
    epsi = kB * T / Tstar
    epsi_not_per_mol = epsi / NA
    #Order is important in python 2.6, A*B != B*A
    sigma = (epsi_not_per_mol / P * Pstar)**(1.0/3)
    return epsi/units.kilojoules_per_mole, sigma/units.nanometer

def TPstar_from_epsi_sig(epsi, sigma, T=refT, P=refP):
    '''
    Compute the dimensionless temperature and pressure for an LJ fluid

    epsi  : simtk.unit Quantity in energy per mole
        LJ epsilon
    sigma : simtk.unit Quantity in length
        LJ sigma
    T     : simtk.unit Quantity in temperature (prefered kelvin), default refT
        Temperature at which to compute epsilon and sigma
    P     : simtk.unit Quantity in pressure (prefered atm), default refP
        Pressure at which to compute epsilon and sigma
    '''
    #Assumes kJ/mol and nm
    if type(epsi) == units.Quantity:
        epsi = epsi.in_units_of(units.kilojoules_per_mole)
    else:
        try:
            epsi *= units.kilojoules_per_mole
        except:
            #Python 2.6 fix
            epsi = units.Quantity(value=epsi, unit=units.kilojoules_per_mole)
    epsi_not_per_mol = epsi / NA
    if type(sigma) == units.Quantity:
        sigma = sigma.in_units_of(units.nanometer)
    else:
        try:
            sigma *= units.nanometer
        except:
            sigma = units.Quantity(value=sigma, unit=units.nanometer)
    Tstar = kB*T/epsi
    Pstar = P*sigma**3/epsi_not_per_mol
    return Tstar, Pstar
    
class ReducedLJSystem(object):
    '''
    Helper class to compute T, P, Tstar, and Pstar from an object instead of passing epsilon/sigma to separate functions
    '''
    def Pstar(self, P):
        #Assumes atmospheres for units
        if type(P) == units.Quantity:
            P = P.in_units_of(units.atmosphere)
        else:
            try:
                P *= units.atmosphere
            except:
                P = units.Quantity(value=P, unit=units.atmosphere)
        Pstar = P * self.sigma**3 / (self.epsilon/NA)
        return Pstar

    def Tstar(self, T):
        #Assumes Kelvin for units
        if type(T) == units.Quantity:
            T = T.in_units_of(units.kelvin)
        else:
            try:
                T *= units.kelvin
            except:
                T = units.Quantity(value=T, unit=units.kelvin)
        Tstar = T * kB / self.epsilon
        return Tstar

    def BetaStar(self, Beta):
        # beta = 1 / kB T $ 
        # beta* = 1 / T* = epsilon / kB T = epsilon * beta 
        # assume beta in mole/joule
        if type(Beta) == units.Quantity:
            Beta = Beta.in_units_of(units.mole/units.joule)
        else:
            try:
                Beta *= units.joules
            except:
                Beta = units.Quantity(value=Beta,units=units.mole/units.joule)
        BetaStar = Beta * self.epsilon
        return BetaStar

    def T(self, Tstar):
        Tout = (self.epsilon / kB)*Tstar
        return Tout.in_units_of(units.kelvin)

    def P(self, Pstar):
       Pout = (self.epsilon_not_per_mol/self.sigma**3) * Pstar
       return Pout.in_units_of(units.atmosphere)

    def beta(self, BetaStar):
        Betaout = (BetaStar)/self.epsilon
        return Betaout.in_units_of(units.mole/units.joules)

    def __init__(self, epsilon=1.386685989 * units.kilojoules_per_mole, sigma=3.678*units.angstrom):
        '''
        Reference (Default) System is UA Methane
        Parameters from Stat. Mech. for Thermophysical Property Calculation, Karl J. Johnson
        '''
        self.epsilon = epsilon
        self.epsilon_not_per_mol = self.epsilon/NA
        self.sigma = sigma.in_units_of(units.nanometer)

def findBase10Exponent(x):

    #Find the integer exponent of the base 10 float (i.e. scientific notation value)
    #Known issue: if 0 is passed in, -inf is returned
    return np.floor(np.log10(x)).astype(np.int)


def fCnew(epsi, sig, sig_power, topology_precision = 5):

    '''
    Convert the calculation of C12/C6 into the same precision as GROMACS
    
    sig_power : Int or Float
        Exponent on sigma when computing, usually 12 or 6
    topology_precision : Int
        User set positive integer number equal to the number of significant figures in the GROMACS topology file to the right of the decimal on C6 and C12, in scienticic notation
    '''

    #Compute C, cast to array if not already
    C = np.array(4*epsi*sig**sig_power)
    #Calculate exponent of scientific notation
    #Loop through each element since e in 10^e may be different for each element of C
    for element in np.nditer(C, op_flags=['readwrite']): 
        E = findBase10Exponent(element)
        element[...] = np.around(element, -E+topology_precision)
    return C

def build_ukln_data(nstates, Tstar_sample, Pstar_sample, load_ukln):
    '''
    Build the u_kln matrix and compute the basis functions from the generated data
    Assumes fixed sample size, spcific file names, and specific file structure of the GROMACS outputs

    TODO:
    -Generalize inputs
    -Convert data to more compressed format instead of many .xvg energy files

    Inputs: 
    nstates : int
        Number of states to draw data from
    Tstar_sample : 1-D array of floats size [nstates]
        Dimensionless temperature that was sampled. E.g. State 0 is index 0
    Pstar_sample : 1-D array of floats size [nstates]
        Dimensionless pressure that was sampled. E.g. State 0 is index 0
   
    Returns:
    consts : object
        Instance of consts class constructed from this data
    '''

    #Reference sampled States

    refK_A = 0 #Location of minimum epsilon/sigma
    refK_B = 16 #location of maximum epsilon/sigma

    #This will be all fixed Pstar for now
    epsi_sample, sig_sample = epsi_sig_from_TPstar(Tstar_sample, Pstar_sample)

    #Set limits
    epsiA = epsi_sample[refK_A]
    epsiB = epsi_sample[refK_B]
    sigA = sig_sample[refK_A]
    sigB = sig_sample[refK_B]

    #generate sample length
    g_en_start = 23 #Row where data starts in g_energy output
    g_en_energy = 1 #Column where the energy is located
    g_en_pv     = 2 #Column where pv is located
    niterations_max = 30001

    #Min and max sigmas:
    fC12 = lambda epsi,sig: fCnew(epsi, sig, 12)
    fC6 = lambda epsi,sig: fCnew(epsi, sig, 6)

    C12_delta = fC12(epsiA, sigA) - fC12(epsiB, sigB)
    C6_delta = fC6(epsiA, sigA) - fC6(epsiB, sigB)
    C12_delta_sqrt = fC12(epsiA, sigA)**.5 - fC12(epsiB, sigB)**.5
    C6_delta_sqrt = fC6(epsiA, sigA)**.5 - fC6(epsiB, sigB)**.5

    #Set up lambda calculation equations
    #lambda = [(CzCz)**(2m) - (CaCa)**(2m)]/[(CbCb)**(2m) - (CaCa)**(2m)]
    #since geometric mixing rules, 2m = 2*(1/2) = 1, just use flamC6 and flamC12
    flamC6 = lambda epsi, sig: (fC6(epsi, sig) - fC6(epsiB, sigB))/C6_delta
    flamC12 = lambda epsi, sig: (fC12(epsi, sig) - fC12(epsiB, sigB))/C12_delta

    #Set the lambda of each state
    lamC12 = flamC12(epsi_sample, sig_sample)
    lamC6 = flamC6(epsi_sample, sig_sample)

    #Try to load u_kln
    subsampled = np.zeros([nstates],dtype=np.bool)
    if load_ukln and os.path.isfile('data/ardata/ar_ukln_consts_n%i.npz'%nstates):
        energies = consts(nstates, file='data/ardata/ar_ukln_consts_n%i.npz'%nstates)
    else:
        #Initial u_kln
        energies = consts(nstates)
        g_t = np.zeros([nstates])
        #Read in the data
        for k in xrange(1):
            print "Importing ar = %02i" % k
            energy_dic = {'full':{}, 'rep':{}}
            #Try to load the subsampled filenames
            try:
                energy_dic['null'] = open('data/ar%s/prod/subenergy%s_null.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the null energies (unaffected) of the K states
                for l in xrange(nstates):
                    energy_dic['full']['%s'%l] = open('data/ar%s/prod/subenergy%s_%s.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the full energies for each state at KxL
                    if l == refK_A or l == refK_B:
                        energy_dic['rep']['%s'%l] = open('data/ar%s/prod/subenergy%s_%s_rep.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the repulsive energies at 0, nstates-1, and K
                iter = len(energy_dic['null'])
                subsampled[k] = True
                # Set the object to iterate over, since we want every frame of the subsampled proces, we just use every frame
                frames = xrange(iter)
            except: #Load the normal way
                energy_dic['null'] = open('data/ar%s/prod/energy%s_null.xvg' %(k,k),'r').readlines()[g_en_start:] #Read in the null energies (unaffected) of the K states
                for l in xrange(nstates):
                    energy_dic['full']['%s'%l] = open('data/ar%s/prod/energy%s_%s.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the full energies for each state at KxL
                    if l == refK_A or l == refK_B:
                        energy_dic['rep']['%s'%l] = open('data/ar%s/prod/energy%s_%s_rep.xvg' %(k,k,l),'r').readlines()[g_en_start:] #Read in the repulsive energies at 0, nstates-1, and K
                iter = niterations_max
                #Subsample
                tempenergy = np.zeros(iter)
                temppv = np.zeros(iter)
                for frame in xrange(iter):
                    tempenergy[frame] = float(energy_dic['full']['%s'%k][frame].split()[g_en_energy])
                    temppv[frame]     = float(energy_dic['full']['%s'%k][frame].split()[g_en_pv])
                frames, temp_series, g_t[k] = subsample_series(tempenergy, return_g_t=True)
                print "State %i has g_t of %i" % (k, g_t[k])
                iter = len(frames)
            #Update iterations if need be
            energies.updateiter(iter)
            #Fill in matricies
            n = 0
            for frame in frames:
                #Unaffected state
                energies.const_Un_matrix[k,n] = float(energy_dic['null'][frame].split()[g_en_energy])
                energies.const_pv_matrix[k,n] = float(energy_dic['null'][frame].split()[g_en_pv])
                #Isolate the data
                for l in xrange(nstates):
                    energies.u_kln[k,l,n] = float(energy_dic['full']['%s'%l][frame].split()[g_en_energy]) #extract the kln energy, get the line, split the line, get the energy, convert to float, store
                #Repulsive terms: 
                #R0 = U_rep[k,k,n] + dhdl[k,0,n] - Un[k,n]
                energies.const_R0_matrix[k,n] = float(energy_dic['rep']['%s'%(refK_B)][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                #R1 = U_rep[k,k,n] + dhdl[k,-1,n] - Un[k,n]
                energies.const_R1_matrix[k,n] = float(energy_dic['rep']['%s'%(refK_A)][frame].split()[g_en_energy]) - energies.const_Un_matrix[k,n]
                energies.const_R_matrix[k,n] = energies.const_R1_matrix[k,n] - energies.const_R0_matrix[k,n]
                #Attractive term
                #u_A = U_full[k,n] - constR[k,n] - const_unaffected[k,n]
                energies.const_A0_matrix[k,n] = energies.u_kln[k,refK_B,n] - energies.const_R0_matrix[k,n] - energies.const_Un_matrix[k,n]
                energies.const_A1_matrix[k,n] = energies.u_kln[k,refK_A,n] - energies.const_R1_matrix[k,n] - energies.const_Un_matrix[k,n]
                energies.const_A_matrix[k,n] = energies.const_A1_matrix[k,n] - energies.const_A0_matrix[k,n]
                #Finish the total unaffected term
                #Total unaffected = const_Un + U0 + pV = pv + const_Un + (U_full[k,0,n] - const_Un) = pv + U_full[k,0,n]
                energies.const_unaffected_matrix[k,n] = energies.u_kln[k,refK_B,n] + energies.const_pv_matrix[k,n]
                energies.u_kln[k,:,n] += energies.const_pv_matrix[k,n] #add in the PV work, this will mostly cancel out, but for throughness
                n += 1
        energies.determine_all_N_k()
        constname = 'data/ardata/ar_ukln_consts_n%i.npz'%nstates
        if load_ukln and not os.path.isfile(constname):
            energies.save_consts(constname)
    #Sanity check

    sanity_kln = np.zeros(energies.u_kln.shape)
    for l in xrange(nstates):
        sanity_kln[:,l,:] = lamC12[l]*energies.const_R_matrix + lamC6[l]*energies.const_A_matrix + energies.const_unaffected_matrix
    del_kln = np.abs(energies.u_kln - sanity_kln)
    del_tol = 1 #in kJ per mol

    print "Max Delta: %f" % np.nanmax(del_kln)
    for l in xrange(nstates):
        print "Max for state %2d %10.5g" % (l,np.nanmax(del_kln[l,l,:]))
    #MRS: these almost certainly have nonnegligible weights, since they are sampled.  This seems big.
   
    if np.nanmax(del_kln) > 0.2: #Check for numeric error
        #Double check to see if the weight is non-zero.
        #Most common occurance is when small particle is tested in large particle properties
        #Results in energies > 60,000 kj/mol, which carry 0 weight to machine precision
        nonzero_weights = np.count_nonzero(np.exp(-energies.u_kln[np.where(del_kln > .2)] * refkjpermolTokT))
        if nonzero_weights != 0:
            print "and there are %d nonzero weights! Stopping execution" % nonzero_weights
        else:
            print "but these carry no weight and so numeric error does not change the answer"

    # mbar takes in (h/k_B T).  since h = epsilon h* , then mbar should take in h* (epsilon/k_B T)
    # which is the same thing as h*/T*.

    # validate the units here. They are in kJ/mol at this point, and
    # are h+pv at a constant T and P.  energies.dimless takes all
    # units, assignes them kJ/mol, then divides by k_B T, the
    # reference temperature.  So this information is then equivalent
    # to h*/T*, where the T* has been computed from the sigma and epsilon.
    
    # note that const_pv_matrix is also transformed into p*v*/T*

    energies.dimless()

    Nk = energies.N_k
    #MRS: Interesting.  At constant pressure, as sigma goes down the average PV goes up, and then starts going down.
    # data from sampled states. 
    print "     ave PV        sigma         eps     eps/sig^3"
    for i in range(len(Nk)):
        print "%12.5f %12.5f %12.5f %12.5f" % (np.average(energies.const_pv_matrix[i,:Nk[i]]), sig_sample[i], epsi_sample[i],epsi_sample[i]/(sig_sample[i])**3)

    #Expand energies class to compute energy
    #Function returns dimensionless if energies.dimless() is set kJ/mol if not
    energies.U = lambda epsi, sig: flamC12(epsi, sig)*energies.const_R_matrix  + flamC6(epsi, sig)*energies.const_A_matrix  + energies.const_unaffected_matrix

    return energies 

def build_MBAR(nstates, energies):
    '''
    Construct the MBAR object for analysis. 
    Attempts to fast-solve the self-consistant MBAR free energies by reading f_k from file if possible.
    Will read the last available f_k file from fewer states if more states were added to the dataset

    nstates : int
        number of states in energies
    energies : consts instance
        Takes the u_kln and the N_k from the consts object and passes them into MBAR. Forces energies to be dimless.
    '''
    if energies.units:
        energies.dimless()
    #Load subsequent f_ki
    f_ki_loaded = False
    state_counter = nstates
    while not f_ki_loaded and state_counter != 0:
        #Load the largest f_ki available to fast-solve
        try:
            f_ki_load = np.load('data/ardata//ar_f_k_{myint:{width}}.npy'.format(myint=state_counter, width=len(str(state_counter))))
            f_ki_loaded = True
            f_ki_n = state_counter
        except:
            pass
        state_counter -= 1
    try:
        if nstates >= f_ki_n:
            draw_ki = f_ki_n
        else:
            draw_ki = nstates
        #Copy the loaded data
        f_ki = np.zeros(nstates)
        f_ki[:draw_ki] = f_ki_load[:draw_ki] #- Try without loading the data -RAM
        mbar = MBAR(energies.u_kln, energies.N_k, verbose = True, initial_f_k=f_ki, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':True}}], subsampling=1)
    except:
        mbar = MBAR(energies.u_kln, energies.N_k, verbose = True, subsampling_protocol=[{'method':'L-BFGS-B','options':{'disp':True}}], subsampling=1)
    if not f_ki_loaded or f_ki_n != nstates:
        try:
            np.save_compressed('data/ar_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), mbar.f_k)
        except:
            np.save('data/ar_f_k_{myint:{width}}.npy'.format(myint=nstates, width=len(str(nstates))), mbar.f_k)

    return mbar

def estimate_rho_hvap(T, P, epsilon, sigma, energies, mbar):
    '''
    Estimate density and heat of vaporization as a function of (a possible range of T and P) and fixed epsilon, sigma (which could be arrays as well)

    T       : simtk.unit Quantity, either single or 1-D array in unit of temperature
        The temperature or temperatures to estimate rho and hvap at
    P       : simtk.unit Quantity, single value unit of pressure
        The pressure to estimate rho and hvap at. For now, this function only computes at single value of P
    epsilon : simtk.unit Quantity, either single or 1-D array in unit of energy/mole
        epsilons to compute rho and hvap at, must be same size as sigma
    sigma   : simtk.unit Quantity, either single or 1-D array in unit of length
        sigmas to compute rho and hvap at, must be the same size as epsilon
    energies: consts instance
        Instance of the consts object which has all the sampled data input
    mbar    : MBAR instance
        MBAR instance used to compute thermodynamic properties from, must be constructed from the same data that is contained in energies

    NOTE: If no units are on T, P, epsilon, or sigma; then units are assumed.
    '''

    #Check units, assign defaults, alert operator
    if type(T) != units.Quantity:
        T = units.Quantity(T, unit=units.kelvin)
        print("T had no units so kelvin was assmed")
    if type(P) != units.Quantity:
        P = units.Quantity(P, unit=units.atmosphere)
        print("P had no units so atmosphere was assmed")
    if type(epsilon) != units.Quantity:
        epsilon = units.Quantity(epsilon, unit=units.kilojoules_per_mole)
        print("Epsilon had no units so kJ/mol was assmed")
    if type(sigma) != units.Quantity:
        sigma = units.Quantity(sigma, unit=units.nanometer)
        print("Sigma had no units so nm was assmed")

    NT = len(T)
    
    #Cast the T, P, epsi, sig in to T* P*
    # epsilon/sigma and Tstar could both be arrays. We want to be able to reshape the outputs
    # because they have to be passed into u_kln as arrays
    NE = len(epsilon)
    NS = len(sigma)
    if NE != NS:
        print "number of epsilon and sigma must be equal!"
        return

    Tstar_matrix, Pstar_matrix = TPstar_from_epsi_sig(epsilon, sigma, T=T, P=P)

    Tstar = Tstar_matrix.reshape(NT*NE,1)
    Pstar = Pstar_matrix.reshape(NT*NS,1)

    # Tstar is of the form (all T's with E1,all T's with E2, All T's with E3, . . all T's with EN)
    # Pstar is of the form (all T's with S1,all T's with S2, All T's with S3, . . all T's with SN)

    Nall = NT*NE

    #Cast T* and P* back into the reference T and P to be used with energies/mbar
    epsimbar, sigmbar = epsi_sig_from_TPstar(Tstar,Pstar)

    #Set up perturbed energies
    u_kln_pert = np.zeros([energies.nstates,Nall,energies.itermax])
    for l in xrange(Nall):
        epsil = epsimbar[l]
        sigl  = sigmbar[l]
        u_kln_pert[:,l,:] = energies.U(epsil, sigl)

    # Get free energies relative to the reference state (the index l=0)
    (Deltaf_ij, dDeltaf_ij) = mbar.computePerturbedFreeEnergies(u_kln_pert, uncertainty_method='svd-ew')
    #Compute the <u> which in this case is the reduced enthalpy of the liquid
    (hu_array, dhu_array) = mbar.computeExpectations(u_kln_pert, u_kn=u_kln_pert, state_dependent=True)
    # what have we computed?  We have computed h* = u* + p*v* in reduced units at a series of different T* and P*  
    # now, we want to convert that to H at specified values of T, sig, and eps.

    # first, we want to get back out of reduced enthalpy to actual enthalpy in reduced units (tricky!) 
    # the simulations are all treated as occurring at the same Tref (150 K), and Pref (45 atm)
    # Assign units, kj/mol for hu and DeltafF

    # now we need to reshape the hu back out
    # hu = is h*/T*. So first, get h*. 

    # hu is matrix; we need to multiply arraywise . . . hu_matrix and Tstar_T are actually already aligned!
    hu_array = hu_array * Tstar.T
    dhu_array= dhu_array * Tstar.T

    # now the enthalpy is in reduced units.  Get out the actual enthalpy.  
    # need to align the matrices. 

    epsilon_array = np.repeat(epsilon,NT,axis=1).reshape(1,NT*NE)
    hu_array = hu_array*epsilon_array
    dhu_array = dhu_array*epsilon_array

    # reset to a matrix. Units get obliterated by matrix algebra, need to add back here.
    hu = hu_array.reshape(NE,NT)*epsilon.unit
    dhu = dhu_array.reshape(NE,NT)*epsilon.unit

    # enthalpy of vaporization
    hu = -hu.in_units_of(units.kilojoules/units.mole)/Natoms
    dhu = dhu.in_units_of(units.kilojoules/units.mole)/Natoms

    ## now handle volumes.

    # const_pv_matrix is in units of refT
    v_kn = energies.const_pv_matrix
    (vu_array, dvu_array) = mbar.computeExpectations(energies.const_pv_matrix, u_kn=u_kln_pert, state_dependent=False)
    # const_pv_matrix was in units of kJ/mol, then was put in reduced units, so convert back to kJ/mol
    # convert reduced p*v*/T* into p*v* (reduced units) 

    vu_array = vu_array*Tstar.T
    dvu_array = dvu_array*Tstar.T

    # now, convert p*v* into real p*
    vu_array = vu_array*epsilon_array
    dvu_array = dvu_array*epsilon_array

    # now divide out the pressue
    vu_array = vu_array*1.0/(np.repeat(P,NE)*P.unit)
    dvu_array = dvu_array*1.0/(np.repeat(P,NE)*P.unit)

    # reshape, and add back in units
    u = vu_array.unit
    vu = vu_array.value_in_unit(u).reshape(NE,NT)*epsilon.unit*u
    dvu = dvu_array.value_in_unit(u).reshape(NE,NT)*epsilon.unit*u

    # now convert to liters/mol.
    vu = vu.in_units_of(units.liter/units.mole)
    dvu = dvu.in_units_of(units.liter/units.mole)

    # leave Deltaf_ij and dDeltaf_ij in units of G*/T* for now

    # convert volume of Natom into volume in proper units.

    rho = masstotal / vu
    # rho = 1/v, drho = mass/v^2 dv = (mass/v)(dv/v) = rho(dv/v)
    drho = rho*(dvu/vu)

    return Deltaf_ij, dDeltaf_ij, hu, dhu, rho.in_units_of(units.kilogram/units.meter**3), drho.in_units_of(units.kilogram/units.meter**3)

def ideal_gas_enthalpy(N,T):

    return (3.0/2)*N*T*kB + N*kB*T

def execute(nstates, Tstar_sample, Pstar_sample,load_ukln):
    '''
    Run main computation/
    '''

    uameth = ReducedLJSystem()

    Tstar_range = Tstar_sample.copy()
    Pstar_range = Pstar_sample.copy()
    Nparm = len(Tstar_range)
    epsi_range, sig_range = epsi_sig_from_TPstar(Tstar_range, Pstar_range)

    #Finite Differences setup
    #determine the effective epsilon and sigma to reweight at for deltaBeta and deltaP for finite differences
    #note that the dT corresponds to an even difference in beta, not T. 
    #May not be the right way to compute these

    dTrefTstarplus, dTrefPstarplus = TPstar_from_epsi_sig(epsi_range, sig_range, T=refTplus)
    dTrefTstarminus, dTrefPstarminus = TPstar_from_epsi_sig(epsi_range, sig_range, T=refTminus)

    dPrefTstarplus, dPrefPstarplus = TPstar_from_epsi_sig(epsi_range, sig_range, P=refPplus)
    dPrefTstarminus, dPrefPstarminus = TPstar_from_epsi_sig(epsi_range, sig_range, P=refPminus)

    # index -/+ matricies for doing derivative

    dTderivepsi_range = np.zeros((2,Nparm))
    dTderivsig_range = np.zeros((2,Nparm))
    dPderivepsi_range = np.zeros((2,Nparm))
    dPderivsig_range = np.zeros((2,Nparm))

    dTderivepsi_range[0,:], dTderivsig_range[0,:] = epsi_sig_from_TPstar(dTrefTstarminus, dTrefPstarminus)
    dTderivepsi_range[1,:], dTderivsig_range[1,:] = epsi_sig_from_TPstar(dTrefTstarplus, dTrefPstarplus)
    dPderivepsi_range[0,:], dPderivsig_range[0,:] = epsi_sig_from_TPstar(dPrefTstarminus, dPrefPstarminus)
    dPderivepsi_range[1,:], dPderivsig_range[1,:] = epsi_sig_from_TPstar(dPrefTstarplus, dPrefPstarplus)
    
    energies = build_ukln_data(nstates, Tstar_sample, Pstar_sample,load_ukln)

    ##################################################
    ############### END DATA INPUT ###################
    ##################################################

    mbar = build_MBAR(nstates, energies)
    
    ######## #Begin computing free energies ##########
    
    #Load data from file or compute
    #if not (os.path.isfile('data/ardata/ns%iNp%i.npz' % (nstates, Nparm)) and savedata): #nand gate
    if 1:
        #Create numpyp arrys: epsi, sig
        #Populate energies
        u_kln_pert = np.zeros([nstates,5*Nparm,energies.itermax]) #Account for and the derivatives for dT, derivatives for dP
        #G0 =  0:Nparm
        #dT- = 1*Nparm:2*Nparm
        #dT+ = 2*Nparm:3*Nparm
        #dP- = 3*Nparm:4*Nparm
        #dP+ = 4*Nparm:5*Nparm
        for l in xrange(Nparm):
            epsi = epsi_range[l]
            sig = sig_range[l]
            dTepsiminus= dTderivepsi_range[0,l]
            dTsigminus = dTderivsig_range[0,l]
            dTepsiplus = dTderivepsi_range[1,l]
            dTsigplus  = dTderivsig_range[1,l]
            dPepsiminus= dPderivepsi_range[0,l]
            dPsigminus = dPderivsig_range[0,l]
            dPepsiplus = dPderivepsi_range[1,l]
            dPsigplus  = dPderivsig_range[1,l]

            u_kln_pert[:,l+0*Nparm,:] = energies.U(epsi,sig)
            u_kln_pert[:,l+1*Nparm,:] = energies.U(dTepsiminus,dTsigminus)
            u_kln_pert[:,l+2*Nparm,:] = energies.U(dTepsiplus,dTsigplus)  
            u_kln_pert[:,l+3*Nparm,:] = energies.U(dPepsiminus,dPsigminus)
            u_kln_pert[:,l+4*Nparm,:] = energies.U(dPepsiplus,dPsigplus)  

        # all in reduced units
        # Get free energies relative to the reference state (the index l=0)
        (Deltaf_ij, dDeltaf_ij) = mbar.computePerturbedFreeEnergies(u_kln_pert, uncertainty_method='svd-ew')
        #Compute the <u> which in this case is the reduced enthalpy of the liquid
        (hu, dhu) = mbar.computeExpectations(u_kln_pert, u_kn=u_kln_pert, state_dependent=True)
        v_kn = np.zeros(np.shape(energies.const_pv_matrix),float)
        for l, pl in enumerate(Pstar_sample):
            v_kn[l,:] =  energies.const_pv_matrix[l,:]/pl
        (vu, dvu) = mbar.computeExpectations(v_kn, u_kn=u_kln_pert, state_dependent=False)

        if savedata:
            if not os.path.isdir('data/ardata'):
                os.makedirs('data/ardata') #Create folder
            savez('data/ardata/ns%iNp%i.npz' % (nstates, Nparm), Deltaf_ij=Deltaf_ij, dDeltaf_ij=dDeltaf_ij, hu=hu, dhu=dhu, vu=vu, dvu = dvu ) #Save file
    else: #Load the files instead

        Deltaf_file = np.load('data/ardata/ns%iNp%i.npz' % (nstates, Nparm))
        Deltaf_ij = Deltaf_file['Deltaf_ij']
        dDeltaf_ij = Deltaf_file['dDeltaf_ij']

    ###### Derivatives around the free energy, this is its own block to prevent reworking row logic 

    ###############################################
    ######### END FREE ENERGY CALCULATIONS ########
    ###############################################

    #Dimensionless quantities don't need these conversions

    #DeltaF_ij *= uameth.epsilon
    #dDeltaF_ij *= uameth.epsilon
    #Hu *= uameth.epsilon
    #dHu *= uameth.epsilon

    ##### Python 2.7 version #####
    #unitDeltaF_ij = DeltaF_ij * units.kilocalories_per_mole
    #unitdDeltaF_ij = dDeltaF_ij * units.kilocalories_per_mole

    ##### Python 2.6 version #####
    #unitDeltaF_ij = units.Quantity(value=DeltaF_ij, unit=units.kilocalories_per_mole)
    #unitdDeltaF_ij = units.Quantity(value=dDeltaF_ij, unit=units.kilocalories_per_mole)

    #Convert Tstar and Pstar into T and P for uamethane

    Betastar_range = 1/Tstar_range
    Trange = uameth.T(Tstar_range) 
    Beta_range = 1.0/(kB * Trange)

    # dh = dG/dbeta

    deltarefBetastar = uameth.BetaStar(deltarefBeta)

    # try to do all the calculations in reduced units first, then convert back to units.

    #Compute enthalpy. first derivative of f with respect to beta.
    Hstar = (Deltaf_ij[Nparm:2*Nparm, Nparm] - Deltaf_ij[2*Nparm:3*Nparm, Nparm])/(2*deltarefBetastar)

    H = uameth.epsilon*Hstar

    # double checking

    div = (len(Tstar_range)-1)/(len(Tstar_sample)-1)
    print "*************"
    print "the following should be approximately equal:"
    print "1) average sampled u"
    print "2) reevaluated u"
    print "3) reweighted u"
    print "4) h by free energy derivative"
    for l, T in enumerate(Tstar_sample):
        h1 = np.average(energies.u_kln[l,l,:mbar.N_k[l]])
        h2 = np.average(u_kln_pert[l,l*div,:mbar.N_k[l]])
        h3 = hu[l*div]
        h4 = Hstar[l*div]
        print "%12.5f %12.5f %15.8f %15.8f %15.8f %15.8f" % (T,Pstar_sample[l],h1,h2,h3,h4) 
    print "*************"

    Hu = uameth.epsilon*hu[0:Nparm]
    dHu = uameth.epsilon*dhu[0:Nparm]

    #MRS: H and Hu should agree, do not.

    #Compute heat capacities. Second derivative of F, derivative of U.  First dimensionless
    #Cp = (-kB*(Beta_range**3) *(unitDeltaF_ij[Nparm, Nparm:2*Nparm] - 2*unitDeltaF_ij[Nparm, 0*Nparm:1*Nparm] + unitDeltaF_ij[Nparm, 2*Nparm:3*Nparm])/deltarefBeta**2) / (units.kilocalories_per_mole/units.kelvin)

    # Cpstart = beta
    Cpstar = (Betastar_range)**3*(Deltaf_ij[Nparm:2*Nparm, Nparm] - 2*Deltaf_ij[0*Nparm:1*Nparm, Nparm] + Deltaf_ij[2*Nparm:3*Nparm, Nparm])/(deltarefBetastar**2)
    # cp = dH/dT = epsilon dH* / (epsilon dT* / kB ) = kB * dH*/dT*

    Cp = kB * Cpstar

    # figure out how to propagate Cp

    # try as Temperature derivative of H? Can't, because not equally spaced in T.  However, dH/dT = dH/dbeta dbeta/dT = 
    # dH/dbeta -1/T^2 = -beta^2 dH/dbeta
    Cpstar2 = - (Betastar_range)**2 * (hu[2*Nparm:3*Nparm] - hu[1*Nparm:2*Nparm])/(2*deltarefBetastar)

    Cp2 = kB * Cpstar2

    #MRS: Cp and Cp2 should be equal as well (are not)

    #This isnt exactly a fair calulation since the choice of K strongly influences the error
    #dCp = (-kB*(Beta_range**3)*(unitdDeltaF_ij[Nparm, Nparm:2*Nparm]**2 + (2*unitdDeltaF_ij[Nparm, 0*Nparm:1*Nparm])**2 + unitDeltaF_ij[Nparm, 2*Nparm:3*Nparm]**2)**(0.5)/deltarefBeta**2) /(units.kilocalories_per_mole/units.kelvin)

    Prange = uameth.P(Pstar_range)

    #Compute densities
    deltarefPstar = uameth.Pstar(deltarefP)

    # I think this actually might need to be dG/d(betaP)?

    # central difference
    Vstar = (Deltaf_ij[4*Nparm:5*Nparm, 3*Nparm] - Deltaf_ij[3*Nparm:4*Nparm,3*Nparm])/(2*deltarefPstar)

    print "*************"
    print "the following should be approximately equal:"
    print "1) average volume measured"
    print "2) reweighted v"
    print "3) v by free energy derivatives"
    for l, P in enumerate(Pstar_sample):
        v1 = np.average(energies.const_pv_matrix[l,:mbar.N_k[l]]/P)
        v2 = vu[div*l]
        v3 = Vstar[div*l]
        print "print %12.5f %12.5f %15.8f %15.8f %15.8f" % (Tstar_sample[l], Pstar_sample[l], v1, v2, v3)
    print "*************"

    #rho = (masstotal) / (Vstar*uameth.sigma**3)
    #rho = rho / (units.gram/units.milliliter)

    rhou = (masstotal) / (NA*vu[:Nparm]*(uameth.sigma/units.nanometer)**3)*1e21
    rhou = rhou / (units.gram) 

    #df = |df/du| du  d(1/x) = x^(-2) dx = dx/x * x(-1)
    drhou = rhou * (dvu[:Nparm]/vu[:Nparm]) 

    #MRS:rho and rhou should agree -- do not

    #Ideal Gas Enthalpy from stat. mech.
    IGEnthalpy = ideal_gas_enthalpy(Natoms, Trange)
    #Enthalpy of vaporization from Brittany's notes
    Hvap = (IGEnthalpy + kB*Trange - Hu)/1000  #MRS: bad units kludge

#    f,(a,b,c) = plt.subplots(3,1)
#    a.plot(Trange/units.kelvin, Hvap, '-b', linewidth=2)
#    a.plot(Trange/units.kelvin, Hvap+dHu/1000, '--b', linewidth=2)  # bad units kludge
#    a.plot(Trange/units.kelvin, Hvap-dHu/1000, '--b', linewidth=2)  # bad units kludge
#    a.set_xlabel("Temperature in K", fontsize=14)
#    a.set_ylabel("Enthalpy of Vaporization\nin kJ/mol", fontsize=14)
#    a.set_title(r'$\Delta H_{\mathrm{vap}}$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere)) 
#
#    b.plot(Trange/units.kelvin, Cp2, '-b', linewidth=2)
#    #b.plot(Trange/units.kelvin, Cp2+dCp, '--b', linewidth=2)
#    #b.plot(Trange/units.kelvin, Cp2-dCp, '--b', linewidth=2)
#    b.set_xlabel("Temperature in K", fontsize=14)
#    b.set_ylabel("Heat Capacity (const P)\nin kJ/(mol K)", fontsize=14)
#    b.set_title(r'$C_{p}$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere))
#
#    #c.plot(Trange/units.kelvin, rhou, '-b', linewidth=2)
#    #c.set_xlabel("Temperature in K", fontsize=14)
#    #c.set_ylabel("Density in g/ml", fontsize=14)
#    #c.set_title(r'$\rho$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere))    
#
#
#    f.subplots_adjust(hspace=0.40)
#    plt.show()

#Scatter plots look better RAM
    
    plt.scatter(Trange/units.kelvin, Hvap)
    #plt.scatter(Trange/units.kelvin, Hvap+dHu/1000, '--b', linewidth=2)  # bad units kludge
    #plt.scatter(Trange/units.kelvin, Hvap-dHu/1000, '--b', linewidth=2)  # bad units kludge
    plt.xlabel("Temperature in K", fontsize=14)
    plt.ylabel("Enthalpy of Vaporization\nin kJ/mol", fontsize=14)
    plt.title(r'$\Delta H_{\mathrm{vap}}$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere)) 
    plt.show()

    plt.scatter(Trange/units.kelvin, Cp2)
    #plt.scatter(Trange/units.kelvin, Cp2+dCp, '--b', linewidth=2)
    #plt.scatter(Trange/units.kelvin, Cp2-dCp, '--b', linewidth=2)
    plt.xlabel("Temperature in K", fontsize=14)
    plt.ylabel("Heat Capacity (const P)\nin kJ/(mol K)", fontsize=14)
    plt.title(r'$C_{p}$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere))
    plt.show()
    
    plt.scatter(Trange/units.kelvin, rhou)
    plt.xlabel("Temperature in K", fontsize=14)
    plt.ylabel("Density in gm/ml", fontsize=14)
    plt.title(r'$\rho$ for Liquid UA Methane at $P=%.2f$ atm'%(Prange[0]/units.atmosphere))
    plt.show()
    
    return Hu/units.kilojoules_per_mole, rhou, energies, mbar
    
####################################################################################
####################################################################################
####################################################################################

if __name__ == "__main__":
    '''
    python lj_compare_u0ln_with_u92ln.py
    
    This is a rather crude code that runs the execute command twice and compares
    the results. In the first case it uses all the sampled configurations from the 92 states
    while in the second case it only uses the sampled configurations from the 0th state.
    '''
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--nstates", dest="nstates", default=None, help="Set the number of states", metavar="NSTATES")
    (options, args) = parser.parse_args()
    x = np.load('LJTPstar.npy')
    Tstar = x[:,0]
    Pstar = x[:,1]
    if options.nstates is None:
        nstates = x.shape[0]
    else:
        nstates = int(options.nstates)
        
    Hu_all, rhou_all, energies_all, mbar_all = execute(nstates, Tstar, Pstar,True) # This uses all of the sampled configurations from states 0-91 -RAM
    Hu_0, rhou_0, energies_0, mbar_0 = execute(nstates, Tstar, Pstar,False) # This uses just the sampled configurations from state 0 -RAM
    
    dev_Hu = (Hu_0 - Hu_all) / Hu_all * 100.
    dev_rhou = (rhou_0 - rhou_all) / rhou_all * 100. 

    eps_sampled, sig_sampled = epsi_sig_from_TPstar(Tstar, Pstar)                                         
                                              
    plt.scatter(Hu_all,Hu_0)
    plt.plot([np.min([Hu_all,Hu_0]),np.max([Hu_all,Hu_0])],[np.min([Hu_all,Hu_0]),np.max([Hu_all,Hu_0])])
    plt.xlim([np.min([Hu_all,Hu_0]),np.max([Hu_all,Hu_0])])
    plt.ylim([np.min([Hu_all,Hu_0]),np.max([Hu_all,Hu_0])])
    plt.xlabel('All Sampled Configurations from States 0-91')
    plt.ylabel('Only Sampled Configurations from State 0')
    plt.title('Parity Plot of Enthalpy (kJ/mol)')
    plt.show()
    
    plt.scatter(rhou_all,rhou_0)
    plt.plot([np.min([rhou_all,rhou_0]),np.max([rhou_all,rhou_0])],[np.min([rhou_all,rhou_0]),np.max([rhou_all,rhou_0])])
    plt.xlim([np.min([rhou_all,rhou_0]),np.max([rhou_all,rhou_0])])
    plt.ylim([np.min([rhou_all,rhou_0]),np.max([rhou_all,rhou_0])])
    plt.xlabel('All Sampled Configurations from States 0-91')
    plt.ylabel('Only Sampled Configurations from State 0')
    plt.title('Parity Plot of Density (gm/ml)')
    plt.show()
    
    plt.scatter(mbar_all.f_k,mbar_0.f_k)
    plt.plot([np.min([mbar_all.f_k,mbar_0.f_k]),np.max([mbar_all.f_k,mbar_0.f_k])],[np.min([mbar_all.f_k,mbar_0.f_k]),np.max([mbar_all.f_k,mbar_0.f_k])])
    plt.xlim([np.min([mbar_all.f_k,mbar_0.f_k]),np.max([mbar_all.f_k,mbar_0.f_k])])
    plt.ylim([np.min([mbar_all.f_k,mbar_0.f_k]),np.max([mbar_all.f_k,mbar_0.f_k])])
    plt.xlabel('All Sampled Configurations from States 0-91')
    plt.ylabel('Only Sampled Configurations from State 0')
    plt.title('Parity Plot of Free Energy (kJ/mol)')
    plt.show()
    
    plt.plot(mbar_all.W_nk)
    plt.xlabel('Configuration')
    plt.ylabel('Weight')
    plt.title('Using All Sampled Configurations from States 0-91')
    plt.show()
    
    plt.plot(mbar_0.W_nk)
    plt.xlabel('Configuration')
    plt.ylabel('Weight')
    plt.title('Using Only Sampled Configurations from State 0')
    plt.show()
    
    plt.scatter(eps_sampled,dev_Hu)
    plt.scatter(eps_sampled[0],dev_Hu[0],label='State 0')
    plt.xlabel(r'$\epsilon$ (kJ/mol)')
    plt.ylabel('% Deviation in Enthalpy')
    plt.legend()
    plt.show()
    
    plt.scatter(sig_sampled,dev_Hu)
    plt.scatter(sig_sampled[0],dev_Hu[0],label='State 0')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel('% Deviation in Enthalpy')
    plt.legend()
    plt.show()
    
    plt.scatter(eps_sampled,dev_rhou)
    plt.scatter(eps_sampled[0],dev_rhou[0],label='State 0')
    plt.xlabel(r'$\epsilon$ (kJ/mol)')
    plt.ylabel('% Deviation in Density')
    plt.legend()
    plt.show()
    
    plt.scatter(sig_sampled,dev_rhou)
    plt.scatter(sig_sampled[0],dev_rhou[0],label='State 0')
    plt.xlabel(r'$\sigma$ (nm)')
    plt.ylabel('% Deviation in Density')
    plt.legend()
    plt.show()

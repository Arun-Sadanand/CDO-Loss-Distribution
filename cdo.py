import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import expon

######################################################################################################
# CDO Implementation as a Class
######################################################################################################
# Monte Carlo Simulation is used to generate simulation default loss amounts. 
# The fair spread for each tranche is based on the expected loss for each tranche.
######################################################################################################
class CDO:
    def __init__(self,lams,corr,maturity,attach_points,recovery=0.4,sims=10**4,notionals=np.array([])):
        self.n_bonds = len(lams) # No. of bonds in CDO contract
        self.n_trche = len(attach_points) # No. of tranches
        
        self.lams = lams # Array of default intensities for constituent bonds
        self.corr = corr # Correlation matrix for constituent bonds
        self.recovery = recovery # Constant recovery rate for each bond
        self.maturity = maturity # Maturity of CDO contract
        self.sims = sims # No. of Monte Carlo runs
        
        self.bond_pars = notionals # Array of bond face values
        if self.bond_pars.size==0:
            self.bond_pars = np.ones(self.n_bonds) # Default face value of $1
        
        self.att = attach_points*np.sum(self.bond_pars) # Trache attachment points
        self.det = np.append(attach_points[1:],1)*np.sum(self.bond_pars) # Trache detachment points
        
        self.payouts = self.sim_payout()
        self.spreads = self.get_spreads()
        
    # Generate array of simulated default amounts for the full CDO
    def sim_payout(self):
        notionals = np.tile(self.bond_pars,(self.sims,1)).T # Used to scale payouts to bond par values
            
        # Generate Multivariate random samples
        f = np.random.multivariate_normal(np.zeros(self.n_bonds),self.corr,self.sims)
    
        # Transform them into cumulative default probability space
        cdp = multivariate_normal.cdf(f.ravel()) #uniform distribution
    
        # get the default time samples for each issuer
        # default time given cumulative default probability
        taus = np.zeros_like(notionals)
        for i in range(self.n_bonds):
            taus[i,:] = expon(scale=1/self.lams[i]).ppf(cdp[i::self.n_bonds])
    
        # CDS payout for each simulation
        payouts = np.where(taus<=self.maturity,(1-self.recovery)*notionals,0)
        payouts = np.sum(payouts, axis=0)
    
        return payouts
        
    # Distrubute payouts into tranches to get expected payout and calculate tranche spreads
    def get_spreads(self,rf_rate=0.04): # Default risk-free rate of 4%
        tr_payouts = np.zeros([self.n_trche,self.sims])
            
        # For each run, distribute total CDO payout into tranches based on attach and detach points
        for i in range(self.n_trche):
            tr_payouts[i,:] = np.where(self.payouts>self.att[i],self.payouts-self.att[i],0)
            tr_payouts[i,:] = np.where(self.payouts>self.det[i],self.det[i]-self.att[i],tr_payouts[i,:])
            
        # Expecated payout for each tranche
        expected_payout = np.mean(tr_payouts, axis=1)/np.sum(self.bond_pars)
            
        # Calculate risk-neutral spread for each tranche
        ts = np.arange(1,self.maturity+1) # Annual payments
        discounts = 1/(1+rf_rate)**ts # Discount factors based on flat risk-free rate
        # Annuity formula to get spread for each Tranche
        spreads = np.zeros(self.n_trche)
        for i in range(self.n_trche):
            spreads[i] = expected_payout[i]/(1+rf_rate)**self.maturity / np.sum(discounts)
        return spreads
    
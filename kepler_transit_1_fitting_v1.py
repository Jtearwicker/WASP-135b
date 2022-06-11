"""
Name
----
kepler_transit_1_fitting_v1.py

Description
-----------
This script uses fits a Kepler-167e from Kepler individually using BATMAN and emcee.
It puts Gaussian priors on all parameters (using Kipping+ 2016 values) except for the 
mid-transit time, which is the parameter that is really desired in the end. This code is
for the transit 1.

Input
-----
Kepler data and Gaussian priors

Output
------
The best fit parameters, of which the mid-transit time is most important. It also creates
plots of the best fit model. 

Author
------
P. A. Dalba -- January 2019, UC Riverside
"""

#Import various math, science, and plotting packages.
imprt = 0
while imprt==0:
	import numpy, scipy
	from scipy.interpolate import interp1d
	#import commands  #replaced by subprocess in 3
	import matplotlib
	#matplotlib.use('macosx')  #it's MacOSX by default for 3, perhaps comment out
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import *
	from matplotlib import colors, cm
	from matplotlib.pylab import *
	from matplotlib.font_manager import FontProperties
	from mpl_toolkits.mplot3d import Axes3D
	plt.ion()
	import os, glob
	import sys, math
	from scipy import stats, signal
	from numpy import *
	from scipy import optimize
	from scipy.optimize import curve_fit
	from scipy.optimize import fsolve
	import idlsave
	import time
	import multiprocessing
	from multiprocessing import Pool
	import emcee
	import pickle
	from scipy.spatial import distance
	import itertools
	import astropy
	from astropy.io import fits
	from astropy.wcs import WCS
	from astropy.time import Time
	import batman
	import corner
	imprt = 1 
close('all')

#Set paths and various things
#-----------------------------------------------------------------------------------------
root_path = '/Users/paul/Documents/UC_Riverside/Research/Kepler-167e_transit/'
kepler_path = root_path+'Kepler_data/'
kepler_files = ['kplr003239945-2010078095331_llc_trim.fits',\
	'kplr003239945-2013065031647_slc_trim.fits']
save_mode = False
#Offsets to get the transit times on the order of unity. All times are BKJD and TDB, as
# specific in the headers
BKJD_offsets = [419.,1491.]  
#-----------------------------------------------------------------------------------------


#Kepler-167(e) parameters from Kipping et al. 2016
#-----------------------------------------------------------------------------------------
per_guess, per_scale = 1071.23228,0.00056
rp_guess, rp_scale = 0.12810,0.00093
a_guess, a_scale = 560.,15.
inc_guess, inc_scale = 89.9760,0.0070
ecc_guess, ecc_scale = 0.062,0.104
w_guess, w_scale = 3.5*180./pi,2.9*180./pi
u1_guess, u1_scale = 0.915,0.02
u2_guess, u2_scale = -0.243,0.04
#Transform limb darkening params
q1_guess, q1_scale = (u1_guess+u2_guess)**2,0.1
q2_guess, q2_scale = 0.5*u1_guess/(u1_guess+u2_guess),0.1
#-----------------------------------------------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TRANSIT 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Load in data and do any processing at this point
#-----------------------------------------------------------------------------------------
kepler_data = fits.open(kepler_path+kepler_files[0])

t_data, data, err_data = kepler_data[1].data['TIME'], kepler_data[1].data['PDCSAP_FLUX'],\
	kepler_data[1].data['PDCSAP_FLUX_ERR']

#Normalize flux data 
err_data /= median(data[where(data>36500)[0]])
data /= median(data[where(data>36500)[0]])

#Scatter plot
fig1 = figure(1)
ax1 = fig1.add_subplot(211)
ax1.scatter(t_data-BKJD_offsets[0],data,facecolor='none',edgecolor='b',marker='o')
#ax1.set_xlabel('BKJD',fontsize='large')
ax1.set_ylabel('Normalized Flux',fontsize='large')
axhline(1,c='0.5',ls='--')

stop = input('Stop after data display...')
#-----------------------------------------------------------------------------------------


#Define transit model functions
#-----------------------------------------------------------------------------------------
#Transit model from batman using Kipping (2013) limb darkening conversion
def get_transit(t,t0,per,rp,a,inc,ecc,w,u1,u2):
	#Build transit params object
	params = batman.TransitParams()
	params.t0 = t0
	params.per = per
	params.rp = rp
	params.a = a
	params.inc = inc
	params.ecc = ecc
	params.w = w
	params.u = [u1,u2]
	params.limb_dark = 'quadratic'
	#Initialize model and return it
	m = batman.TransitModel(params, t)
	return m.light_curve(params)

#Function to correct for long cadence data through numerical integration (critical time
# scale is 29.4 minutes)	
def lc_integrate(t,t0,per,rp,a,inc,ecc,w,u1,u2):
	#Timescale of cadence
	t_cad = 29.4/(60.*24.)
	#Oversampling parameter
	oversamp = 20
	#For a given timestamp, this function calls quad to integrate the LC function
	lims = [t-(0.5*t_cad),t+(0.5*t_cad)]
	#Call the light curve function, oversampled by 20 (Southworth 2011 showed that an 
	# oversampling of 15 should be sufficient
	t_chunk = linspace(lims[0],lims[1],oversamp)
	model_chunk = get_transit(t_chunk,t0,per,rp,a,inc,ecc,w,u1,u2)
	#Integrate, normalize to the cadence time, and return
	return trapz(model_chunk,x=t_chunk)/t_cad
#-----------------------------------------------------------------------------------------


#Define MCMC functions
#-----------------------------------------------------------------------------------------
#Likelihood
def ln_likelihood(p,t_data,data,err_data):
	#Unpack params
	t0,per,rp,a,inc,ecc,w,q1,q2 = p
	
	#Convert limb darkening params
	u1 = 2.*q2*sqrt(q1)		
	u2 = (1.-2.*q2)*sqrt(q1)
	
	#Make model	
	model = zeros(size(t_data))
	for ii in range(size(t_data)):
		model[ii] = lc_integrate(t_data[ii]-BKJD_offsets[0],t0,per,rp,a,inc,ecc,w,u1,u2)	
	return -0.5*sum(((data-model)/err_data)**2 + log(2.*pi*err_data**2))

#Priors
def gaussian(x,mu,sigma):
	return exp(-0.5*((x-mu)/sigma)**2)/sqrt(2.*pi*sigma*sigma)

def ln_prior(p):
	t0,per,rp,a,inc,ecc,w,q1,q2 = p
	
	#Convert limb darkening params
	u1 = 2.*q2*sqrt(q1)		
	u2 = (1.-2.*q2)*sqrt(q1)
	
	#Uniform prior on w (b/c of non-gaussian shape in Kipping posteriors)
	if (0.<=w<=360.) and (0.<=q1<=1.) and (0<=q2<=1.):
		#Gaussian priors on everything else except for t0, which has no prior
		prior = 0.
		prior += log(gaussian(per,per_guess,per_scale))
		#prior += log(gaussian(rp,rp_guess,rp_scale))
		prior += log(gaussian(a,a_guess,a_scale))
		prior += log(gaussian(inc,inc_guess,inc_scale))
		prior += log(gaussian(ecc,ecc_guess,ecc_scale))
		#prior += log(gaussian(w,w_guess,w_scale))
		#prior += log(gaussian(u1,u1_guess,u1_scale))
		#prior += log(gaussian(u2,u2_guess,u2_scale))
		return prior
	return -inf	
	
#Posterior
def ln_probability(p,t_data,data,err_data):
    global iteration
    iteration +=1
    sys.stdout.write("Sampler progress: %d%%   \r" % \
    	(n_threads*100.*iteration/(n_walkers*n_steps + n_walkers -1)))
    sys.stdout.flush()
    priors = ln_prior(p)
    if not isfinite(priors):
        return -inf
    else:
    	return priors + ln_likelihood(p,t_data,data,err_data)	
#-----------------------------------------------------------------------------------------


#Set up and run MCMC
#-----------------------------------------------------------------------------------------
n_dim, n_walkers, n_steps, iteration, n_threads = 9, 300, 50000, -1, 4
 
#Initialize walkers
t0_rand = random.normal(loc=1.28655,scale=0.0001,size=n_walkers)
per_rand = random.normal(loc=1071.232295,scale=0.0001,size=n_walkers)
rp_rand = random.normal(loc=0.1214,scale=0.0002,size=n_walkers)
a_rand = random.normal(loc=556.,scale=1.,size=n_walkers)
inc_rand = random.normal(loc=89.97606,scale=0.001,size=n_walkers)
ecc_rand = random.normal(loc=0.026,scale=0.005,size=n_walkers)
#w_rand = random.normal(loc=w_guess,scale=0.1*w_scale,size=n_walkers)
w_rand = random.uniform(low=1,high=359,size=n_walkers)
q1_rand = random.normal(loc=0.64,scale=0.02,size=n_walkers)
q2_rand = random.normal(loc=0.32,scale=0.01,size=n_walkers) 
positions = []
for param in range(n_walkers):
	positions.append(np.array([t0_rand[param],per_rand[param],rp_rand[param],a_rand[param],\
		inc_rand[param],ecc_rand[param],w_rand[param],q1_rand[param],q2_rand[param]]))

#Initialize and run the sampler
sampler = emcee.EnsembleSampler(n_walkers,n_dim,ln_probability,args=(t_data,data,err_data),\
	threads=n_threads)
sampler.run_mcmc(positions,n_steps)

chain_shape = np.shape(sampler.chain) 

#Plot the walkers
fig2, axes = plt.subplots(n_dim, sharex=True, figsize=(13,10))

#Y-axis labels and other pleasantries
axes[0].set_title('Walker Paths', fontsize='large')
axes[0].set_ylabel('$t_0$',fontsize='large')
axes[1].set_ylabel('$P$',fontsize='large')
axes[2].set_ylabel('$R_p/R_{\star}$',fontsize='large')
axes[3].set_ylabel('$a/R_{\star}$',fontsize='large')
axes[4].set_ylabel('$inc$',fontsize='large')
axes[5].set_ylabel('$ecc$',fontsize='large')
axes[6].set_ylabel('$\omega$',fontsize='large')
axes[7].set_ylabel('$q_1$',fontsize='large')
axes[8].set_ylabel('$q_2$',fontsize='large')
axes[8].set_xlabel('Step Number', fontsize='large')
axes[8].set_xlim(0,chain_shape[1])
for walkers in range(chain_shape[0]):
	for params in range(chain_shape[2]):
		#If huge number of samples (i.e., huge plot) only show every Nth walker
		if ((n_steps*n_walkers) > 1e5) and (walkers % 3 == 0):
			axes[params].plot(sampler.chain[walkers,:,params], linewidth=0.5, \
				alpha=0.5,color='k',rasterized=True)
		else:	
			axes[params].plot(sampler.chain[walkers,:,params], linewidth=0.5, alpha=0.5,\
				color='k',rasterized=True)
#Burn in
burn = int(input('Enter the burn_in length: '))
#Now apply the burn by cutting those steps out of the chain. 
chain_burnt = sampler.chain[:, burn:, :] 
#Also flatten the chain to just a list of samples
samples = chain_burnt.reshape((-1, n_dim))


#Corner plot
fig3 = corner.corner(samples, labels=['$t_0$','$P$','$R_p/R_{\star}$','$a/R_{\star}$',\
	'$inc$','$ecc$','$\omega$','$q_1$','$q_2$'],rasterized=True)
plt.show()

#Transform q's back into u's for limb darkening
u1_dist = 2.*samples[:,8]*sqrt(samples[:,7])
u2_dist = sqrt(samples[:,7])*(1.-2.*samples[:,8])

#Get values
t0_mcmc, per_mcmc, rp_mcmc, a_mcmc, inc_mcmc, ecc_mcmc, w_mcmc, q1_mcmc, q2_mcmc = \
	map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples,[16, 50, 84],\
	axis=0)))
u1_mcmc, u2_mcmc = map(lambda v: (v[1], v[2]-v[1],v[1]-v[0]),\
	zip(*np.percentile(append(u1_dist,u2_dist).reshape(2,size(u1_dist)).transpose(),\
	[16, 50, 84],axis=0)))	

print('')
print('t0\t', t0_mcmc)
print('per\t', per_mcmc)
print('rp\t', rp_mcmc)
print('a\t', a_mcmc)
print('inc\t', inc_mcmc)
print('ecc\t', ecc_mcmc)
print('w\t', w_mcmc)
print('q1\t', q1_mcmc)
print('q2\t', q2_mcmc)
print('u1\t', u1_mcmc)
print('u2\t', u2_mcmc)
#-----------------------------------------------------------------------------------------



#Plot best-fit model onto data and find residuals
#-----------------------------------------------------------------------------------------
t_model = linspace(t_data[0]-BKJD_offsets[0],t_data[-1]-BKJD_offsets[0],1000)
best_model = get_transit(t_model,t0_mcmc[0],per_mcmc[0],rp_mcmc[0],a_mcmc[0],inc_mcmc[0],\
	ecc_mcmc[0],w_mcmc[0],u1_mcmc[0],u2_mcmc[0])
ax1.plot(t_model,best_model,c='orange')

#Residuals
best_resid = zeros(size(t_data))
for i in range(size(t_data)):
	best_resid[i] = data[i] - lc_integrate(t_data[i]-BKJD_offsets[0],t0_mcmc[0],\
	per_mcmc[0],rp_mcmc[0],a_mcmc[0],inc_mcmc[0],ecc_mcmc[0],w_mcmc[0],u1_mcmc[0],u2_mcmc[0])

ax2 = fig1.add_subplot(212,sharex=ax1)
ax2.set_xlabel('BKJD',fontsize='large')
ax2.set_ylabel('Data - Model',fontsize='large')
ax2.scatter(t_data-BKJD_offsets[0],best_resid,facecolor='none',edgecolor='k')
ax2.errorbar(t_data-BKJD_offsets[0],best_resid,yerr=err_data,capsize=0,fmt='none',ecolor='k')
ax2.axhline(0,c='r',ls='--')
#-----------------------------------------------------------------------------------------


#Save information from the run if desired
#-----------------------------------------------------------------------------------------
if save_mode:
	#Save figures
	fig1.savefig(kepler_path+'transit1_transit_and_residuals_v1.pdf',bbox_inches='tight')
	fig2.savefig(kepler_path+'transit1_walkers_v1.pdf',bbox_inches='tight')
	fig3.savefig(kepler_path+'transit1_corner_v1.pdf',bbox_inches='tight')
	#Save samples
	pickle.dump(samples[:,[0,2]], open(kepler_path+'transit1_samples_v1.pickle','wb'))	
	#Save mcmc values
	pickle.dump((t0_mcmc,per_mcmc,rp_mcmc,a_mcmc,inc_mcmc,ecc_mcmc,w_mcmc,q1_mcmc,\
		q2_mcmc,u1_mcmc,u2_mcmc),open(kepler_path+'transit1_params_v1.pickle','wb'))
#-----------------------------------------------------------------------------------------


#Add a function for convergence testing
#-----------------------------------------------------------------------------------------
def conv_test(save_mode=True):
	print('')
	print('')
	print('----------------------- Convergence test -----------------------')
	print('')
	print('N walkers per param = '+str(n_walkers))
	print('Size of chains after burn = '+str(n_steps-burn))
	print('')
	try: print('t0:   tau = '+str(emcee.autocorr.integrated_time(samples[:,0])))
	except: print('t0:   Chain is too short to find accurate autocorr. time')
	try: print('per:  tau = '+str(emcee.autocorr.integrated_time(samples[:,1])))
	except: print('per:   Chain is too short to find accurate autocorr. time')
	try: print('rp:   tau = '+str(emcee.autocorr.integrated_time(samples[:,2])))
	except: print('rp:   Chain is too short to find accurate autocorr. time')
	try: print('a:    tau = '+str(emcee.autocorr.integrated_time(samples[:,3])))
	except: print('a:   Chain is too short to find accurate autocorr. time')
	try: print('inc:  tau = '+str(emcee.autocorr.integrated_time(samples[:,4])))
	except: print('inc:   Chain is too short to find accurate autocorr. time')
	try: print('ecc:  tau = '+str(emcee.autocorr.integrated_time(samples[:,5])))
	except: print('ecc:   Chain is too short to find accurate autocorr. time')
	try: print('w:    tau = '+str(emcee.autocorr.integrated_time(samples[:,6])))
	except: print('w:   Chain is too short to find accurate autocorr. time')
	try: print('q1:   tau = '+str(emcee.autocorr.integrated_time(samples[:,7])))
	except: print('q1:   Chain is too short to find accurate autocorr. time')
	try: print('q2:   tau = '+str(emcee.autocorr.integrated_time(samples[:,8])))
	except: print('q2:   Chain is too short to find accurate autocorr. time')
	if save_mode:
		file = open(kepler_path+'transit1_convergence_test_v1.txt','w')
		file.write('----------------------- Convergence test -----------------------\n')
		file.write('\n')
		file.write('N walkers per param = '+str(n_walkers)+'\n')
		file.write('Size of chains after burn = '+str(n_steps-burn)+'\n')
		file.write('\n')
		try: file.write('t0:   tau = '+str(emcee.autocorr.integrated_time(samples[:,0]))+\
			'\n')
		except: file.write('t0:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('per:  tau = '+str(emcee.autocorr.integrated_time(samples[:,1]))+\
			'\n')
		except: file.write('per:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('rp:   tau = '+str(emcee.autocorr.integrated_time(samples[:,2]))+\
			'\n')
		except: file.write('rp:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('a:    tau = '+str(emcee.autocorr.integrated_time(samples[:,3]))+\
			'\n')
		except: file.write('a:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('inc:  tau = '+str(emcee.autocorr.integrated_time(samples[:,4]))+\
			'\n')
		except: file.write('inc:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('ecc:  tau = '+str(emcee.autocorr.integrated_time(samples[:,5]))+\
			'\n')
		except: file.write('ecc:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('w:    tau = '+str(emcee.autocorr.integrated_time(samples[:,6]))+\
			'\n')
		except: file.write('w:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('q1:   tau = '+str(emcee.autocorr.integrated_time(samples[:,7]))+\
			'\n')
		except: file.write('q1:   Chain is too short to find accurate autocorr.time\n')	
		try: file.write('q2:   tau = '+str(emcee.autocorr.integrated_time(samples[:,8])))
		except: file.write('q2:   Chain is too short to find accurate autocorr.time')	
		file.close()
	return None
conv_test(save_mode=save_mode)		
#-----------------------------------------------------------------------------------------













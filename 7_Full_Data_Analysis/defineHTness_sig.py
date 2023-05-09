from ROOT import *
import numpy as np
import math
import glob
import sys
import pandas as pd
from scipy.optimize import minimize


def defineHTness():


	h_mass,t_mass,w_mass,z_mass = 125.18,172.26,80.379,91.1876

	sig = pd.read_csv('/cms/ldap_home/chdlalsnr/HH/CMSSW_11_0_0/src/HHAnalysis/HH/python/7_Full_Data_Analysis/csv/signal.csv')

	sig = sig.values.tolist()

	columns_name = ['jet1_mass','jet1_pt','jet1_eta','jet1_phi',
                        'jet2_mass','jet2_pt','jet2_eta','jet2_phi',
                        'lep1_mass','lep1_pt','lep1_eta','lep1_phi',
                        'lep2_mass','lep2_pt','lep2_eta','lep2_phi',
                        'dr_jj','dr_ll','met_pt','met_phi',
                        'H1_mass','H1_pt','H1_eta','H1_phi',
                        'H2_mass','H2_pt','H2_eta','H2_phi']


	signal = pd.DataFrame(sig, columns=columns_name)

	jet1_mass = signal.jet1_mass.tolist()
	jet1_pt = signal.jet1_pt.tolist()
	jet1_eta = signal.jet1_eta.tolist()
	jet1_phi = signal.jet1_phi.tolist()

	jet2_mass = signal.jet2_mass.tolist()
        jet2_pt = signal.jet2_pt.tolist()
        jet2_eta = signal.jet2_eta.tolist()
        jet2_phi = signal.jet2_phi.tolist()

	lep1_mass = signal.lep1_mass.tolist()
        lep1_pt = signal.lep1_pt.tolist()
        lep1_eta = signal.lep1_eta.tolist()
        lep1_phi = signal.lep1_phi.tolist()

        lep2_mass = signal.lep2_mass.tolist()
        lep2_pt = signal.lep2_pt.tolist()
        lep2_eta = signal.lep2_eta.tolist()
        lep2_phi = signal.lep2_phi.tolist()

	dr_jj = signal.dr_jj.tolist()
	dr_ll = signal.dr_ll.tolist()

	met_pt = signal.met_pt.tolist()
	met_phi = signal.met_phi.tolist()

	h1_mass = signal.H1_mass.tolist()
        h1_pt = signal.H1_pt.tolist()
        h1_eta = signal.H1_eta.tolist()
        h1_phi = signal.H1_phi.tolist()

        h2_mass = signal.H2_mass.tolist()
        h2_pt = signal.H2_pt.tolist()
        h2_eta = signal.H2_eta.tolist()
        h2_phi = signal.H2_phi.tolist()

	higgsness,topness = [],[]
	# =================================================================
	# Define Higgsness and Topness
	data = []
	for i in range(len(jet1_mass)):

		lep1_p4 = TLorentzVector()
                lep2_p4 = TLorentzVector()

		jet1_p4 = TLorentzVector()
                jet2_p4 = TLorentzVector()

		lep1_p4.SetPtEtaPhiM(lep1_pt[i],lep1_eta[i],lep1_phi[i],lep1_mass[i])
		lep2_p4.SetPtEtaPhiM(lep2_pt[i],lep2_eta[i],lep2_phi[i],lep2_mass[i])

		jet1_p4.SetPtEtaPhiM(jet1_pt[i],jet1_eta[i],jet1_phi[i],jet1_mass[i])
                jet2_p4.SetPtEtaPhiM(jet2_pt[i],jet2_eta[i],jet2_phi[i],jet2_mass[i])

                nu1_p4 = TLorentzVector()
                nu2_p4 = TLorentzVector()

		pt = met_pt[i]
		phi = met_phi[i]

		def defineTopness(x):

			chi2 = []

			pt1,eta1,phi1 = x[0],x[1],x[2]
			pt2,eta2,phi2 = x[3],x[4],x[5]

			nu1_p4.SetPtEtaPhiM(x[0], x[1], x[2], 0)
			nu2_p4.SetPtEtaPhiM(x[3], x[4], x[5], 0)

			# w1 and w2 have different electric charges
			w1_p4 = lep1_p4 + nu1_p4
			w2_p4 = lep2_p4 + nu2_p4

			# (t1, t2) and (t5, t6) have anti-particle relations
			t1_p4 = w1_p4 + jet1_p4
			t2_p4 = w1_p4 + jet2_p4
			t3_p4 = w2_p4 + jet1_p4
			t4_p4 = w2_p4 + jet2_p4

			t_cand = [t1_p4.M(),t2_p4.M(),t3_p4.M(),t4_p4.M()]
			w_cand = [w1_p4.M(),w2_p4.M()]

			sig_t = 48.1
			sig_w = 21.4

			chi2.append( ((t_cand[0]**2-t_mass**2)**2/sig_t**4) + ((w_cand[0]**2-w_mass**2)**2/sig_w**4) + ((t_cand[3]**2-t_mass**2)**2/sig_t**4) + ((w_cand[1]**2-w_mass**2)**2/sig_w**4) )
			chi2.append( ((t_cand[1]**2-t_mass**2)**2/sig_t**4) + ((w_cand[0]**2-w_mass**2)**2/sig_w**4) + ((t_cand[2]**2-t_mass**2)**2/sig_t**4) + ((w_cand[1]**2-w_mass**2)**2/sig_w**4) )

			topness = min(chi2)

			return topness


		ini = np.array([0,0,0, 0,0,0])
                bnd = [(0,50),(-2.4,2.4),(-3.14,3.14), (0,50),(-2.4,2.4),(-3.14,3.14)]

		cons = ({'type':'eq',
			 'fun': lambda x: x[0]+x[3] - pt},
			{'type':'eq',
			 'fun': lambda x: x[2]+x[5] - phi})

                opt = minimize(defineTopness, ini, bounds=bnd, method='SLSQP', constraints=cons, options={'ftol':5}) #'disp':True

		if opt.success:
			topness.append(opt.fun)

		if not opt.success: continue

		nu1_p4.SetPtEtaPhiM(opt.x[0],opt.x[1],opt.x[2], 0)
		nu2_p4.SetPtEtaPhiM(opt.x[3],opt.x[4],opt.x[5], 0)

		w1_p4 = lep1_p4 + nu1_p4
		w2_p4 = lep2_p4 + nu2_p4

		h1_p4 = w1_p4 + w2_p4

		dilep_p4 = lep1_p4 + lep2_p4


		def defineHiggsness():

                        sig_h  = 46.9
                        sig_on,sig_off = 21.0,25.4
			sig_lep = 20.1
                        peak_off = (1/np.sqrt(3)) * np.sqrt( 2*(h_mass**2 + w_mass**2) - np.sqrt(h_mass**4 + 14*(h_mass**2)*(w_mass**2) + w_mass**4) )
			peak_dilep = 30

                        inner = [((w1_p4.M()**2-w_mass**2)**2/sig_on**4 + (w2_p4.M()**2-peak_off**2)**2/sig_off**4), ((w2_p4.M()**2-w_mass**2)**2/sig_on**4 + (w1_p4.M()**2-peak_off**2)**2/sig_off**4)]

                        higgsness = (h1_p4.M()**2-h_mass**2)**2/sig_h**4 + (dilep_p4.M()**2-peak_dilep**2)**2/sig_lep**4  + min(inner)

                        return higgsness


		# Save CSV file
                data.append([jet1_p4.M(),jet1_p4.Pt(),jet1_p4.Eta(),jet1_p4.Phi(),
                        jet2_p4.M(),jet2_p4.Pt(),jet2_p4.Eta(),jet2_p4.Phi(),
                        lep1_p4.M(),lep1_p4.Pt(),lep1_p4.Eta(),lep1_p4.Phi(),
                        lep2_p4.M(),lep2_p4.Pt(),lep2_p4.Eta(),lep2_p4.Phi(),
			dr_jj[i],dr_ll[i],met_pt[i],met_phi[i],
			h1_mass[i],h1_pt[i],h1_eta[i],h1_phi[i],
			h2_mass[i],h2_pt[i],h2_eta[i],h2_phi[i],
                        defineHiggsness(),opt.fun,1])

        df = pd.DataFrame(data, columns=
                                ['jet1_mass','jet1_pt','jet1_eta','jet1_phi',
                                 'jet2_mass','jet2_pt','jet2_eta','jet2_phi',
                                 'lep1_mass','lep1_pt','lep1_eta','lep1_phi',
                                 'lep2_mass','lep2_pt','lep2_eta','lep2_phi',
				 'dr_jj','dr_ll','met_pt','met_phi',
				 'h1_mass','h1_pt','h1_eta','h1_phi',
				 'h2_mass','h2_pt','h2_eta','h2_phi',
                                 'higgsness','topness','target'])

        df.to_csv("htness_sig.csv", header=True, index=False)

	input("Press Enter to continue...")


if __name__ == "__main__":
	defineHTness()

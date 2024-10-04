#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import damask

class LoadY(damask.LoadcaseGrid):
    def __init__(self, fn_load):
        super().__init__(damask.LoadcaseGrid.load(fn_load))
        self.fn = fn_load
        self.restart_n = None
    
    def get_sim_restart_n(self, fn_hdf5):
        res = damask.Result(fn_hdf5)
        inc = int(res.increments[-1].split('_')[-1])
        ls_id, n_done, n_over = self.find_loadinc(inc)
        f_out = self['loadstep'][ls_id]['f_out']
        f_restart = self['loadstep'][ls_id]['f_restart']
        if f_out == f_restart:
            self.restart_n = inc
        else:
            self.restart_n = inc - n_done + f_restart * (n_done//f_restart)
        print(f'last record  inc is {inc}')
        print(f'last restart inc is {self.restart_n}')

    def upd_load_with_restart(self):
        pass

    def find_loadinc(self, load_inc):
        N_l = [ls['discretization']['N'] for ls in self.get('loadstep')]
        assert load_inc<=sum(N_l), Exception('restart larger than end of total incs')
        ls_id = np.digitize(load_inc, np.cumsum(N_l), True)
        n_done = load_inc - sum(N_l[:ls_id])
        n_over = sum(N_l[:ls_id+1]) - load_inc
        print(f'load_inc ({load_inc}) in loadstep #{ls_id}, remaing inc N={n_over} in #{ls_id}')
        return ls_id, n_done, n_over

    def clip_after_loadinc(self, load_inc):     
        """ based on load_inc, clean load.yaml to be valid """
        load_upd = self.copy()
        ls_id, n_done, n_over = self.find_loadinc(load_inc)
        t = load_upd['loadstep'][ls_id]['discretization']['t']
        N = load_upd['loadstep'][ls_id]['discretization']['N']
        t_upd = t*n_done/N
        load_upd['loadstep'][ls_id]['discretization']['t'] = t_upd
        load_upd['loadstep'][ls_id]['discretization']['N'] = n_done
        load_upd['loadstep'][ls_id+1:] = [] # discard later steps
        return load_upd

    def load_append_refine(self, load_inc, t_more, refine_factor):
        load_upd = self.clip_after_loadinc(load_inc) 
        step_last = load_upd['loadstep'][-1].copy()     # get last
        t = step_last['t']
        N = step_last['N']
        step_last['discretization'] = {
            't': t_more,
            'N': int(refine_factor*N*t_more/t)
        }
        load_upd['loadstep'].append(step_last)
        return load_upd

    def create_restart(self, t_more, refine_factor):
        """ this can be used also as append restart! """
        if self.restart_n:
            load_upd = self.load_append_refine(self.restart_n, t_more, refine_factor)
            return load_upd
        else:
            raise Exception('use hdf5 to find restart_n first!')

    def create_refine_before_fail(self, refine_factor):
        if self.restart_n:
            load_upd = self.copy()
            ls_id, n_done, n_over = self.find_loadinc(self.restart_n)
            t = load_upd['loadstep'][ls_id]['discretization']['t']
            N = load_upd['loadstep'][ls_id]['discretization']['N']
            load_upd['loadstep'][ls_id]['discretization']['N'] = int(N*refine_factor)
            load_upd['loadstep'][ls_id+1:] = [] # discard later steps
            return load_upd
        else:
            raise Exception('use hdf5 to find restart_n first!')

    def create_refine_2nd_last_append(self, t_more, refine_factor):
        """ refine the 2nd last loadstep with refine_factor, and then continue with t_more time """
        load_upd = self.copy()
        ls_2nd_last = load_upd['loadstep'][-2]
        t = ls_2nd_last['discretization']['t']
        N = ls_2nd_last['discretization']['N']
        ls_2nd_last['discretization'] = {
            't': t + t_more,
            'N': int(refine_factor*N*(t + t_more)/t)
        }
        load_upd['loadstep'][-1:] = []
        return load_upd

    def analyze_eqv_strain(self):
        print('this is a very approximate estimate of overall eqv strain!')
        print('every element with x is replaced by 0, which is not realistic')
        pass

    def unfold_loadsteps(self):
        """ loadstep into load/time/record series, esp. for plot and analysis """
        case_strain_d = {
            'L': lambda time_series, f_or_df: time_series,
            'dot_F': lambda time_series, f_or_df: time_series,
            'F': lambda time_series, f_or_df: time_series
        }
        case_stress_d = {
            'P': lambda time_series, f_or_df, start_val: time_series,
            'dot_P': lambda time_series, f_or_df, start_val: time_series,
        }

        loadsteps = self.get('loadstep')
        time = np.array([])
        id_rec = np.array([])
        strain = [[[] for i in range(3)] for j in range(3)]
        stress = [[[] for i in range(3)] for j in range(3)]
        t_last = 0.
        id_rec_last = 0
        strain_last = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        stress_last = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        for step in loadsteps:
            t = float(step['discretization']['t'])
            N = int(step['discretization']['N'])
            f_out = int(step['f_out'])
            dt = t / N
            for k, v in step['boundary_conditions']['mechanical'].items():
                if k in case_strain_d.keys():
                    strain_curr = v                
                if k in case_stress_d.keys():
                    stress_curr = v
            
            # Generate time steps
            time_curr = np.linspace(0, t, N, endpoint=False)
            time = np.concatenate((time, time_curr + t_last if len(time) else time_curr))
            id_rec_curr = np.arange(0, N, f_out)
            id_rec = np.concatenate((id_rec, id_rec_curr + id_rec_last if len(id_rec) else id_rec_curr))
            
            # Extract and append L values
            for i in range(3):
                for j in range(3):
                    if strain_curr[i][j] == 'x':
                        strain[i][j] = np.append(strain[i][j], np.full(N, np.nan))
                    else:
                        strain_ij = strain_last[i][j] + time_curr * strain_curr[i][j] # !! this integrator should match L/dot_F/F
                        strain[i][j] = np.append(strain[i][j], strain_ij)
                        strain_last[i][j] += strain_curr[i][j] * t
                    
                    if stress_curr[i][j] == 'x':
                        stress[i][j] = np.append(stress[i][j], np.full(N, np.nan))
                    else:
                        stress_ij = stress_last[i][j] + time_curr * stress_curr[i][j] # !! this integrator should match L/dot_F/F
                        stress[i][j] = np.append(stress[i][j], stress_ij)
                        stress_last[i][j] += stress_curr[i][j] * t

            t_last += t
            id_rec_last += N # !! this will assume N%f_out = 0
        
        # append the missing last
        time = np.append(time, t_last)
        id_rec = np.append(id_rec, id_rec_last)
        for i in range(3):
            for j in range(3):
                strain[i][j] = np.append(strain[i][j], strain_last[i][j])
                stress[i][j] = np.append(stress[i][j], stress_last[i][j])

        return strain, stress, time, id_rec.astype(int)

    def plot_load(self, fn_save=None, opt='diag'):
        if fn_save is None:
            fn_save = self.fn.replace('yaml','png')
        
        strain, stress, time, id_rec = self.unfold_loadsteps()

        # Plot the data in three subplots in a row
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))

        for i in range(3):
            axs[0,i].plot(time, strain[i][i], ',', ms=4, label=f"F[{i},{i}]")
            axs[0,i].plot(time[id_rec], strain[i][i][id_rec], 'o', markerfacecolor="None")
            axs[0,i].set_xlabel('Time')
            axs[0,i].set_ylabel(f'F[{i},{i}]')
            # axs[0,i].set_title('F[{i},{i}] over Time')
            axs[0,i].grid(True)

            axs[1,i].plot(time, stress[i][i], ',', label=f"P[{i},{i}]", color='orange')
            axs[1,i].set_xlabel('Time')
            axs[1,i].set_ylabel(f'P[{i},{i}]')
            # axs[1,i].set_title('f"P[{i},{i}]" over Time')
            axs[1,i].grid(True)


        fig.savefig(fn_save,bbox_inches='tight')




def test_operate():
    fn_load = 'test_load.yaml'
    fn_hdf5 = 'test_result.hdf5'
    l = LoadY(fn_load)
    print(l)
    l['loadstep'][2]['f_restart'] = 110
    l.save('t.yaml')
    l = LoadY('t.yaml')
    print(l)
    # print(l.load.get('loadstep'))
    # print(l.find_loadinc(1200))
    # print(l.clip_after_loadinc(1200))
    # print(l.load_append_refine(1200,4,2))
    # r = damask.Result(fn_hdf5)
    # print(r)
    l.get_sim_restart_n(fn_hdf5)
    # print(l.create_restart(4,2))
    # print(l.create_refine_before_fail(2))
    # l.create_restart(4,2).save('t.yaml')

def test_plot():
    fn_load = 'test_load.yaml'
    l = LoadY(fn_load)
    print(l)
    strain, stress, time, id_rec = l.unfold_loadsteps()
    print(time)
    print(id_rec)
    print(strain[0][0])
    print(stress[1][1])
    print(stress[1][1].shape)
    print(time.shape)
    print(id_rec.shape)
    print(id_rec)
    l.plot_load('test.png')
 

if __name__ == '__main__':
    # test_plot()
    test_operate()
 


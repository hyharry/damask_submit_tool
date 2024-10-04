import os
from pathlib import Path
from string import Template

from damask_submit_tool.load_util import LoadY

core_list_l = ['0-15', '80-95', '192-207']

def create_batch_script(fn_job_l, ws_label='ws15'):
    fn_templ = Path(__file__).parent / 'file_template/batch_submit.sh'
    with fn_templ.open('r') as f:
        templ = Template(f.read())

    submit_script = ''
    for fn_job in fn_job_l:
        submit_script += f'bash {fn_job} & \n'

    fn_submit = f'submit_{ws_label}.sh'
    with open(f'job/{fn_submit}','w') as f:
        run_str = templ.safe_substitute(submit_script=submit_script)
        f.write(run_str)

def create_normal_run_job(sim_cases, core_list_l, damask_bin, parent_dir, 
                          n_omp=2, ws_label='ws15'):
    ''' sim_cases = {dire1: {fn_geom: geom.vti, fn_load: load.yaml, fn_mat: m.yaml}} 
    parent_dir is on workstation! '''
    required_cpus = sum(-eval(cpus)+1 for cpus in core_list_l)
    print(f'create noraml run suite in {parent_dir} on N_cpu={required_cpus}, i.e. {core_list_l}')
    if not os.path.isdir('job'): os.mkdir('job')
    fn_templ = Path(__file__).parent / 'file_template/run.sh'
    with fn_templ.open('r') as f:
        templ = Template(f.read())
    sub_d = {}
    fn_job_l = []
    for sim_id, (cas,param_d) in enumerate(sim_cases.items()):
        dire = param_d['workdir']
        absdir = os.path.join(parent_dir, dire)
        fn_hdf = '_'.join((
            Path(param_d['fn_geom']).stem, 
            Path(param_d['fn_load']).stem, 
            Path(param_d['fn_mat']).stem, 
            'numerics'
            )) + '.hdf5'
        n_cpu = (-eval(core_list_l[sim_id])+1)//n_omp
        sub_d.update(
            simname     = f'sim case - {cas}',
            n_omp       = n_omp,
            n_cpu       = n_cpu,
            core_list   = core_list_l[sim_id],
            damask_bin  = damask_bin,
            workdir     = absdir,
            fn_geom     = param_d['fn_geom'],
            fn_load     = param_d['fn_load'],
            fn_material = param_d['fn_mat'],
            fn_hdf      = fn_hdf
        )
        run_str = templ.safe_substitute(**sub_d)
        fn_job = f'run_{sim_id:02d}_{dire}.sh'
        with open(f'job/{fn_job}','w') as f:
            f.write(run_str)
        fn_job_l.append(fn_job)
        print(f'create #{sim_id:02d} run in {dire} == done!')
    create_batch_script(fn_job_l, ws_label)

def create_restart_run_job(sim_cases, core_list_l, damask_bin, parent_dir, 
                           n_omp=2, ws_label='ws15'):
    ''' sim_cases = {dire1: {fn_geom: geom.vti, fn_load: load.yaml, fn_mat: m.yaml}} '''
    if not os.path.isdir('job'): os.mkdir('job')
    fn_templ = Path(__file__).parent / 'file_template/rerun.sh'
    with fn_templ.open('r') as f:
        templ = Template(f.read())
    sub_d = {}
    fn_job_l = []
    for sim_id, (cas,param_d) in enumerate(sim_cases.items()):
        dire = param_d['workdir']
        absdir = os.path.join(parent_dir, dire)
        fn_hdf = '_'.join((
            Path(param_d['fn_geom']).stem, 
            Path(param_d['fn_load']).stem, 
            Path(param_d['fn_mat']).stem, 
            'numerics'
            )) + '.hdf5'
        l = LoadY(os.path.join(dire, param_d['fn_load'])) # ! this implies one is in relative path view
        l.get_sim_restart_n(os.path.join(dire, fn_hdf))
        restart_inc = l.restart_n
        n_cpu = (-eval(core_list_l[sim_id])+1)//n_omp
        sub_d.update(
            simname     = f'sim case - {cas}',
            n_omp       = n_omp,
            n_cpu       = n_cpu,
            core_list   = core_list_l[sim_id],
            damask_bin  = damask_bin,
            workdir     = absdir,
            fn_geom     = param_d['fn_geom'],
            fn_load     = param_d['fn_load'],
            fn_material = param_d['fn_mat'],
            restart_inc = restart_inc,
            fn_hdf      = fn_hdf
        )
        run_str = templ.safe_substitute(**sub_d)
        fn_job = f'restart_{sim_id:02d}_{dire}.sh'
        with open(f'job/{fn_job}','w') as f:
            f.write(run_str)
        fn_job_l.append(fn_job)
        print(f'create #{sim_id:02d} restart in {dire} == done!')
    create_batch_script(fn_job_l, ws_label)


## SUBMIT JOB TO CLUSTER
def create_mpcdf_run_job(sim_cases, partition='p.cmfe', n_node=1, n_cpu=40, n_thread=1, 
                         mem='60G', run_hour=24, damask_bin='DAMASK', parent_dir='/path/to/parent'):
    """
    Create and submit MPCDF jobs using the run_mpcdf.bsh template.
    
    :param sim_cases: Dictionary of simulation cases
    :param partition: SLURM partition to use
    :param n_node: Number of nodes to request
    :param n_cpu: Number of CPUs to request
    :param n_thread: Number of threads per task
    :param mem: Memory to request
    :param run_hour: Run time in hours
    :param damask_bin: Path to DAMASK binary
    :param parent_dir: Parent directory for simulations
    """
    if not os.path.isdir('job'):
        os.mkdir('job')
    
    fn_templ = Path(__file__).parent / 'file_template/run_mpcdf.bsh'
    with fn_templ.open('r') as f:
        templ = Template(f.read())
    
    for sim_id, (cas, param_d) in enumerate(sim_cases.items()):
        dire = param_d['workdir']
        absdir = os.path.join(parent_dir, dire)
        fn_hdf = '_'.join((
            Path(param_d['fn_geom']).stem, 
            Path(param_d['fn_load']).stem, 
            Path(param_d['fn_mat']).stem, 
            'numerics'
        )) + '.hdf5'
        
        sub_d = {
            'simname': f'sim_case_{cas}',
            'partition': partition,
            'n_node': n_node,
            'n_cpu': n_cpu,
            'n_thread': n_thread,
            'mem': mem,
            'run_hour': run_hour,
            'damask_bin': damask_bin,
            'workdir': absdir,
            'fn_geom': param_d['fn_geom'],
            'fn_load': param_d['fn_load'],
            'fn_material': param_d['fn_mat'],
            'fn_hdf': fn_hdf
        }
        
        run_str = templ.safe_substitute(**sub_d)
        fn_job = f'mpcdf_run_{sim_id:02d}_{dire}.bsh'
        
        with open(f'job/{fn_job}', 'w') as f:
            f.write(run_str)
        
        # Submit the job
        print(f'Submitted job #{sim_id:02d} for {dire}')


def test():
    sim_cases = {'tes_cas': {'workdir': 'ww', 'fn_geom': 'aa', 'fn_load':'bb', 'fn_mat':'cc'}}
    create_normal_run_job(sim_cases, ['1-10'], 'bin', os.getcwd())
    create_restart_run_job(sim_cases, ['1-10'], 'bin', os.getcwd())


if __name__ == '__main__':
    # create_normal_run_job('a','b','c','d')
    test()
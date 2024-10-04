#!/usr/bin/env bash

WORK_DIR="${workdir}"
SIM_NAME="${simname}"

### send email before run ###
mail_body="job script $0 runs on $(hostname) with core list ${core_list}"
echo "${mail_body}" | mailx -s "START SIMULATION <${SIM_NAME}> (PID $$$$)" y.hu@mpie.de

### run sim ###
cd ${WORK_DIR}
export OMP_NUM_THREADS=${n_omp}
taskset -c ${core_list} mpirun -np ${n_cpu} --bind-to none \
    ${damask_bin} -g ${fn_geom} -l ${fn_load} -m ${fn_material} \
                  -n numerics.yaml > run.log 2> err.log

### simple post-process ###
# source ~/DAMASK_yi/env/DAMASK.zsh
# source ~/py_venv/damask_spack_python/bin/activate
# analyze_micro.py full ${fn_hdf} 
# short_log=$(tail -n 50 run.log)

###
wait

### send email after sim run ###
mail_body="job script $0 on $(hostname) done! core list ${core_list} free. \n !! run log !! \n ${short_log}"
echo "${mail_body}" | mailx -s "DONE SIMULATION <${SIM_NAME}> (PID $$$$)" y.hu@mpie.de
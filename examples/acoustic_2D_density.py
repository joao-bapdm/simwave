from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
import numpy as np


# available language options:
# c (sequential)
# cpu_openmp (parallel CPU)
# gpu_openmp (GPU)
# gpu_openacc (GPU)
compiler_options = {
    'c': {
        'cc': 'gcc',
        'language': 'c',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared'
    },
    'cpu_openmp': {
        'cc': 'gcc',
        'language': 'cpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared -fopenmp'
    },
    'gpu_openmp': {
        'cc': 'clang',
        'language': 'gpu_openmp',
        'cflags': '-O3 -fPIC -ffast-math -fopenmp \
                   -fopenmp-targets=nvptx64-nvidia-cuda \
                   -Xopenmp-target -march=sm_75'
    },
    'gpu_openacc': {
        'cc': 'pgcc',
        'language': 'gpu_openacc',
        'cflags': '-O3 -fPIC -acc:gpu -gpu=pinned -mp'
    },
}

selected_compiler = compiler_options['c']

# set compiler options
compiler = Compiler(
    cc=selected_compiler['cc'],
    language=selected_compiler['language'],
    cflags=selected_compiler['cflags']
)

# Velocity model
vel = np.zeros(shape=(512, 512), dtype=np.float32)
vel[:] = 1500.0
vel[100:] = 2000.0

# Density model
density = np.zeros(shape=(512, 512), dtype=np.float32)
density[:] = 10

# create the space model
space_model = SpaceModel(
    bounding_box=(0, 5120, 0, 5120),
    grid_spacing=(10, 10),
    velocity_model=vel,
    density_model=density,
    space_order=2,
    dtype=np.float32
)

# config boundary conditions
# (none,  null_dirichlet or null_neumann)
space_model.config_boundary(
    damping_length=0,
    boundary_condition=(
        "null_neumann", "null_dirichlet",
        "none", "null_dirichlet"
    ),
    damping_polynomial_degree=3,
    damping_alpha=0.002
)

# create the time model
time_model = TimeModel(
    space_model=space_model,
    tf=1.0,
    saving_stride=0
)

# create the set of sources
source = Source(
    space_model,
    coordinates=[(2560, 2560)],
    window_radius=4
)

# crete the set of receivers
receiver = Receiver(
    space_model=space_model,
    coordinates=[(2560, i) for i in range(0, 5120, 10)],
    window_radius=4
)

# create a ricker wavelet with 10hz of peak frequency
ricker = RickerWavelet(10.0, time_model)

# create the solver
solver = Solver(
    space_model=space_model,
    time_model=time_model,
    sources=source,
    receivers=receiver,
    wavelet=ricker,
    compiler=compiler
)

# run the forward
u_full, recv = solver.forward()

print("u_full shape:", u_full.shape)
plot_velocity_model(space_model.velocity_model)
plot_wavefield(u_full[-1])
plot_shotrecord(recv)

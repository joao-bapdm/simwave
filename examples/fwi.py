import scipy
import numpy as np
from scipy.optimize import minimize, show_options

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
from models import Camembert


class Optimizer:
    def __init__(self, options):
        self.options = options
        compiler_options = {
            'c': {
                'cc': 'gcc',
                'language': 'c',
                'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared'},
            'cpu_openmp': {
                'cc': 'gcc',
                'language': 'cpu_openmp',
                'cflags': '-O3 -fPIC -ffast-math -Wall -std=c99 -shared -fopenmp'},
            'gpu_openmp': {
                'cc': 'clang',
                'language': 'gpu_openmp',
                'cflags': '-O3 -fPIC -ffast-math -fopenmp \
                           -fopenmp-targets=nvptx64-nvidia-cuda \
                           -Xopenmp-target -march=sm_75'},
            'gpu_openacc': {
                'cc': 'pgcc',
                'language': 'gpu_openacc',
                'cflags': '-O3 -fPIC -acc:gpu -gpu=pinned -mp'},
        }
        selected_compiler = compiler_options['cpu_openmp']
        self.compiler = Compiler(
            cc=selected_compiler['cc'],
            language=selected_compiler['language'],
            cflags=selected_compiler['cflags']
        )

        self._recv_true = None
        self._stride = 1
        self._name = ""
        self.counter = 1

    @property
    def recv_true(self):
        return self._recv_true

    @recv_true.setter
    def recv_true(self, value):
        self._recv_true = value

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, value):
        self._stride = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def create_space_model(self, vel_model):
        space_options = self.options["space_options"]
        domain_size = space_options["domain_size"]
        spacing = space_options["spacing"]
        space_order = space_options["space_order"]
        if "float" in space_options["dtype"]:
            dtype = np.float64
        else:
            dtype = None
        space_model = SpaceModel(
            bounding_box=(0, domain_size, 0, domain_size),
            grid_spacing=(spacing, spacing),
            velocity_model=vel_model,
            space_order=space_order,
            dtype=dtype
        )
        bc_options = self.options["boundary_condition_options"]
        damping = bc_options["damping_size"]
        space_model.config_boundary(
            damping_length=(damping, damping, damping, damping),
            boundary_condition=bc_options["boundary_condition"],
            damping_polynomial_degree=bc_options["damping_polynomial_degree"],
            damping_alpha=bc_options["damping_alpha"]
        )
        return space_model

    def create_time_model(self, space_model, stride=0):
        time_model = TimeModel(
            space_model=space_model,
            tf=self.options["time_options"]["propagation_time"],
            dt=self.options["time_options"]["dt"],
            saving_stride=stride
        )
        return time_model

    def create_source(self, space_model, location, wr=4):
        source = Source(
            space_model,
            coordinates=location,
            window_radius=wr
        )
        return source

    def create_receivers(self, space_model, wr=4):
        receiver = Receiver(
            space_model=space_model,
            coordinates=self.options["receiver_options"]["coords"],
            window_radius=wr
        )
        return receiver

    def create_wavelet(self, time_model):
        return RickerWavelet(
            self.options["wavelet_options"]["frequency"], time_model)

    def create_solver(self, space_model, time_model,
                      source, receiver, wavelet):
        solver = Solver(
            space_model=space_model,
            time_model=time_model,
            sources=source,
            receivers=receiver,
            wavelet=wavelet,
            compiler=self.compiler
        )
        return solver

    def forward(self, vel_model, source_location, recv_true=None,
                stride=1, name="", plot=True, acq=False, seis=False):
        space_model = self.create_space_model(vel_model)
        time_model = self.create_time_model(space_model, stride=stride)
        source = self.create_source(space_model, source_location)
        receivers = self.create_receivers(space_model)
        wavelet = self.create_wavelet(time_model)
        solver = self.create_solver(
            space_model, time_model, source, receivers, wavelet)
        u, recv = solver.forward()
        if plot:
            if acq:
                plot_velocity_model(vel_model,
                                    sources=solver.sources.grid_positions,
                                    receivers=solver.receivers.grid_positions,
                                    file_name=name + "_model")
            elif self.counter:
                if name.endswith("0"):
                    plot_velocity_model(
                        vel_model, file_name=name + "_it_" + str(self.counter) + "_model")
            if seis:
                plot_shotrecord(
                    recv,
                    file_name=name +
                    "_seismogram",
                    solver=solver)
        if recv_true is not None and stride != 0:
            res = recv - recv_true
            f_obj = np.linalg.norm(
                np.trapz(res ** 2, x=time_model.time_values, axis=0), 2)
            grad = solver.gradient(u, res)
            return f_obj, grad
        else:
            return u, recv

    def _seq_fwd(self, vel_model, source_locations,
                 recv_true_all=None, stride=1, name="", plot=True):
        f_obj, grad = 0, 0
        for i, source_location in enumerate(source_locations):
            f, g = self.forward(
                vel_model, source_location, recv_true=recv_true_all[i], stride=1, name=name + "_shot_" + str(i), plot=plot)
            f_obj += f
            grad += g
        # print(f"At counter {self.counter}, fobj = {f_obj}")
        return f_obj, grad

    def sequential_forward(self, vel_model):
        f, g = self._seq_fwd(
            vel_model.reshape(self.options["space_options"]["shape"]),
            self.options["source_options"]["coords"],
            recv_true_all=self.recv_true,
            stride=self.stride,
            name=self.name)
        plot_velocity_model(g, file_name=Op.name + "_grad")
        self.counter += 0
        return f, self.crop_field(g).flatten()

    def plot_acquisition(self, vel_model, source_location, name=""):
        space_model = self.create_space_model(vel_model)
        time_model = self.create_time_model(space_model)
        source = self.create_source(space_model, source_location)
        receivers = self.create_receivers(space_model)
        wavelet = self.create_wavelet(time_model)
        solver = self.create_solver(
            space_model, time_model, source, receivers, wavelet)
        plot_velocity_model(vel_model,
                            sources=solver.sources.grid_positions,
                            receivers=solver.receivers.grid_positions,
                            file_name=name + "_model")

    def crop_field(self, u):
        space_options = self.options["space_options"]
        bc_options = self.options["boundary_condition_options"]
        spacing = space_options["spacing"]
        damping = bc_options["damping_size"]
        crop = damping // spacing
        return u[crop:-crop, crop:-crop]


if __name__ == '__main__':

    # Load ground truth
    Ca = Camembert()
    vp = Ca.create_velocity_model()
    src_loc = Ca.options["source_options"]["coords"]

    # Generate 'true' data
    Op = Optimizer(Ca.options)
    Op.plot_acquisition(vp, src_loc, name="acquisition")
    Op.recv_true = [
        Op.forward(
            vp,
            src,
            stride=0,
            name="ground_truth", seis=True)[1] for src in src_loc]
    Op.name = "Camembert"

    # Optimize
    vp_guess = 2700 * np.ones(vp.shape)
    # vp_guess = scipy.ndimage.gaussian_filter(vp, sigma=6)
    res = minimize(
        Op.sequential_forward,
        vp_guess.flatten(),
        method="L-BFGS-B",
        jac=True,
        bounds=[(2500, 3000) for _ in vp_guess.flatten()],
        options={
            "maxiter": 100,
            "disp": True,
            "iprint": 1})

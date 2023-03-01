import json
import logging

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, show_options

from simwave import (
    SpaceModel, TimeModel, RickerWavelet, Solver, Compiler, utils,
    Receiver, Source, plot_wavefield, plot_shotrecord, plot_velocity_model
)
from models import DevitoCamembert


class Optimizer:
    def __init__(self, optionsn name=""):
        self.options = options
        with open(".compiler_options.json") as f:
            compiler_options = json.load(f)
        selected_compiler = compiler_options['cpu_openmp']
        self.compiler = Compiler(
            cc=selected_compiler['cc'],
            language=selected_compiler['language'],
            cflags=selected_compiler['cflags']
        )

        self._recv_true = None
        self._stride = 1
        self._name = name
        self.counter = 0
        self.obj_fun = []

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
                stride=1, name="", seis=False):
        space_model = self.create_space_model(vel_model)
        time_model = self.create_time_model(space_model, stride=stride)
        source = self.create_source(space_model, source_location)
        receivers = self.create_receivers(space_model)
        wavelet = self.create_wavelet(time_model)
        solver = self.create_solver(
            space_model, time_model, source, receivers, wavelet)
        u, recv = solver.forward()
        if seis:
            plot_shotrecord(recv,file_name=name + "_seismogram", solver=solver)
        if recv_true is not None and stride != 0:
            res = recv - recv_true
            f_obj = 0.5 * np.linalg.norm(res) ** 2
            grad = solver.gradient(u, res)
            return f_obj, grad
        else:
            return u, recv

    def _seq_fwd(self, vel_model, source_locations,
                 recv_true_all=None, stride=1, name=""):
        """Sum objective function and gradient contribution from all sources"""
        f_obj, grad = 0, 0
        for i, source_location in enumerate(source_locations):
            f, g = self.forward(
                vel_model, source_location, recv_true=recv_true_all[i], stride=1, name=name + "_shot_" + str(i))
            f_obj += f
            grad += g
        self.obj_fun.append(f_obj)
        return f_obj, grad

    def sequential_forward(self, vel_model):
        f, g = self._seq_fwd(
            vel_model.reshape(self.options["space_options"]["shape"]),
            self.options["source_options"]["coords"],
            recv_true_all=self.recv_true,
            stride=self.stride,
            name=self.name)
        plot_velocity_model(g, file_name=Op.name + "_grad" + "_it_" + str(self.counter))
        plot_velocity_model(
            vel_model.reshape(
                self.options["space_options"]["shape"]),
            file_name=Op.name +
            "_model" +
            "_it_" +
            str(
                self.counter))
        self.counter += 1
        # print(f"objective function: {f}")
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

    def clip_values(self, u, v_min, v_max):
        return np.maximum(v_min, np.minimum(v_max, u))

    def optimize(self, vel_model, mode, scale=1):
        if mode in ["sequential"]:
            f, g = self.sequential_forward(vel_model)
            return f, - scale * g
        else:
            print("only sequential mode implemented so far")

    def callback(self, xk):
        plt.plot(self.obj_fun)
        plt.xlabel("Iterations")
        plt.ylabel("Misfit")
        plt.savefig("plots/obj_fun")


if __name__ == '__main__':

    utils.set_verbosity(logging.WARNING)
    
    # Load ground truth
    Ca = DevitoCamembert()
    vp = Ca.create_velocity_model()
    src_loc = Ca.options["source_options"]["coords"]

    # Generate 'true' data
    Op = Optimizer(Ca.options, name="Camembert")
    Op.plot_acquisition(vp, src_loc, name="acquisition")
    Op.recv_true = [
        Op.forward(
            vp,
            src,
            stride=0,
            name="shot" + str(src), seis=True)[1] for src in src_loc]

    # Optimize
    vp_guess = 2500 * np.ones(vp.shape)
    res = minimize(
        Op.optimize,
        vp_guess.flatten(),
        method="L-BFGS-B",
        args = ("sequential", 1e-8),
        jac=True,
        callback=Op.callback,
        bounds=[(2500, 3000) for _ in vp_guess.flatten()],
        options={
            "disp": True,
            "iprint": 1,
            "maxiter": 100,
            "maxls": 20})


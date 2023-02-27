import numpy as np

class Camembert:
    options = {
	"space_options": {
	    "domain_size": 1000,
	    "spacing": 10,
	    "space_order": 4,
	    "dtype": "np.float64"
	},
	"time_options": {
	    "propagation_time": 1.0,
	    "dt": 0.001
	},
	"boundary_condition_options": {
	    "boundary_condition": (
		"null_neumann", "null_dirichlet",
		"null_dirichlet", "null_dirichlet"
	    ),
	    "damping_size": 500,
	    "damping_polynomial_degree":3,
	    "damping_alpha":0.002
	},
	"source_options": {
	    "coords": [(100, i) for i in range(0, 1000, 100)]
	},
	"receiver_options": {
	    "coords": [(900, i) for i in range(0, 1000, 10)],
	},
	"wavelet_options": {
	    "frequency": 10.0
	},
    }   

    def create_velocity_model(self, grid_size=None):

        domain_size = Camembert.options["space_options"]["domain_size"]
        spacing = Camembert.options["space_options"]["spacing"]
        if "float64" in Camembert.options["space_options"]["dtype"]:
            dtype = np.float64
        else:
            dtype=None

        grid_size = domain_size // spacing + 1
        shape = (grid_size, grid_size)
        self.options["space_options"]["shape"] = shape

        vel = np.zeros(shape, dtype=dtype)
        vel[:] = 2500

        a, b = shape[0] / 2, shape[1] / 2
        z, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
        vel[z*z + x*x <= 15*15] = 3000

        return vel


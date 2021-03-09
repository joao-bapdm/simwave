from pywave.kernel.frontend import kws
import numpy as np


class Source:
    """
    Implement the structure of a source/receiver.

    Parameters
    ----------
    space_model : SpaceModel
        Space model object.
    coordinates : list of tuple, optional
        Physical coordinates (in meters) for this source.
    window_radius : int, optional
        Window half-width of the kaiser windowing function. Default is 4.
    """
    def __init__(self, space_model, coordinates=None, window_radius=4):

        if coordinates is None:
            self._coordinates = []
        else:
            # make sure, each source coordinate is float
            self._coordinates = [
                [np.float32(i) for i in coord] for coord in coordinates
            ]

        self._space_model = space_model

        self._window_radius = window_radius

    @property
    def space_model(self):
        """Corresponding space model."""
        return self._space_model

    @property
    def coordinates(self):
        """Physical coordinates (in meters) for this source."""
        return self._coordinates

    @property
    def window_radius(self):
        """Window half-width of the kaiser windowing function."""
        return self._window_radius

    @property
    def grid_positions(self):
        """List of point positions (in grid points) for this source set."""

        positions = []

        # for each position (one source or receiver)
        for coord in self.coordinates:
            # 2D coordinates
            if len(coord) == 2:
                zmin, zmax, xmin, xmax = self.space_model.bounding_box
                z_spacing, x_spacing = self.space_model.grid_spacing

                if not(zmin <= coord[0] <= zmax) or \
                   not(xmin <= coord[1] <= xmax):
                    raise Exception("Coordinates %s out of bounds." % coord)

                zpos = (coord[0] - zmin) / z_spacing
                xpos = (coord[1] - xmin) / x_spacing

                positions.append((zpos, xpos))
            # 3D coordinates
            elif len(coord) == 3:
                zmin, zmax, xmin, xmax, ymin, ymax = \
                                self.space_model.bounding_box

                z_spacing, x_spacing, y_spacing = self.space_model.grid_spacing

                if not(zmin <= coord[0] <= zmax) or \
                   not(xmin <= coord[1] <= xmax) or \
                   not(ymin <= coord[2] <= ymax):
                    raise Exception("Coordinates %s out of bounds." % coord)

                zpos = (coord[0] - zmin) / z_spacing
                xpos = (coord[1] - xmin) / x_spacing
                ypos = (coord[2] - ymin) / y_spacing

                positions.append((zpos, xpos, ypos))
            else:
                raise Exception("Dimension %d not supported." % len(coord))

        return positions

    @property
    def adjusted_grid_positions(self):
        """
        Adjusted source positions (in grid points) according to nbl extension
        and space order halo. In this case, the origin is shifted.
        """
        # adjust the origin (nbl + halo size)
        nbl = list(self.space_model.nbl[::2])
        halo = list(self.space_model.halo_size[::2])
        origin = [nbl[i] + halo[i] for i in range(len(nbl))]

        adjusted_positions = [
            tuple(src[i] + origin[i] for i in range(len(src)))
            for src in self.grid_positions
        ]

        return adjusted_positions

    @property
    def interpolated_points_and_values(self):
        """
        Calculate the point interval of a source/receiver and their values.
        Use Kaiser windowing sinc for the source positioning.

        Returns
        ----------
        ndarray
            1D Numpy array with [begin_point_axis1, end_point_axis1,
            .., begin_point_axisN, end_point_axisN].
        ndarray
            1D Numpy array with [source_values_axis1, .., source_values_axisN].
        """

        points = np.array([], dtype=np.uint)
        values = np.array([], dtype=np.float32)

        for position in self.adjusted_grid_positions:
            # apply kasier window to interpolate the source/receiver
            # in a region of grid points
            p, v = kws.get_source_points(
                grid_shape=self.space_model.extended_shape,
                source_location=position,
                half_width=self.window_radius
            )
            points = np.append(points, p)
            values = np.append(values, v)

        return points, values

    @property
    def count(self):
        """Return the number of sources/receives."""
        return len(self.coordinates)

    def add(self, location):
        """
        Add a new source/receiver location.

        Parameters
        ----------
        position : tuple of float
            Source/receiver position (in meters) along each axis.
        """
        # make sure coordinates in location are float
        location = tuple([np.float32(i) for i in location])
        self._coordinates.append(location)

    def remove_all(self):
        """
        Remove all sources/receiver coordinates.
        """
        self._coordinates = []


# a receiver is also a type of source
Receiver = Source


class Wavelet:
    """
    Implement a wavelet for the source.

    Parameters
    ----------
    function : object
        Function (expression) that creates the wavelet.
    kwargs : dict
        key word arguments of the function.
    """
    def __init__(self, function, **kwargs):
        self._function = function
        self._kwargs = kwargs

    @property
    def function(self):
        """Function (expression) that creates the wavelet."""
        return self._function

    @property
    def kwargs(self):
        """key word arguments of the function."""
        return self._kwargs

    @property
    def values(self):
        """Wavelet values."""
        return self.function(**self.kwargs)


class RickerWavelet(Wavelet):
    """
    Implement a ricker wavelet.

    Parameters
    ----------
    peak_frequency : float
        Peak frequency for the wavelet in Hz.
    time_model: TimeModel
        Time model object.
    amplitude : float, optional
        Amplitude of the wavelet. Default is 1.0.
    """
    def __init__(self, peak_frequency, time_model, amplitude=1):

        self._peak_frequency = peak_frequency
        self._time_model = time_model
        self._amplitude = amplitude

        super().__init__(
            self._function,
            peak_frequency=self.peak_frequency,
            time_model=self.time_model,
            amplitude=self.amplitude
        )

    @property
    def peak_frequency(self):
        """Peak frequency of the wavelet in Hz."""
        return self._peak_frequency

    @property
    def time_model(self):
        """Corresponding time model."""
        return self._time_model

    @property
    def amplitude(self):
        """Amplitude of the wavelet."""
        return self._amplitude

    def _function(self, peak_frequency, time_model, amplitude):
        """Function that generates the ricker wavelet."""
        t0 = 1 / peak_frequency
        r = np.pi * peak_frequency * (time_model.time_values - t0)
        return amplitude * (1 - 2.0 * r**2) * np.exp(-r**2)

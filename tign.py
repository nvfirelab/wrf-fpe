"""
Module producing TIGN_G, the ignition-time array WRF-SFIRE consumes.

The method's whole output is a time-of-arrival field, and TIGN_G *is* a
time-of-arrival field, so export is arithmetic:

    t_ign(x) = t_i + s(x) * (t_f - t_i)

in seconds from the simulation start, on the fire grid. Two things make it
more than a multiplication, and both come from how SFIRE reads the array
rather than from the method.

The contract (module_fr_sfire_core.F, init_no_fire)
----------------------------------------------------
With fire_perimeter_time > 0 or fire_tign_in_time > 0, SFIRE initialises

    lfn(i,j) = tign(i,j) - time_now ,   time_now = time_start + dt

and then crashes with

    'init_no_fire: ignition time must be after the end of the first time step'

unless lfn > 0 at *every* fire-mesh node. So every value in the array -
inside the first perimeter, between the perimeters, and outside the last one
- must be strictly greater than the end of the first time step. There is no
sentinel convention in the Fortran: a cell that never ignites is simply one
whose time never arrives. `sfire_init_check` is that loop, reimplemented, so
the array can be rejected here rather than by a queued job.

What to put outside the last perimeter
--------------------------------------
lfn is used directly as the level-set function, with no reinitialisation. The
no-fire branch of the same routine sets it to twice the domain diameter - a
distance - and the level-set equation is homogeneous in lfn, so what matters
is not the magnitude but the shape. A constant sentinel gives a cliff at the
final perimeter; continuing the arrival time outward as distance over a
speed gives the signed-distance-like field the solver was written for. The
default is the continuation. The constant is kept as an option because some
workflows expect it, and either passes initialisation.

Chaining a series
-----------------
Each consecutive pair is evolved on the shared grid, mapped into absolute
seconds, and the elementwise minimum taken across segments: a cell's
ignition time is the earliest any segment claims it. Before that, each
segment's final mask is compared cell for cell with the next segment's
initial mask, and the series is checked for nesting. A mismatch means the
observations disagree with each other, and the minimum would hide it rather
than resolve it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from pyproj import CRS, Transformer

from evolution import GaugeEvolution

# Fortran REAL is single precision; the array is written as float32.
TIGN_DTYPE = np.float32


# ----------------------------------------------------------------------
# The SFIRE-side check
# ----------------------------------------------------------------------

def sfire_init_check(tign, dt, time_start=0.0):
    """
    init_no_fire's acceptance loop, reimplemented.

    Every fire-mesh node must satisfy tign - (time_start + dt) > 0, or SFIRE
    calls crash() during initialisation with the message quoted in the module
    docstring. Non-finite values fail too - a NaN compares false against
    everything, which the Fortran `.not. lfn > 0.` catches as a violation.

    Parameters:
    -----------
    tign : numpy.ndarray
        Ignition times in seconds from the simulation start
    dt : float
        The model time step, in seconds
    time_start : float
        Simulation start in the array's clock, normally zero

    Returns:
    --------
    report : dict
        passes, the number of violating nodes, the array minimum, the value
        it must exceed, and the crash message SFIRE would have printed
    """
    tign = np.asarray(tign, dtype=float)
    time_now = float(time_start) + float(dt)
    lfn = tign - time_now
    ok = np.isfinite(lfn) & (lfn > 0.0)
    violations = int((~ok).sum())
    finite = tign[np.isfinite(tign)]
    minimum = float(finite.min()) if finite.size else float("nan")
    report = {
        "passes": violations == 0,
        "violations": violations,
        "nodes": int(tign.size),
        "non_finite": int((~np.isfinite(tign)).sum()),
        "min_tign": minimum,
        "time_now": time_now,
        "needs_shift_s": max(time_now - minimum, 0.0) if np.isfinite(minimum)
                         else float("nan"),
        "message": None,
    }
    if violations:
        report["message"] = (
            "init_no_fire: ignition time must be after the end of the first "
            f"time step (min tign={minimum:g} <= time_now={time_now:g}; "
            f"{violations} nodes)")
    return report


# ----------------------------------------------------------------------
# The arrival-time field
# ----------------------------------------------------------------------

@dataclass
class ArrivalTime:
    """
    Absolute ignition time on the grid, in seconds from a simulation start.

    Attributes:
    -----------
    seconds : numpy.ndarray
        Ignition time per cell; NaN where no observation says the cell burned
    grid : GridSpec
        The grid the field lives on
    simulation_start : datetime
        The instant the seconds count from, timezone-aware
    first_mask, last_mask : numpy.ndarray
        The first and last observed perimeters, rasterised; everything in
        `first_mask` carries the first observation's time, everything outside
        `last_mask` is NaN
    segments : list of dict
        Provenance of each evolved pair, in order, including the segment's
        ellipse (or None under a Euclidean metric)
    speed_hint : float or None
        A spread rate in m/s to continue the field outward with, if the
        segments offered one
    perimeters : list of Perimeter
        Every observation in order, on the grid's CRS - one more than there
        are segments - so the field can be drawn against what it was fitted to
    """

    seconds: np.ndarray
    grid: object
    simulation_start: datetime
    first_mask: np.ndarray
    last_mask: np.ndarray
    segments: list = field(default_factory=list)
    speed_hint: float = None
    perimeters: list = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_evolution(cls, evolution, simulation_start):
        """
        Step 6 for a single pair.

        Parameters:
        -----------
        evolution : GaugeEvolution
            The evolved pair
        simulation_start : datetime
            When the WRF-SFIRE run begins. Must be timezone-aware and no
            later than the initial observation.

        Returns:
        --------
        arrival : ArrivalTime
        """
        return cls.chain([evolution], simulation_start)

    @classmethod
    def chain(cls, evolutions, simulation_start):
        """
        Step 7: consecutive pairs on one grid, combined by elementwise minimum.

        Parameters:
        -----------
        evolutions : sequence of GaugeEvolution
            One per consecutive pair, in time order, all on the same grid
        simulation_start : datetime
            When the WRF-SFIRE run begins

        Returns:
        --------
        arrival : ArrivalTime
        """
        evolutions = list(evolutions)
        if not evolutions:
            raise ValueError("nothing to chain")
        simulation_start = _aware(simulation_start)
        _check_consecutive(evolutions)

        grid = evolutions[0].grid
        seconds = np.full(grid.shape, np.nan)
        segments, speeds = [], []
        for k, ev in enumerate(evolutions):
            t0 = (ev.initial.time - simulation_start).total_seconds()
            if t0 < 0:
                raise ValueError(
                    f"segment {k} begins {-t0:g} s before the simulation start "
                    f"({ev.initial.time.isoformat()} < "
                    f"{simulation_start.isoformat()})")
            t = ev.arrival_seconds(t0)
            # A cell claimed by more than one segment ignites at the earliest
            # claim. The overlap is exactly the shared perimeter, where the
            # earlier segment says t_f and the later one says t_i = t_f, so
            # the minimum is a tie there and a real choice nowhere else.
            seconds = np.fmin(seconds, t)
            ellipse = getattr(ev.metric, "ellipse", None)
            segments.append({
                "initial": ev.initial.time, "final": ev.final.time,
                "t0_s": t0, "interval_s": ev.interval,
                "metric": ev.metric.describe(),
                "metric_provenance": ev.metric_provenance,
                "ellipse": ellipse,
                "endpoint_jaccard": ev.endpoint_jaccard(),
                "monotonicity_violations": ev.monotonicity_violations,
            })
            if ellipse is not None:
                speeds.append(ellipse.head)

        return cls(seconds=seconds, grid=grid, simulation_start=simulation_start,
                   first_mask=evolutions[0].mask_start,
                   last_mask=evolutions[-1].mask_end, segments=segments,
                   speed_hint=float(np.mean(speeds)) if speeds else None,
                   perimeters=[ev.initial for ev in evolutions] + [evolutions[-1].final])

    @classmethod
    def from_series(cls, series, grid, simulation_start, nesting_tolerance=0.0,
                    **kwargs):
        """
        Evolve every consecutive pair of a series and chain them.

        The nesting check runs first, on the rasterised masks the solver will
        see, and a series that is not nested is refused: the method treats
        every observation as a hard constraint, and two constraints that
        contradict each other are a data problem to fix upstream.

        Parameters:
        -----------
        series : PerimeterSeries
            Two or more timed observations
        grid : GridSpec
            The shared grid
        simulation_start : datetime
            When the WRF-SFIRE run begins
        nesting_tolerance : float
            Fraction of an earlier perimeter allowed to fall outside the
            next one before the series is refused. Zero by default; a
            percent or so absorbs rasterisation noise on real data.
        **kwargs
            Passed to every GaugeEvolution - metric, lb, wind_mps, clock...

        Returns:
        --------
        arrival : ArrivalTime
        """
        if len(series) < 2:
            raise ValueError("a series needs at least two observations to evolve")
        series = series.on_grid(grid)
        bad = [r for r in series.check_nesting(grid)
               if r["fraction_lost"] > max(float(nesting_tolerance), 1e-9)]
        if bad:
            worst = max(bad, key=lambda r: r["fraction_lost"])
            raise ValueError(
                f"{len(bad)} pair(s) are not nested; worst loses "
                f"{100 * worst['fraction_lost']:.2f}% of the earlier perimeter "
                f"between {worst['earlier']} and {worst['later']}. Fix the "
                "observations rather than chaining over the disagreement.")
        evolutions = [GaugeEvolution(a, b, grid, **kwargs) for a, b in series.pairs()]
        return cls.chain(evolutions, simulation_start)

    # ------------------------------------------------------------------
    # The exported array
    # ------------------------------------------------------------------

    @property
    def t_first(self):
        """Seconds from the simulation start to the first observation."""
        return self.segments[0]["t0_s"]

    @property
    def observation_times(self):
        """Seconds from the simulation start to every observation, in order."""
        return [seg["t0_s"] for seg in self.segments] + [self.t_last]

    @property
    def t_last(self):
        """Seconds from the simulation start to the last observation."""
        last = self.segments[-1]
        return last["t0_s"] + last["interval_s"]

    def tign_g(self, outside="extend", inside="flat", speed=None, sentinel=None):
        """
        The array SFIRE reads, with every cell given a finite value.

        Parameters:
        -----------
        outside : str
            What cells beyond the last perimeter get.
            "extend"   - t_last + distance to the last perimeter / speed, so
                         lfn continues outward as a scaled signed distance
            "sentinel" - one constant, `sentinel`
        inside : str
            What cells already burned at the first observation get.
            "flat"   - t_first, everywhere inside. Honest: nothing earlier
                       was observed. But it ignites the whole interior at
                       one instant, and if t_first is short it fails the
                       first-time-step check for every one of those cells.
            "extend" - t_first - distance inside the first perimeter / speed,
                       a burn-in ramp that reaches the centre earliest. A
                       spin-up convenience, not an observation.
        speed : float, optional
            Metres per second for either extension. Defaults to the mean
            head rate of the segments' ellipses; required if none had one.
        sentinel : float, optional
            The constant for outside="sentinel"

        Returns:
        --------
        tign : numpy.ndarray
            float32 array of the grid's shape, finite everywhere
        """
        t = np.array(self.seconds, dtype=float, copy=True)
        if outside == "sentinel":
            if sentinel is None:
                raise ValueError("outside='sentinel' needs sentinel=")
            if sentinel <= self.t_last:
                raise ValueError(
                    f"sentinel {sentinel:g} s is not later than the last "
                    f"observation at {self.t_last:g} s")
            t[~self.last_mask] = float(sentinel)
        elif outside == "extend":
            speed = self._speed(speed)
            phi = self.grid.sdf(self.last_mask)
            beyond = ~self.last_mask
            t[beyond] = self.t_last + phi[beyond] / speed
        else:
            raise ValueError(f"unknown outside={outside!r}")

        if inside == "extend":
            speed = self._speed(speed)
            phi = self.grid.sdf(self.first_mask)
            t[self.first_mask] = self.t_first + phi[self.first_mask] / speed
        elif inside != "flat":
            raise ValueError(f"unknown inside={inside!r}")

        if not np.isfinite(t).all():
            raise RuntimeError("the arrival field still has non-finite cells")
        return t.astype(TIGN_DTYPE)

    def _speed(self, speed):
        """The extension speed, from the argument or the segments' ellipses."""
        if speed is None:
            speed = self.speed_hint
        if speed is None:
            raise ValueError(
                "no speed to extend the field with: the segments ran under a "
                "Euclidean metric, so pass speed= in m/s")
        if speed <= 0:
            raise ValueError(f"speed must be positive; got {speed}")
        return float(speed)

    def check(self, dt, **tign_kwargs):
        """
        Would SFIRE accept this field at time step dt? Raises if not.

        Parameters:
        -----------
        dt : float
            The model time step, in seconds
        **tign_kwargs
            Passed to tign_g()

        Returns:
        --------
        report : dict
            The sfire_init_check report, when it passes
        """
        report = sfire_init_check(self.tign_g(**tign_kwargs), dt)
        if not report["passes"]:
            raise ValueError(
                f"SFIRE would reject this array: {report['message']}. Start "
                f"the simulation at least {report['needs_shift_s']:.0f} s "
                "earlier, or use inside='extend' only if that ramp is "
                "acceptable as spin-up.")
        return report

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def lonlat(self):
        """
        Longitude and latitude of every cell centre.

        Returns:
        --------
        lon, lat : numpy.ndarray
            Arrays of the grid's shape, in degrees
        """
        X, Y = self.grid.meshgrid()
        to_wgs = Transformer.from_crs(self.grid.crs, CRS.from_epsg(4326),
                                      always_xy=True)
        lon, lat = to_wgs.transform(X, Y)
        return np.asarray(lon), np.asarray(lat)

    def to_geotiff(self, path, tign=None, **tign_kwargs):
        """
        Write the array as a single-band GeoTIFF in the grid's CRS.

        Rows are flipped on the way out: the grid stores row 0 at miny and
        GeoTIFFs are north-up.

        Parameters:
        -----------
        path : str
            Output path
        tign : numpy.ndarray, optional
            A prepared array; built from tign_kwargs otherwise
        """
        import rasterio

        tign = self.tign_g(**tign_kwargs) if tign is None else tign
        with rasterio.open(path, "w", driver="GTiff", height=self.grid.ny,
                           width=self.grid.nx, count=1, dtype="float32",
                           crs=self.grid.crs.to_wkt(), transform=self.grid.transform,
                           compress="deflate") as dst:
            dst.write(np.flipud(tign).astype(np.float32), 1)
            dst.update_tags(units="s", simulation_start=self.simulation_start.isoformat(),
                            description="TIGN_G ignition time")
        return path

    def to_netcdf(self, path, tign=None, **tign_kwargs):
        """
        Write a standalone NetCDF carrying TIGN_G with cell-centre lon/lat.

        This is the method's grid, not a WRF fire mesh; it is the archival
        form and the input to `write_wrfinput`, which does the resampling.
        Row 0 is the southern edge, as WRF stores its arrays.

        Parameters:
        -----------
        path : str
            Output path
        tign : numpy.ndarray, optional
            A prepared array; built from tign_kwargs otherwise
        """
        import netCDF4

        tign = self.tign_g(**tign_kwargs) if tign is None else tign
        lon, lat = self.lonlat()
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("Time", 1)
            ds.createDimension("south_north", self.grid.ny)
            ds.createDimension("west_east", self.grid.nx)
            ds.setncatts({
                "title": "TIGN_G ignition time from observed perimeters",
                "simulation_start": self.simulation_start.isoformat(),
                "crs_wkt": self.grid.crs.to_wkt(),
                "cell_size_m": float(self.grid.cell_size),
                "first_observation_s": float(self.t_first),
                "last_observation_s": float(self.t_last),
                "segments": "\n".join(
                    f"{s['initial'].isoformat()} -> {s['final'].isoformat()}: "
                    f"{s['metric']}" for s in self.segments),
            })
            v = ds.createVariable("TIGN_G", "f4", ("Time", "south_north", "west_east"),
                                  zlib=True)
            v.units, v.description = "s", "ignition time on ground"
            v[0] = tign
            for name, data, desc in (("XLONG", lon, "longitude of cell centres"),
                                     ("XLAT", lat, "latitude of cell centres")):
                w = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"),
                                      zlib=True)
                w.units, w.description = "degrees", desc
                w[0] = data
            for name, data in (("x", self.grid.x_coords), ("y", self.grid.y_coords)):
                w = ds.createVariable(name, "f8", ("west_east" if name == "x"
                                                   else "south_north",))
                w.units = "m"
                w[:] = data
        return path

    def write_wrfinput(self, path, tign=None, method="nearest", dt=None,
                       variable="TIGN_G", **tign_kwargs):
        """
        Resample onto an existing wrfinput file's fire mesh and overwrite TIGN_G.

        The fire mesh's node positions are read from FXLONG/FXLAT, projected
        into the grid's CRS, and the array is sampled there. Nodes that fall
        outside the grid get the array's maximum - the far end of the outward
        extension, or the sentinel - which is the same "not for a long time"
        both modes mean. The last sr_x columns and sr_y rows of a WRF fire
        array are padding that SFIRE never reads; they are filled the same
        way, so nothing in the array is left at whatever it was before.

        The array is checked against init_no_fire's loop before anything is
        written, at the file's own DT unless `dt` overrides it, and nothing is
        written if the check fails.

        Parameters:
        -----------
        path : str
            An existing wrfinput_d0X, opened for update
        tign : numpy.ndarray, optional
            A prepared array; built from tign_kwargs otherwise
        method : str
            "nearest" or "linear" sampling
        dt : float, optional
            Model time step for the check; read from the file's DT otherwise
        variable : str
            The fire-mesh variable to write

        Returns:
        --------
        report : dict
            The init check on the array as written, plus mesh bookkeeping
        """
        import netCDF4
        from scipy.interpolate import RegularGridInterpolator

        tign = self.tign_g(**tign_kwargs) if tign is None else tign
        with netCDF4.Dataset(path, "r+") as ds:
            for name in ("FXLONG", "FXLAT", variable):
                if name not in ds.variables:
                    raise KeyError(f"{path} has no {name}; is it a wrfinput with a fire mesh?")
            if dt is None:
                if "DT" not in ds.ncattrs():
                    raise ValueError("the file has no DT attribute; pass dt=")
                dt = float(ds.getncattr("DT"))

            fxlong = np.asarray(ds.variables["FXLONG"][0], dtype=float)
            fxlat = np.asarray(ds.variables["FXLAT"][0], dtype=float)
            target = ds.variables[variable]
            if target.shape[1:] != fxlong.shape:
                raise ValueError(
                    f"{variable} {target.shape[1:]} and FXLONG {fxlong.shape} "
                    "disagree on the fire mesh shape")

            to_grid = Transformer.from_crs(CRS.from_epsg(4326), self.grid.crs,
                                           always_xy=True)
            fx, fy = to_grid.transform(fxlong, fxlat)
            fx, fy = np.asarray(fx), np.asarray(fy)

            fill = float(np.nanmax(tign))
            sampler = RegularGridInterpolator(
                (self.grid.y_coords, self.grid.x_coords), np.asarray(tign, dtype=float),
                method=method, bounds_error=False, fill_value=fill)
            # Padding nodes carry no meaningful coordinates - often zero -
            # and land far outside the grid, so the fill handles them too.
            valid = np.isfinite(fx) & np.isfinite(fy)
            out = np.full(fxlong.shape, fill)
            out[valid] = sampler(np.column_stack([fy[valid], fx[valid]]))
            out = out.astype(TIGN_DTYPE)

            report = sfire_init_check(out, dt)
            if not report["passes"]:
                raise ValueError(
                    f"not written: SFIRE would reject this array at DT={dt:g}: "
                    f"{report['message']}")

            inside = (self.grid.minx <= fx) & (fx <= self.grid.maxx) & \
                     (self.grid.miny <= fy) & (fy <= self.grid.maxy)
            target[0] = out
            ds.setncattr("TIGN_G_source",
                         f"perimeter evolution, simulation_start="
                         f"{self.simulation_start.isoformat()}, "
                         f"{len(self.segments)} segment(s), {method} resampling")

        report.update({"dt": dt, "mesh_shape": fxlong.shape,
                       "nodes_inside_grid": int(inside.sum()),
                       "nodes_filled": int((~inside).sum()), "fill": fill})
        return report

    def summary(self):
        """
        What went into the field, for a run log.

        Returns:
        --------
        summary : dict
        """
        finite = self.seconds[np.isfinite(self.seconds)]
        return {
            "grid": repr(self.grid),
            "simulation_start": self.simulation_start.isoformat(),
            "segments": len(self.segments),
            "first_observation_s": self.t_first,
            "last_observation_s": self.t_last,
            "cells_igniting": int(np.isfinite(self.seconds).sum()),
            "cells_burned_at_start": int(self.first_mask.sum()),
            "min_s": float(finite.min()) if finite.size else None,
            "max_s": float(finite.max()) if finite.size else None,
            "speed_hint_mps": self.speed_hint,
        }

    def __repr__(self):
        return (f"ArrivalTime({len(self.segments)} segment(s), "
                f"{self.t_first:.0f}-{self.t_last:.0f} s from "
                f"{self.simulation_start.isoformat()}, {self.grid!r})")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _aware(when):
    """A timezone-aware datetime, from a datetime or an ISO 8601 string."""
    if isinstance(when, str):
        text = when.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        when = datetime.fromisoformat(text)
    if not isinstance(when, datetime):
        raise TypeError(f"simulation_start must be a datetime; got {type(when).__name__}")
    if when.tzinfo is None:
        raise ValueError("simulation_start must be timezone-aware (use UTC)")
    return when.astimezone(timezone.utc)


def _check_consecutive(evolutions):
    """Every segment shares its grid with the first and its masks with its neighbours."""
    grid_key = evolutions[0].grid.key()
    for k, ev in enumerate(evolutions):
        if ev.grid.key() != grid_key:
            raise ValueError(f"segment {k} is on a different grid from segment 0")
        if ev.initial.time is None or ev.final.time is None:
            raise ValueError(f"segment {k} has an untimed perimeter")
    for k, (a, b) in enumerate(zip(evolutions[:-1], evolutions[1:])):
        if a.final.time != b.initial.time:
            raise ValueError(
                f"segment {k} ends at {a.final.time.isoformat()} but segment "
                f"{k + 1} begins at {b.initial.time.isoformat()}")
        if not np.array_equal(a.mask_end, b.mask_start):
            differ = int((a.mask_end ^ b.mask_start).sum())
            raise ValueError(
                f"segment {k}'s final mask and segment {k + 1}'s initial mask "
                f"differ in {differ} cells; the observations disagree with "
                "each other and chaining would hide it")


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def main(argv=None):
    """Evolve a perimeter series and write TIGN_G."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolve a series of perimeter shapefiles into a TIGN_G "
                    "ignition-time array for WRF-SFIRE.")
    parser.add_argument("perimeters", nargs="+", help="perimeter shapefiles, in time order")
    parser.add_argument("--simulation-start", required=True,
                        help="ISO 8601 UTC instant the run begins, e.g. 2024-07-23T00:00:00Z")
    parser.add_argument("--dt", type=float, required=True,
                        help="model time step in seconds, for the acceptance check")
    parser.add_argument("--cell-size", type=float, default=25.0)
    parser.add_argument("--padding", type=float, default=0.1)
    parser.add_argument("--lb", type=float, default=None)
    parser.add_argument("--wind-mps", type=float, default=None)
    parser.add_argument("--infer-from-geometry", action="store_true")
    parser.add_argument("--outside", choices=("extend", "sentinel"), default="extend")
    parser.add_argument("--inside", choices=("flat", "extend"), default="flat")
    parser.add_argument("--speed", type=float, default=None,
                        help="extension speed in m/s; defaults to the ellipse head rate")
    parser.add_argument("--sentinel", type=float, default=None)
    parser.add_argument("--netcdf", default=None, help="standalone NetCDF to write")
    parser.add_argument("--geotiff", default=None, help="GeoTIFF to write")
    parser.add_argument("--wrfinput", default=None,
                        help="existing wrfinput_d0X whose TIGN_G to overwrite")
    parser.add_argument("--method", choices=("nearest", "linear"), default="nearest")
    args = parser.parse_args(argv)

    from perimeter import PerimeterSeries

    series = PerimeterSeries.from_shapefiles(args.perimeters)
    grid = series.common_grid(cell_size=args.cell_size, padding=args.padding)
    arrival = ArrivalTime.from_series(
        series, grid, args.simulation_start, lb=args.lb, wind_mps=args.wind_mps,
        infer_from_geometry=args.infer_from_geometry)

    for key, value in arrival.summary().items():
        print(f"  {key:24} {value}")
    for k, seg in enumerate(arrival.segments):
        print(f"  segment {k}: {seg['metric_provenance']}")

    tign_kwargs = dict(outside=args.outside, inside=args.inside,
                       speed=args.speed, sentinel=args.sentinel)
    report = arrival.check(args.dt, **tign_kwargs)
    print(f"  init_no_fire check     passes at DT={args.dt:g} "
          f"(min tign {report['min_tign']:.1f} s > {report['time_now']:g} s)")

    tign = arrival.tign_g(**tign_kwargs)
    if args.netcdf:
        print("  wrote", arrival.to_netcdf(args.netcdf, tign=tign))
    if args.geotiff:
        print("  wrote", arrival.to_geotiff(args.geotiff, tign=tign))
    if args.wrfinput:
        r = arrival.write_wrfinput(args.wrfinput, tign=tign, method=args.method,
                                   dt=args.dt)
        print(f"  wrote {args.wrfinput}: {r['nodes_inside_grid']} fire-mesh nodes "
              f"sampled, {r['nodes_filled']} filled with {r['fill']:.0f} s; "
              f"init check passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

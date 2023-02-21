import logging
from pathlib import Path
from typing import Optional, Union

import click
from tqdm.autonotebook import tqdm

from cartes.utils.cache import CacheResults
from traffic.core import Flight, Traffic

Directory = Path
global_airport: Optional[str] = None


def cache_results(path: Directory, filename: str) -> CacheResults:
    return CacheResults(
        cache_dir=path,
        hashing=lambda *args, **kwargs: filename,
        reader=Traffic.from_file,
        writer=lambda t, path: t.to_pickle(path),
    )


def merge_all_data(path: Directory) -> Traffic:
    all_ = []
    for filename in tqdm(sorted(path.glob("*")), desc="reading data"):
        all_.append(Traffic.from_file(filename))

    merged = Traffic.from_flights(all_)  # type: ignore
    assert merged is not None
    return merged


def crop_airborne(flight: Flight) -> Optional[Flight]:
    assert global_airport is not None
    # all_ = flight.aligned_on_ils(global_airport).all() #landings
    all_ = flight.takeoff_from_runway(global_airport).all() #takeoffs
    if all_ is None:
        return None
    # return flight.before(all_.stop) #landings
    return flight.after(all_.start) #takeoffs


def processing(
    input: Traffic, airport: str, max_workers: int, sampling_rate: str
) -> Traffic:
    global global_airport
    global_airport = airport

    return (
        input.assign_id()
        .resample("1s")
        .filter()
        .resample(sampling_rate)
        # .has(f"aligned_on_{airport}") #landings
        .cumulative_distance(compute_gs = False, compute_track = False)
        .diff("cumdist")
        .filter(cumdist_diff=13, strategy=None)
        .query("cumdist_diff.notnull()")
        .pipe(crop_airborne)
        .eval(desc="", max_workers=max_workers)
        .drop(
            columns=[
                "alert",
                "day",
                "destination",
                "firstseen",
                "geoaltitude",
                "hour",
                "lastseen",
                "onground",
                "spi",
                "track_unwrapped",
                # "start",
                # "stop",
                "distance",
                "cumdist",
                # "compute_gs",
                # "compute_track",
                "cumdist_diff",
            ]
        )
    )


@click.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("-s", "--sampling-rate", default="2s", help="Final sampling rate")
@click.option("-w", "--max-workers", default=8, help="Number of cores")
@click.option("-v", "--verbose", count=True, help="Verbosity level")
def main(
    path: Union[str, Directory],
    verbose: int,
    max_workers: int,
    sampling_rate: str,
):

    logger = logging.getLogger()
    if verbose == 1:
        logger.setLevel(logging.INFO)
    elif verbose > 1:
        logger.setLevel(logging.DEBUG)

    path = Path(path)
    airport = path.stem
    click.echo(f"Processing {path} for airport {airport}")

    pkl_cache = cache_results(path, f"{airport}_history.pkl.gz")
    merged = pkl_cache(merge_all_data)(path)

    click.echo(f"Processing ...")
    processed = processing(
        merged, airport, max_workers=max_workers, sampling_rate=sampling_rate
    )
    processed.to_parquet(path.parent / f"{airport}_dataset.parquet.gz")


if __name__ == "__main__":
    main()
# fmt: off
from argparse import ArgumentParser
from pathlib import Path

from traffic.core import Traffic
from traffic.core.projection import EuroPP

# fmt: on
# n_samples


def cli_main() -> None:
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument(
        "--input_path",
        dest="input_path",
        type=str,
    )

    parser.add_argument(
        "--samples",
        dest="n_samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--jobs",
        dest="n_jobs",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
    )
    parser.add_argument(
        "--dp",
        dest="douglas_peucker_coeff",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    # ------------
    # preprocessing
    # ------------

    input_traffic = Traffic.from_file(Path(args.input_path))

    t: Traffic = (
        input_traffic
        .assign_id()
        .query("track == track") #remove nan
        .query("groundspeed == groundspeed")
        .query("altitude == altitude")
        # .resample(args.n_samples)
        .unwrap()
        .eval(max_workers=args.n_jobs, desc="")
    )

    t = t.compute_xy(projection=EuroPP())

    if args.douglas_peucker_coeff is not None:
        print("Simplification...")
        t = t.simplify(tolerance=1e3).eval(desc="")

    t = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in t
    )

    t.to_pickle(Path(args.output_path))


if __name__ == "__main__":
    cli_main()

from datetime import datetime, timezone
from pathlib import Path

from traffic.data import airports, opensky

output_dir = Path(".")
thresholds = dict(LFP0=50)

for airport in ["LFPO"]:
    for month in [6,7]:
        for day in range(1, 32):

            if month == 6 and day == 31:
                continue

            if not (output_dir / airport).exists():
                (output_dir / airport).mkdir(parents=True)

            date = datetime(2022, month, day, 0, 0, 0, tzinfo=timezone.utc)

            filename = f"{airport}_{date:%Y-%m-%d}_history.pkl.gz"
            print(filename)

            if (output_dir / airport / filename).exists():
                continue

            distance_threshold = thresholds.get(airport, 50)

            # history = opensky.history(date, arrival_airport=airport)
            history = opensky.history(date, departure_airport=airport)
            if history is not None:
                output = (
                    history.iterate_lazy()
                    .resample("1s")
                    .distance(airports[airport])
                    .query(f"distance < {distance_threshold}")
                    .eval(desc="", max_workers=70)
                )

                if output is not None:
                    output.to_pickle(output_dir / airport / filename)

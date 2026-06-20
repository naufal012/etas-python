"""
Example script to fetch real earthquake data from the International 
Seismological Centre (ISC) and prepare it for ETAS modeling.
"""

import pandas as pd
from etas import search_isc, catalog

def main():
    print("Fetching earthquake data for Japan from ISC...")
    print("This may take a minute depending on the connection...")
    
    # 1. Fetch earthquake data from the ISC API for a specific region (e.g. Japan)
    # Returns a Pandas DataFrame with 'date', 'time', 'lat', 'long', 'mag'
    df = search_isc(
        start_year=2010, start_month=1, start_day=1,
        end_year=2010, end_month=12, end_day=31,
        searchshape="RECT",
        lat_bot=30.0, lat_top=45.0,
        long_left=130.0, long_right=145.0,
        mag_min=4.5
    )

    print(f"\nSuccessfully fetched {len(df)} events!")
    print("\nFirst 5 rows of the raw data:")
    print(df.head())

    print("\nSaving raw data to 'japan_quakes_sample.csv'...")
    df.to_csv("japan_quakes_sample.csv", index=False)

    # 2. Feed it directly into the ETAS catalog processor
    print("\nProcessing the data into an ETAS Catalog object...")
    cat = catalog(
        data=df,
        time_begin="2010-01-01",
        study_start="2010-03-01",
        study_length=250, # days
        mag_threshold=4.5,
        dist_unit="degree"
    )
    
    print("\nCatalog processing complete! The catalog is ready for ETAS fitting:")
    print(f"Number of events in study region: {len(cat.revents)}")

if __name__ == "__main__":
    main()

"""
Fetch earthquake catalogs from the International Seismological Centre (ISC) API.

Equivalent of search.isc.R from the R ETAS package.
"""

import pandas as pd
import urllib.request

def search_isc(start_year=1900, start_month=1, start_day=1,
               end_year=2018, end_month=12, end_day=31,
               searchshape="RECT",
               lat_bot=None, lat_top=None,
               long_left=None, long_right=None,
               lat_ctr=None, long_ctr=None,
               radius=None, dist_units="deg",
               dep_min=0, dep_max=100, nulldep=True,
               mag_min=4.0, mag_max=None,
               mag_type='MB', mag_agcy=None):
    """
    Search and download earthquake data from the ISC database.
    
    Returns a pandas DataFrame formatted for ETAS modeling (containing 
    date, time, lat, long, mag columns).
    """
    if searchshape not in ["RECT", "CIRC"]:
        raise ValueError("searchshape must be either 'RECT' or 'CIRC'")
    
    if dist_units not in ["deg", "km"]:
        raise ValueError("dist_units must be either 'deg' or 'km'")
        
    url = "http://www.isc.ac.uk/cgi-bin/web-db-v4?request=COMPREHENSIVE&out_format=CATCSV"
    
    url += f"&searchshape={searchshape}"
    if searchshape == "RECT":
        url += f"&bot_lat={lat_bot if lat_bot is not None else ''}"
        url += f"&top_lat={lat_top if lat_top is not None else ''}"
        url += f"&left_lon={long_left if long_left is not None else ''}"
        url += f"&right_lon={long_right if long_right is not None else ''}"
    else:
        url += f"&ctr_lat={lat_ctr if lat_ctr is not None else ''}"
        url += f"&ctr_lon={long_ctr if long_ctr is not None else ''}"
        url += f"&radius={radius if radius is not None else ''}"
        url += f"&max_dist_units={dist_units}"
        
    url += "&srn=&grn="
    url += f"&start_year={start_year}&start_month={start_month}&start_day={start_day}&start_time=00%3A00%3A00"
    url += f"&end_year={end_year}&end_month={end_month}&end_day={end_day}&end_time=00%3A00%3A00"
    
    url += f"&min_dep={dep_min}&max_dep={dep_max}"
    if nulldep:
        url += "&null_dep=on"
        
    url += f"&min_mag={mag_min}&max_mag={mag_max if mag_max is not None else ''}"
    url += f"&req_mag_type={mag_type if mag_type is not None else ''}"
    url += f"&req_mag_agcy={mag_agcy if mag_agcy is not None else ''}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8').splitlines()
    except Exception as e:
        raise Exception(f"Failed to fetch data from ISC: {e}")
        
    if len(content) <= 27:
        raise Exception("Could not download the data: please try again")
        
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(content):
        if "--EVENT--" in line:
            start_idx = i
        if "STOP" in line:
            end_idx = i
            
    if start_idx == -1 or end_idx == -1:
        raise Exception("Failed to parse data payload from ISC response")
        
    data_lines = content[start_idx + 1 : end_idx - 1]
    
    parsed_data = []
    for line in data_lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 11:
            row = {
                'eventid': parts[0],
                'date': parts[3],
                'time': parts[4],
                'lat': parts[5],
                'long': parts[6],
                'depth': parts[7],
                'mag': parts[11] if len(parts) > 11 else None
            }
            parsed_data.append(row)
            
    df = pd.DataFrame(parsed_data)
    
    # Convert numerical columns
    for col in ['lat', 'long', 'depth', 'mag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows with missing critical information
    df = df.dropna(subset=['date', 'time', 'lat', 'long', 'mag'])
    
    # Filter bounds just in case
    if searchshape == "RECT":
        if long_left is not None and long_right is not None:
            df = df[(df['long'] >= long_left) & (df['long'] <= long_right)]
        if lat_bot is not None and lat_top is not None:
            df = df[(df['lat'] >= lat_bot) & (df['lat'] <= lat_top)]
            
    return df

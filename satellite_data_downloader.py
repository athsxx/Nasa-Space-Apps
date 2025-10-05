#!/usr/bin/env python3
"""
Satellite Data Downloader - Local Synthetic Feature Generator

Purpose:
- Provide non-empty, useful satellite-like features (NO2 column, O3 column, AOD, cloud_fraction, sza)
- Works entirely offline using the existing ground dataset to synthesize daily overpass features
- Saves per-parameter CSVs and a merged features file under output_dir

Notes:
- This is a pragmatic stand-in for real APIs (e.g., Sentinel-5P, MODIS). It preserves realistic signal
  relationships (e.g., column NO2 ~ ground NO2 daily mean, AOD ~ PM/humidity) and adds noise.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SatelliteDataDownloader:
    """Local synthetic satellite data generator for air quality modeling"""

    def __init__(self, output_dir: str = "satellite_data", base_csv: str = "tcn_models/real_air_quality_time_series.csv"):
        self.output_dir = Path(output_dir)
        self.base_csv = Path(base_csv)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output files
        self.no2_file = self.output_dir / "sat_no2.csv"
        self.o3_file = self.output_dir / "sat_o3.csv"
        self.aod_file = self.output_dir / "sat_aod.csv"
        self.features_file = self.output_dir / "satellite_features.csv"

    def _load_base(self) -> pd.DataFrame:
        if not self.base_csv.exists():
            raise FileNotFoundError(f"Base dataset not found: {self.base_csv}")
        df = pd.read_csv(self.base_csv)
        # Ensure timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Try to infer from common alternatives
            for c in ['time', 'date', 'datetime']:
                if c in df.columns:
                    df['timestamp'] = pd.to_datetime(df[c])
                    break
            if 'timestamp' not in df.columns:
                raise ValueError("No timestamp column found in base dataset")
        # Ensure location_id exists
        if 'location_id' not in df.columns:
            # Create a dummy single-location if missing
            df['location_id'] = 0
        return df

    @staticmethod
    def _sza_approx(lat: float, ts: pd.Timestamp) -> float:
        """Approximate Solar Zenith Angle using a simple seasonal/hour proxy (0-90 deg)."""
        # Day of year factor (max sun at mid-year)
        doy = ts.timetuple().tm_yday
        season = np.cos(2 * np.pi * (doy - 172) / 365.25)  # ~1 at Jun 21
        # Hour factor: minimum at ~solar noon (13:00 typical overpass)
        hour = ts.hour if not pd.isna(ts.hour) else 13
        hour_term = abs(hour - 13) / 12  # 0 at 13:00, 1 at 1am/1pm offset extremes
        # Latitude term: higher lat -> higher SZA
        lat_term = min(1.0, abs(lat) / 60.0)
        sza = 20 + 50 * (0.5 * (1 - season) + 0.3 * hour_term + 0.2 * lat_term)
        return float(np.clip(sza, 5, 85))

    def download_data(self, start_date: str, end_date: str, region: str = "North America") -> bool:
        """Generate synthetic daily satellite features from the base dataset within the date range."""
        print(f"SatelliteDataDownloader: Generating satellite-like features {start_date} -> {end_date} [{region}]")
        base = self._load_base()

        # Filter date range
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        base = base[(base['timestamp'] >= sd) & (base['timestamp'] <= ed)].copy()
        if base.empty:
            print("[WARN] No base records in date range; nothing to generate")
            return False

        # Derive daily index (simulate 1 overpass per day per site)
        base['date'] = base['timestamp'].dt.floor('D')

        # Fill optional columns
        for col in ['Latitude', 'Longitude']:
            if col not in base.columns:
                base[col] = 0.0

        # Aggregate to daily per location
        agg = {
            'NO2': 'mean',
            'O3': 'mean',
            'PM2_5': 'mean',
            'PM10': 'mean',
            'humidity': 'mean',
            'temperature': 'mean',
            'Latitude': 'first',
            'Longitude': 'first',
        }
        for k in list(agg.keys()):
            if k not in base.columns:
                agg.pop(k)

        daily = base.groupby(['location_id', 'date'], as_index=False).agg(agg)

        # Simulate satellite overpass time ~13:00 local
        daily['overpass_time'] = daily['date'] + pd.Timedelta(hours=13)

        # NO2 column (mol/m^2) proxy from ground NO2 (ppb) daily mean
        if 'NO2' in daily.columns:
            no2_signal = daily['NO2'].fillna(daily['NO2'].median())
        else:
            no2_signal = pd.Series(0.0, index=daily.index)
        rng = np.random.default_rng(42)
        daily['no2_column'] = np.clip(1e-5 + (no2_signal.clip(lower=0) / 1000.0) * (0.8 + 0.4 * rng.random(len(daily))), 0, None)

        # O3 column (DU) proxy from ground O3 (ppm) daily mean
        if 'O3' in daily.columns:
            o3_signal = daily['O3'].fillna(daily['O3'].median())
        else:
            o3_signal = pd.Series(0.0, index=daily.index)
        daily['o3_column'] = np.clip(200 + (o3_signal * 20) * (0.9 + 0.3 * rng.random(len(daily))), 150, 400)

        # AOD proxy from PM/humidity (dimensionless 0-2)
        pm_proxy = None
        if 'PM2_5' in daily.columns:
            pm_proxy = daily['PM2_5'].fillna(daily['PM2_5'].median())
        elif 'PM10' in daily.columns:
            pm_proxy = (daily['PM10'] / 2.0).fillna(daily['PM10'].median() / 2.0)
        else:
            pm_proxy = pd.Series(10.0, index=daily.index)
        humidity = daily['humidity'] if 'humidity' in daily.columns else pd.Series(60.0, index=daily.index)
        daily['aod'] = np.clip(0.05 + (pm_proxy / 60.0) * (0.7 + 0.4 * rng.random(len(daily))) * (0.8 + 0.004 * humidity.fillna(60.0)), 0.01, 3.0)

        # Cloud fraction (0-1), random with mild seasonality by month
        month = daily['date'].dt.month
        daily['cloud_fraction'] = np.clip(0.25 + 0.15 * np.sin(2 * np.pi * (month / 12.0)) + 0.2 * rng.random(len(daily)), 0, 1)

        # Solar zenith angle
        lat = daily['Latitude'] if 'Latitude' in daily.columns else pd.Series(0.0, index=daily.index)
        daily['sza'] = [self._sza_approx(float(la), ts) for la, ts in zip(lat, daily['overpass_time'])]

        # Quality flag (0-1)
        daily['qa_value'] = np.clip(0.6 + 0.35 * (1 - daily['cloud_fraction']) - 0.002 * (daily['sza'] - 40).abs() + 0.05 * rng.random(len(daily)), 0, 1)

        # Save parameter-specific files
        cols_common = ['location_id', 'overpass_time']
        (daily[cols_common + ['no2_column', 'qa_value']]
         .rename(columns={'overpass_time': 'timestamp'})
         .to_csv(self.no2_file, index=False))
        (daily[cols_common + ['o3_column', 'qa_value']]
         .rename(columns={'overpass_time': 'timestamp'})
         .to_csv(self.o3_file, index=False))
        (daily[cols_common + ['aod', 'cloud_fraction', 'sza', 'qa_value']]
         .rename(columns={'overpass_time': 'timestamp'})
         .to_csv(self.aod_file, index=False))

        # Save merged features
        merged = daily[['location_id', 'overpass_time', 'no2_column', 'o3_column', 'aod', 'cloud_fraction', 'sza', 'qa_value']].copy()
        merged = merged.rename(columns={'overpass_time': 'timestamp'})
        merged.to_csv(self.features_file, index=False)

        print(f"[OK] Saved: {self.no2_file.name}, {self.o3_file.name}, {self.aod_file.name}, {self.features_file.name}")
        return True

    def get_no2_data(self) -> pd.DataFrame:
        """Get NO2 satellite-like data (CSV-backed)."""
        return pd.read_csv(self.no2_file) if self.no2_file.exists() else pd.DataFrame()

    def get_o3_data(self) -> pd.DataFrame:
        """Get O3 satellite-like data (CSV-backed)."""
        return pd.read_csv(self.o3_file) if self.o3_file.exists() else pd.DataFrame()

    def get_aod_data(self) -> pd.DataFrame:
        """Get AOD satellite-like data (CSV-backed)."""
        return pd.read_csv(self.aod_file) if self.aod_file.exists() else pd.DataFrame()

    def get_features(self) -> pd.DataFrame:
        """Get merged satellite feature table (per location_id, daily timestamp)."""
        return pd.read_csv(self.features_file) if self.features_file.exists() else pd.DataFrame()

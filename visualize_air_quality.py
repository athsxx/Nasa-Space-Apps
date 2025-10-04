#!/usr/bin/env python3
"""
North American Air Quality Data Visualization
Creates visualizations from the processed EPA data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_north_america_visualizations():
    """Create visualizations of North American air quality data"""
    
    # Load the summary data
    try:
        df = pd.read_csv('north_america_air_quality_summary.csv')
        print(f"Loaded data for {len(df)} state-pollutant combinations")
    except FileNotFoundError:
        print("Summary file not found. Run epa_data_processor.py first.")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Average pollutant concentrations by state (heatmap)
    plt.subplot(2, 3, 1)
    pivot_data = df.pivot(index='State', columns='Pollutant', values='Average_Concentration')
    
    # Normalize the data for better visualization (different units)
    pivot_normalized = pivot_data.div(pivot_data.max())
    
    sns.heatmap(pivot_normalized, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Normalized Concentration'})
    plt.title('Air Quality Heatmap by State\n(Normalized Concentrations)', fontsize=12, fontweight='bold')
    plt.xlabel('Pollutant')
    plt.ylabel('State')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0, fontsize=8)
    
    # 2. Top 10 states for each pollutant
    plt.subplot(2, 3, 2)
    pollutants = df['Pollutant'].unique()
    colors = sns.color_palette("husl", len(pollutants))
    
    for i, pollutant in enumerate(pollutants):
        pollutant_data = df[df['Pollutant'] == pollutant].nlargest(5, 'Average_Concentration')
        plt.barh(range(i*6, i*6+5), pollutant_data['Average_Concentration'], 
                color=colors[i], alpha=0.7, label=f'{pollutant}')
        
        for j, (idx, row) in enumerate(pollutant_data.iterrows()):
            plt.text(row['Average_Concentration'], i*6+j, f"  {row['State'][:8]}", 
                    va='center', fontsize=8)
    
    plt.title('Top 5 States by Pollutant\n(Highest Average Concentrations)', fontweight='bold')
    plt.xlabel('Average Concentration')
    plt.ylabel('Ranking')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Monitoring coverage (sites per state)
    plt.subplot(2, 3, 3)
    state_sites = df.groupby('State')['Monitoring_Sites'].sum().sort_values(ascending=False).head(15)
    
    plt.bar(range(len(state_sites)), state_sites.values, color='lightblue', edgecolor='navy')
    plt.title('Monitoring Network Coverage\n(Top 15 States by Total Sites)', fontweight='bold')
    plt.xlabel('State')
    plt.ylabel('Total Monitoring Sites')
    plt.xticks(range(len(state_sites)), state_sites.index, rotation=45, ha='right')
    
    # 4. Pollutant comparison across all states
    plt.subplot(2, 3, 4)
    for pollutant in pollutants:
        pollutant_data = df[df['Pollutant'] == pollutant]['Average_Concentration']
        plt.hist(pollutant_data, alpha=0.6, label=pollutant, bins=20)
    
    plt.title('Distribution of Average Concentrations\nby Pollutant', fontweight='bold')
    plt.xlabel('Average Concentration (normalized scale)')
    plt.ylabel('Number of States')
    plt.legend()
    plt.yscale('log')
    
    # 5. Data coverage (measurements per state)
    plt.subplot(2, 3, 5)
    state_measurements = df.groupby('State')['Total_Measurements'].sum().sort_values(ascending=False).head(10)
    
    plt.barh(range(len(state_measurements)), state_measurements.values, color='lightgreen', edgecolor='darkgreen')
    plt.title('Data Volume\n(Top 10 States by Total Measurements)', fontweight='bold')
    plt.xlabel('Total Measurements (millions)')
    plt.ylabel('State')
    
    # Format x-axis to show millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.yticks(range(len(state_measurements)), state_measurements.index)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    
    # Create summary stats
    total_measurements = df['Total_Measurements'].sum()
    total_sites = df['Monitoring_Sites'].sum()
    states_covered = df['State'].nunique()
    pollutants_tracked = df['Pollutant'].nunique()
    
    summary_text = f"""
    NORTH AMERICAN AIR QUALITY DATASET
    
    📊 Coverage Statistics:
    • States/Territories: {states_covered}
    • Pollutants Monitored: {pollutants_tracked}
    • Total Monitoring Sites: {total_sites:,}
    • Total Measurements: {total_measurements/1e6:.1f} Million
    
    🏭 Pollutants Included:
    • SO₂ (Sulfur Dioxide)
    • O₃ (Ozone) 
    • NO₂ (Nitrogen Dioxide)
    • CO (Carbon Monoxide)
    
    📅 Time Period: 2024 (Full Year)
    
    🔬 Data Source: 
    EPA Air Quality System (AQS)
    https://aqs.epa.gov/
    
    ⚡ Analysis includes:
    • Hourly measurements
    • Geographic coverage across North America
    • Seasonal and temporal patterns
    • State-level comparisons
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('north_america_air_quality_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as: north_america_air_quality_analysis.png")
    
    # Show the plot
    plt.show()
    
    # Create detailed state ranking
    create_state_rankings(df)

def create_state_rankings(df):
    """Create detailed state rankings for each pollutant"""
    
    pollutants = df['Pollutant'].unique()
    
    print(f"\n{'='*60}")
    print("DETAILED STATE RANKINGS BY POLLUTANT")
    print(f"{'='*60}")
    
    for pollutant in pollutants:
        print(f"\n🏭 {pollutant} Rankings (by Average Concentration):")
        print("-" * 50)
        
        pollutant_data = df[df['Pollutant'] == pollutant].sort_values(
            'Average_Concentration', ascending=False
        ).head(10)
        
        for i, (_, row) in enumerate(pollutant_data.iterrows(), 1):
            print(f"{i:2d}. {row['State']:<20} "
                  f"{row['Average_Concentration']:>8.4f} {row['Units']:<5} "
                  f"({row['Monitoring_Sites']:>2d} sites, {row['Total_Measurements']:>8,} measurements)")
    
    print(f"\n{'='*60}")

def main():
    """Main function"""
    print("North American Air Quality Data Visualization")
    print("=" * 50)
    
    create_north_america_visualizations()

if __name__ == "__main__":
    main()

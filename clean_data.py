import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- 1. CONFIGURATION ---
# "Path(__file__).parent" allows the script to find files in the same folder as itself
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "processed_data"

def find_file(keyword):
    """
    Searches for a CSV file that starts with the keyword.
    This handles cases like 'Kidney_Data.csv' or 'Kidney_Data_copy.csv' automatically.
    """
    # Look for any .csv file starting with the keyword
    candidates = list(BASE_DIR.glob(f"{keyword}*.csv"))
    
    if not candidates:
        return None
    
    # If multiple found, prefer the one with "copy" if user mentioned it, 
    # otherwise just take the first one found.
    found_file = candidates[0]
    print(f"   🔎 Found matching file: {found_file.name}")
    return found_file

def clean_numeric(x):
    """Turns strings like '1,200' into numbers like 1200."""
    if isinstance(x, str):
        clean_str = x.replace(',', '').replace('"', '').strip()
        if not clean_str or clean_str.lower() == 'nan':
            return 0
        return float(clean_str)
    return x

def process_race_data(file_path):
    print("   🧹 Cleaning Race Data...")
    
    # Header is usually on row 1 (index 1) for these files
    df = pd.read_csv(file_path, header=1)
    
    # Rename key columns
    df.rename(columns={df.columns[0]: 'Donor_Type', df.columns[1]: 'Race_Ethnicity'}, inplace=True)
    
    # Remove 'To Date' total column if present
    if "To Date" in df.columns:
        df.drop(columns=["To Date"], inplace=True)
        
    # Convert 'Wide' format (Years as columns) to 'Long' format (Year as row)
    id_vars = ['Donor_Type', 'Race_Ethnicity']
    value_vars = [c for c in df.columns if str(c).strip().isdigit()]
    
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Count')
    
    # Clean numbers
    df_long['Count'] = df_long['Count'].apply(clean_numeric)
    df_long['Year'] = pd.to_numeric(df_long['Year'])
    
    # Remove summary rows
    df_clean = df_long[df_long['Race_Ethnicity'] != 'All Races/Ethnicities'].copy()
    
    # Simplify Race Names
    df_clean['Race_Ethnicity'] = df_clean['Race_Ethnicity'].replace({
        'White, Non-Hispanic': 'White',
        'Black, Non-Hispanic': 'Black',
        'Asian, Non-Hispanic': 'Asian',
        'Hispanic/Latino': 'Hispanic'
    })
    
    return df_clean

def process_center_data(file_path, map_file):
    print("   🧹 Cleaning Center Data...")
    
    # Define columns manually because raw headers are messy
    cols = ['Center_Long', 'Citizenship', 'Payment_Category', 'Blank', 'Total', 'Deceased', 'Living']
    df = pd.read_csv(file_path, skiprows=2, names=cols, header=None)
    
    # Fill in missing Center names (Forward Fill)
    df['Center_Long'] = df['Center_Long'].ffill()
    df['Center_Code'] = df['Center_Long'].str.slice(0, 8) # Extract "PAAE-TX1"
    
    # Clean numbers
    for c in ['Total', 'Deceased', 'Living']:
        df[c] = df[c].apply(clean_numeric)
        
    # Filter junk rows
    df = df[df['Center_Long'] != 'All Centers']
    df = df[df['Payment_Category'] != 'All Primary Payers']
    
    # Merge with Mapping File (if it exists)
    if map_file:
        print(f"   📍 Merging with map: {map_file.name}")
        mapping = pd.read_csv(map_file)
        mapping['Center_Code'] = mapping['Center_Code'].str.strip()
        df['Center_Code'] = df['Center_Code'].str.strip()
        df = df.merge(mapping[['Center_Code', 'Region', 'Urban']], on='Center_Code', how='left')
    else:
        print("   ⚠️ Mapping file not found. Regions will be blank.")

    return df

def main():
    print("🚀 STARTING DATA CLEANUP...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. FIND FILES using "Smart Match" (Starts with...)
    race_file = find_file("Kidney")         # Finds "Kidney_Transplants..._copy.csv"
    center_file = find_file("Transplants")  # Finds "Transplants_By_..._copy.csv"
    map_file = find_file("center_mapping")
    
    # 2. PROCESS FILES
    if race_file:
        try:
            df_race = process_race_data(race_file)
            df_race.to_csv(OUTPUT_DIR / "clean_race_data.csv", index=False)
            print("   ✅ Created: clean_race_data.csv")
        except Exception as e:
            print(f"   ❌ Error in Race Data: {e}")
    else:
        print("   ❌ Could not find the Kidney/Race file.")

    if center_file:
        try:
            df_center = process_center_data(center_file, map_file)
            df_center.to_csv(OUTPUT_DIR / "clean_center_data.csv", index=False)
            print("   ✅ Created: clean_center_data.csv")
        except Exception as e:
            print(f"   ❌ Error in Center Data: {e}")
    else:
         print("   ❌ Could not find the Transplants/Center file.")

    print("\n🏁 DONE. You can now run the dashboard app.")

if __name__ == "__main__":
    main()
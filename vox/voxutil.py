import pandas as pd
import numpy as np

def resample_lithology_uscs(input_file, output_file, interval):
    # Load the data
    lithology_df = pd.read_csv(input_file)
    
    resampled_data = []

    # Process each borehole separately
    for borehole_id in lithology_df['borehole_id'].unique():
        borehole_data = lithology_df[lithology_df['borehole_id'] == borehole_id]
        borehole_data = borehole_data.sort_values(by='start_depth').reset_index(drop=True)
        
        start_depth = borehole_data['start_depth'].min()
        end_depth = borehole_data['end_depth'].max()
        
        # Create new depth intervals
        new_depths = np.arange(start_depth, end_depth, interval)
        
        for new_depth in new_depths:
            # Find the last known value at the current depth
            previous_data = borehole_data[borehole_data['start_depth'] <= new_depth]
            if not previous_data.empty:
                last_known = previous_data.iloc[-1]
                resampled_data.append({
                    'borehole_id': borehole_id,
                    'start_depth': new_depth,
                    'end_depth': new_depth + interval,
                    'start_elevation': last_known['start_elevation'] - (new_depth - last_known['start_depth']),
                    'end_elevation': last_known['start_elevation'] - (new_depth - last_known['start_depth']) - interval,
                    'lithology': last_known['lithology']
                })
    
    # Create a new DataFrame and save to CSV
    resampled_df = pd.DataFrame(resampled_data)
    resampled_df.to_csv(output_file, index=False)
    
    print(f"Resampled data saved to {output_file}")


def main():
    
    # Example usage
    input_file = r'../borehole-lith/lithology.csv'
    output_file = 'resampled_lithology.csv'
    interval = 0.5  # Resample to every 0.5 depth units
    
    resample_lithology_uscs(input_file, output_file, interval)

    

if __name__ == "__main__":
    main()
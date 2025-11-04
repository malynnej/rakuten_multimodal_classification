"""
Class Balancing Module - Simplified Version
@Johann - 2025-10-19

Supports 4 methods:
1. 'none' - No balancing
2. 'undersample' - Random undersampling
3. 'oversample' - Random oversampling
4. 'smote' - SMOTE synthetic samples

Change method in params.yaml
"""

# Execution time
import time

class ClassBalancing:
    def __init__(self, method='none'):
        self.method = method
        print("#" * 50)
        print(f"\nClass Balancing initialized: method = '{self.method}'\n")
        print("#" * 50 + "\n")
    
    def process(self, dataframe):
        """
        Apply class balancing based on configured method.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Must have 'text' and 'prdtypecode' columns
        
        Returns:
        --------
        dataframe : pd.DataFrame
            Balanced dataframe (or unchanged if method='none')
        """

        # Start execution timer
        t_start = time.time()
        
        if self.method == 'none':
            print("Class balancing: DISABLED (method='none')\n")
            return dataframe
        
        print(f"Applying class balancing: {self.method}")
        
        df = dataframe.copy()
        
        # Prepare data (keep text column, separate target)
        X = df.drop('prdtypecode', axis=1)
        y = df['prdtypecode']
        
        # Select sampler based on method
        if self.method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=42)
            
        elif self.method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
            
        elif self.method == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=42, k_neighbors=3)
        
        else:
            print(f"⚠️  Unknown method: '{self.method}'. No balancing applied.\n")
            return dataframe
        
        # Apply resampling
        print(f"Original size: {len(df):,} samples")
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"Resampled size: {len(y_resampled):,} samples")
        
        # Reconstruct dataframe
        df_resampled = X_resampled.copy()
        df_resampled['prdtypecode'] = y_resampled
        
        print(f"✓ Class balancing complete: {self.method}\n")

        # End execution timer
        t_end = time.time()
        t_exec = t_end-t_start
        print(f"Execution time: {t_exec} seconds.")
        
        return df_resampled
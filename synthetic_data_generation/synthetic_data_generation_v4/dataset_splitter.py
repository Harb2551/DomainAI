"""
Dataset Splitter for V4 Synthetic Domain Generation
Combines normal and edge cases, then splits into stratified train/val/test datasets
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
from creative_llm_generator import CreativeLLMGenerator
from creative_edge_case_generator import CreativeEdgeCaseGenerator

class DatasetSplitter:
    def __init__(self, output_dir: str = "./", model_id: str = "us.deepseek.r1-v1:0", provider: str = None, region: str = "us-east-1"):
        """Initialize dataset splitter"""
        self.output_dir = output_dir
        self.normal_generator = CreativeLLMGenerator(model_id=model_id, provider=provider, region=region)
        self.edge_generator = CreativeEdgeCaseGenerator(model_id=model_id, provider=provider, region=region)
        
        # Output file paths - updated to use datasets_v4 folder
        datasets_v4_dir = os.path.join(output_dir, "datasets", "datasets_v4")
        self.normal_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_normal.csv")
        self.edge_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_edge_cases.csv")
        self.eval_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_eval.csv")
        self.train_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_train.csv")
        self.val_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_val.csv")
        self.test_path = os.path.join(datasets_v4_dir, "synthetic_domain_dataset_test.csv")

    def generate_normal_cases(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate normal cases by calling CreativeLLMGenerator with multithreading"""
        print(f"Generating {n_samples} normal cases using CreativeLLMGenerator...")
        
        # Use the multithreaded combinatorial approach from CreativeLLMGenerator
        cases = self.normal_generator.generate_creative_normal_cases(n_samples)
        
        # Convert to DataFrame
        df_normal = pd.DataFrame(cases)
        
        print(f"Successfully generated {len(df_normal)} normal cases with multithreading")
        return df_normal

    def generate_edge_cases(self) -> pd.DataFrame:
        """Generate edge cases using CreativeEdgeCaseGenerator"""
        print("Generating edge cases...")
        edge_cases = self.edge_generator.generate_all_edge_cases()
        
        # Convert to DataFrame
        df_edge = pd.DataFrame(edge_cases)
        return df_edge

    def combine_datasets(self, df_normal: pd.DataFrame, df_edge: pd.DataFrame) -> pd.DataFrame:
        """Combine normal and edge case datasets"""
        print("Combining normal and edge case datasets...")
        
        # Ensure both datasets have the same columns
        required_columns = ['business_description', 'ideal_domain', 'label']
        
        for col in required_columns:
            if col not in df_normal.columns:
                df_normal[col] = ""
            if col not in df_edge.columns:
                df_edge[col] = ""
        
        # Select only required columns
        df_normal = df_normal[required_columns]
        df_edge = df_edge[required_columns]
        
        # Concatenate datasets
        df_combined = pd.concat([df_normal, df_edge], ignore_index=True)
        
        # Shuffle the combined dataset
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df_combined

    def stratified_split(self, df: pd.DataFrame) -> tuple:
        """Split dataset into train/val/test with stratification"""
        print("Performing stratified split...")
        
        # Ensure no missing labels
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        
        # Check label distribution
        label_counts = df['label'].value_counts()
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        # Remove labels with only 1 sample (cannot stratify)
        valid_labels = label_counts[label_counts > 1].index
        df_valid = df[df['label'].isin(valid_labels)].reset_index(drop=True)
        
        if len(df_valid) < len(df):
            removed_count = len(df) - len(df_valid)
            print(f"Removed {removed_count} samples with singleton labels")
        
        # Stratified split: 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(
            df_valid, 
            test_size=0.2, 
            stratify=df_valid["label"], 
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df["label"], 
            random_state=42
        )
        
        return train_df, val_df, test_df

    def save_datasets(self, df_normal: pd.DataFrame, df_edge: pd.DataFrame,
                     df_eval: pd.DataFrame, train_df: pd.DataFrame,
                     val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save all datasets to CSV files"""
        print("Saving datasets...")
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Also create the specific datasets_v4 directory
        datasets_v4_dir = os.path.join(self.output_dir, "datasets", "datasets_v4")
        os.makedirs(datasets_v4_dir, exist_ok=True)
        
        # Save individual datasets
        df_normal.to_csv(self.normal_path, index=False)
        df_edge.to_csv(self.edge_path, index=False)
        df_eval.to_csv(self.eval_path, index=False)
        
        # Save splits
        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        
        print(f"Datasets saved:")
        print(f"  Normal cases: {self.normal_path} ({len(df_normal)} samples)")
        print(f"  Edge cases: {self.edge_path} ({len(df_edge)} samples)")
        print(f"  Combined eval: {self.eval_path} ({len(df_eval)} samples)")
        print(f"  Train: {self.train_path} ({len(train_df)} samples)")
        print(f"  Validation: {self.val_path} ({len(val_df)} samples)")
        print(f"  Test: {self.test_path} ({len(test_df)} samples)")

    def generate_and_split_datasets(self, n_normal_samples: int = 1000):
        """Complete pipeline: generate normal and edge cases, combine, and split"""
        print("=== V4 Dataset Generation and Splitting Pipeline ===")
        
        # Generate datasets
        df_normal = self.generate_normal_cases(n_normal_samples)
        df_edge = self.generate_edge_cases()
        
        # Combine datasets
        df_eval = self.combine_datasets(df_normal, df_edge)
        
        # Split datasets
        train_df, val_df, test_df = self.stratified_split(df_eval)
        
        # Save all datasets
        self.save_datasets(df_normal, df_edge, df_eval, train_df, val_df, test_df)
        
        print("=== Pipeline completed successfully! ===")
        return {
            'normal': df_normal,
            'edge': df_edge,
            'eval': df_eval,
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

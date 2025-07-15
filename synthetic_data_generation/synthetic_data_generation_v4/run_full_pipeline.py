"""
Complete Pipeline Runner for V4 Synthetic Domain Dataset Generation
Calls all the files to perform the synthetic data generation pipeline
"""

from dataset_splitter import DatasetSplitter

def main():
    """Run complete pipeline by calling DatasetSplitter"""
    print("=== V4 Synthetic Domain Dataset Generation Pipeline ===")
    
    # Initialize dataset splitter and run complete pipeline
    # Point to DomainAI directory (current directory when run from DomainAI folder)
    splitter = DatasetSplitter(output_dir="./")
    
    # This will call CreativeLLMGenerator and CreativeEdgeCaseGenerator internally
    # and perform the complete pipeline: generate, combine, split, and save
    datasets = splitter.generate_and_split_datasets(n_normal_samples=5000)
    
    print("\n=== Pipeline Summary ===")
    for name, df in datasets.items():
        print(f"{name.capitalize()}: {len(df)} samples")
    
    print("\n=== All CSV files generated successfully! ===")

if __name__ == "__main__":
    main()
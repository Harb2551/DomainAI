"""
Pipeline runner for generating both normal and edge case datasets using SOLID principles.
"""
from generate_normal_cases import NormalCaseGenerator
from generate_edge_cases import EdgeCaseGenerator

class DatasetGenerationPipeline:
    def __init__(self, normal_count=1000, normal_path="synthetic_domain_dataset_normal.csv", edge_path="synthetic_domain_dataset_edge_cases.csv"):
        self.normal_count = normal_count
        self.normal_path = normal_path
        self.edge_path = edge_path
        self.normal_generator = NormalCaseGenerator(n=self.normal_count)

    def run(self):
        print("Generating normal cases...")
        self.normal_generator.save(self.normal_path)
        print("Generating edge/inappropriate cases...")
        edge_gen = EdgeCaseGenerator(inappropriate_count=50)
        edge_df = edge_gen.generate()
        edge_df.to_csv(self.edge_path, index=False)
        print(f"Edge/inappropriate cases saved to {self.edge_path}")
        print("All datasets generated.")

if __name__ == "__main__":
    pipeline = DatasetGenerationPipeline(
        normal_path="./synthetic_domain_dataset_normal.csv",
        edge_path="./synthetic_domain_dataset_edge_cases.csv"
    )
    pipeline.run()

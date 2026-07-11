import pandas as pd

def inspect_excel(path, output_path):
    df = pd.read_excel(path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Columns: " + ", ".join(df.columns.astype(str).tolist()) + "\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("First 100 rows:\n")
        f.write(df.head(100).to_string())
        f.write("\n\nLast 30 rows:\n")
        f.write(df.tail(30).to_string())

if __name__ == "__main__":
    inspect_excel(
        "c:/Learn/Thesis/Thesis-advising-system/data/Diem.xlsx",
        "c:/Learn/Thesis/Thesis-advising-system/scratch/inspect_result.txt"
    )

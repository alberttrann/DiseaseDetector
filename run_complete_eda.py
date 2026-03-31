"""
Run complete EDA for both CSV and Image data
"""

from eda_npk_analysis import NPKDataAnalyzer
from eda_image_analysis import RiceImageAnalyzer
import sys

def main():
    print("COMPREHENSIVE EDA FOR RICE AGRICULTURE PROJECT")
    
    print("This script will perform:")
    print("1. Crop Recommendation Data Analysis (NPK.csv)")
    print("2. Rice Disease Image Dataset Analysis")
    print("\n" + "-" * 100 + "\n")
    
    # Part 1: CSV Analysis
    print("PART 1: ANALYZING CROP RECOMMENDATION DATA")
    print("-" * 100)
    try:
        csv_path = r"Các thông số tối ưu cho các loại cây trồng\NPK.csv"
        csv_analyzer = NPKDataAnalyzer(csv_path)
        csv_analyzer.run_complete_analysis()
    except Exception as e:
        print(f"\n Error in CSV analysis: {e}")
        print("Please check the CSV file path and try again.\n")
    
    print("\n" + "=" * 100 + "\n")
    
    # Part 2: Image Analysis
    print("PART 2: ANALYZING RICE DISEASE IMAGE DATASET")
    print("-" * 100)
    try:
        base_dir = r"C:\Users\admin\Downloads\DATASET"
        image_analyzer = RiceImageAnalyzer(base_dir)
        image_analyzer.run_complete_analysis()
    except Exception as e:
        print(f"\n Error in image analysis: {e}")
        print("Please check the image directory path and try again.\n")
    
    print("ALL ANALYSIS COMPLETE!")
    print("Check the 'eda_outputs' folder for all generated reports and visualizations")

if __name__ == "__main__":
    main()
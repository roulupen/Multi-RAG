import os
import argparse
import pandas as pd
from tqdm import tqdm
from bert_score import score


def calculate_bert_score(inp_file, output_file):
    """
    Calculate BERTScore for multiple RAG-generated answers and save results to a new Excel file.
    
    :param inp_file: Path to the input Excel file
    :param output_file: Path to save the output Excel file
    """
    # Load the Excel file
    df = pd.read_excel(inp_file)

    # Check for "Ground truth" column
    if "ground truth" not in df.columns:
        raise ValueError("The 'Ground truth' column is missing from the input file.")

    # Get response columns (excluding S.No, Query, and Ground truth)
    response_columns = [col for col in df.columns if col not in ["S.No", "query", "ground truth"]]

    # Add BERTScore columns to the original DataFrame
    print("Calculating BERTScore for RAG-generated answers:")
    for response_col in tqdm(response_columns, desc="Processing RAG Techniques", unit="technique"):
        # Get ground truth and responses
        ground_truths = df["ground truth"].fillna("").tolist()
        responses = df[response_col].fillna("").tolist()

        # Calculate BERTScore
        _, _, F1 = score(responses, ground_truths, lang="en", verbose=True)
        
        # Add the BERTScore (F1) as a new column
        df[f"{response_col}_bs"] = F1.tolist()

    # Save the updated DataFrame to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"BERTScore results saved to {output_file}")

def summarize_results(results_file, res_summary_output_path):

    results_df = pd.read_excel(results_file)

    mean_results_file = results_df.iloc[:, 21:].mean() # after 21 columns we have the bert scores

    output_df = pd.DataFrame({
    'rag_method_name': mean_results_file.index,
    'bert_score_mean': mean_results_file.values})

    # Save the output DataFrame to an Excel file
    output_df.to_excel(res_summary_output_path, index=False)
    print(f"Results summary saved to {res_summary_output_path}")


# Run the main function when the script is executed
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a directory path from the command line.")
    parser.add_argument("retriever_results_file", type=str, help="Path to the directory to be processed.")
 
    # Parse the command-line argument
    args = parser.parse_args()

    # Calculate cosine similarity scores
    output_file = f"output/{os.path.basename(args.retriever_results_file).split('.')[0]}_bert_score.xlsx"
    calculate_bert_score(args.retriever_results_file, output_file)

    # Summarize cosine similarity results across all the methods
    res_summary_output_path = f"output/{os.path.basename(output_file).split('.')[0]}_summary.xlsx"
    summarize_results(output_file, res_summary_output_path)
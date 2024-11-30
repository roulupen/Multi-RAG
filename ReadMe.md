### Project: Better Retrieval for Generation 

### Team:
- **Amrit Mohanty (CS23MTECH15005)**  
- **Upendra Kumar Roul (CS23MTECH15025)**  

---

## Setting Up the Environment  

### Step 1: Create a New Conda Environment  
Run the following commands to create and activate the environment:  
```bash
conda create -n rag-proj-v2 python=3.10.15  # Create a new conda environment  
conda activate rag-proj-v2  # Activate the environment
```

### Step 2: Install Dependencies  
Navigate to the project directory and install the required dependencies:  
```bash
cd './Multi-RAG Automated/'  
pip install -r ./requirements.txt  
```

---

## Running the Project  

### Running on HotpotQA Dataset

To process the **HotpotQA dataset**, run:  
```bash
python main.py ./multi_rag_query_hotpotqa.xlsx ./docs/hotpotqa
```  
The output will be generated at:  
```
output/retriever_results_combinations_multi_rag_query_hotpotqa.xlsx
```  

#### Generating Evaluation Metric - Cosine Similarity for HotpotQA:  
Run the following command to generate evaluation scores:  
```bash
python ./evaluation_cosine_similarity.py ./output/retriever_results_combinations_multi_rag_query_hotpotqa.xlsx
```  
This will produce two output files:  
- `retriever_results_combinations_multi_rag_query_hotpotqa_cosine_similarity.xlsx`  
- `retriever_results_combinations_multi_rag_query_hotpotqa_cosine_similarity_summary.xlsx`

#### Generating Evaluation Metric - Bert Score for HotpotQA:  
Run the following command to generate evaluation scores:  
```bash
python ./evaluation_bert_score.py ./output/retriever_results_combinations_multi_rag_query_hotpotqa.xlsx
```  
This will produce two output files:  
- `retriever_results_combinations_multi_rag_query_hotpotqa_bert_score.xlsx`  
- `retriever_results_combinations_multi_rag_query_hotpotqa_bert_score_summary.xlsx`

---

### Running on Generic PDF Files  

To process **generic PDF files**, run:  
```bash
python main.py ./multi_rag_query_pdf.xlsx ./docs/pdf
```  
The output will be generated at:  
```
output/retriever_results_combinations_multi_rag_query_pdf.xlsx
```  

#### Generating Evaluation Metric - Cosine Similarity for Generic PDF Files:  
Run the following command to generate evaluation scores:  
```bash
python ./evaluation_cosine_similarity.py ./output/retriever_results_combinations_multi_rag_query_pdf.xlsx
```  
This will produce two output files:  
- `retriever_results_combinations_multi_rag_query_pdf_cosine_similarity.xlsx`  
- `retriever_results_combinations_multi_rag_query_pdf_cosine_similarity_summary.xlsx`

#### Generating Evaluation Metric - Bert Score for Generic PDF Files:  
Run the following command to generate evaluation scores:  
```bash
python ./evaluation_bert_score.py ./output/retriever_results_combinations_multi_rag_query_pdf.xlsx
```  
This will produce two output files:  
- `retriever_results_combinations_multi_rag_query_pdf_bert_score.xlsx`  
- `retriever_results_combinations_multi_rag_query_pdf_bert_score_summary.xlsx`

---

## Folder Structure  

```plaintext
.
├── __pycache__/                # Python cache files
├── chroma_db/                  # Chroma database directory
├── docs/                       # Contains input documents (HotpotQA and PDFs)
├── output/                     # Folder for generated output files
├── config_loader.py            # Script to load configuration
├── config.yaml                 # Configuration file
├── evaluation_bert_score.py               # Script to generate evaluation metrics - bert score
├── evaluation_cosine_similarity.py        # Script to generate evaluation metrics - cosine similarity
├── main.py                     # Main script to process queries
├── multi_rag_query_hotpotqa.xlsx  # Input queries for HotpotQA
├── multi_rag_query_pdf.xlsx       # Input queries for PDFs
├── rag_hybrid_dlm1.py          # RAG hybrid model - DLM1 implementation
├── rag_hybrid_dlm2.py          # RAG hybrid model - DLM2 implementation
├── rag_hyde_dlm1.py            # RAG HYDE model - DLM1 implementation
├── rag_hyde_dlm2.py            # RAG HYDE model - DLM2 implementation
├── rag_hyde+hybrid_dlm1.py     # RAG HYDE + Hybrid DLM1 implementation
├── rag_hyde+hybrid_dlm2.py     # RAG HYDE + Hybrid DLM2 implementation
├── rag_recursive_splitter.py   # Recursive splitting utility
├── rag_semantic_splitter.py    # Semantic splitting utility
├── rag_sentence_splitter.py    # Sentence splitting utility
├── ReadMe.md                   # Project README file
├── requirements.txt            # Dependencies for the project
```

---

## Notes  

- Ensure that all dependencies are installed before running the scripts.  
- Replace file paths as necessary for your system.  
- Output files will be saved in the `output/` directory.  

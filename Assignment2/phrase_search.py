import json
import os
import sys
import spacy
import time

def phrase_search_query(query: str, index: object) -> list:
    """ 
    Args:
        query (str): The phrase to search for.
        index (object): The loaded inverted index object.

    Returns:
        list: A list of sorted document IDs that contain the exact phrase.
    """
    try:
        nlp = spacy.blank("en")
    except OSError:
        print("spaCy model not found.")
        # Exit gracefully if spacy model is not found.
        sys.exit(1)

    # Tokenize the cleaned query using spaCy.
    doc = nlp(query)
    query_terms = [token.text.strip() for token in doc] 

    if not query_terms:
        return []

    # Return empty list if any term is not in the index.
    for term in query_terms:
        if term not in index:
            return []

    # Find the intersection of documents containing all query terms.
    try:
        common_docs = set(index[query_terms[0]]["postings"].keys())
    except KeyError:
        return []

    for term in query_terms[1:]:
        term_docs = set(index[term]["postings"].keys())
        common_docs.intersection_update(term_docs)  # now common doc has only those documents which contain all the terms in query

    if not common_docs:
        return []

    matching_docs = []

    # 3. Verify Positional Adjacency
    for doc_id in common_docs:
        # Start with the positions of the first term.
        current_valid_positions = index[query_terms[0]]["postings"][doc_id]["pos"]

        # Iteratively check each subsequent term.
        for i in range(1, len(query_terms)):
            term = query_terms[i]
            next_term_positions = set(index[term]["postings"][doc_id]["pos"])
            new_valid_positions = []
            
            # Find positions where the next term appears immediately after.
            for pos in current_valid_positions:
                if (pos + 1) in next_term_positions:
                    new_valid_positions.append(pos + 1)
            
            # If no valid sequence is found, stop checking this document.
            if not new_valid_positions:
                current_valid_positions = []
                break 
            
            current_valid_positions = new_valid_positions
        if current_valid_positions:
            matching_docs.append(doc_id)
    return sorted(matching_docs)


def phrase_search( index_dir: str, query_file_path: str, output_file: str) -> None:
    """
    Loads an index, runs queries from a file, and saves the results.
    [cite_start]This function's signature matches the one in the assignment[cite: 114].
    """
    print("--- Starting Phrase Search Task ---")

    # Load the inverted index from the specified directory.
    index_file_path = os.path.join(index_dir, "index.json")
    try:
        with open(index_file_path, 'r') as f:
            inverted_index = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Failed to load index. {e}")
        return

    # Load the queries from the absolute file path.
    # try:
    #     with open(query_file_path, 'r', encoding='utf-16') as f:
    #         queries = [json.loads(line) for line in f if line.strip()]
    # except FileNotFoundError as e:
    #     print(f" ERROR: Failed to load queries. {e}")
    #     return
    
    queries = []
    try:
        try:
            with open(query_file_path, "r", encoding="utf-8") as f:
                queries = [json.loads(line) for line in f if line.strip()]
        except UnicodeError:
            # Fallback to UTF-16 if UTF-8 fails
            with open(query_file_path, "r", encoding="utf-16") as f:
                queries = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load queries. {e}")
        return
    

    # Open the output file for writing results.
    with open(output_file, 'w', encoding='utf-8') as f_out:
        print(f"Processing queries and writing results to: {output_file}")
        for  query in queries:
            query_text = query.get("title")
            query_id = query.get("query_id")
            matching_docs = phrase_search_query(query_text, inverted_index)
            
            if not matching_docs:
                # print(f" Query '{query_id}': No documents found.")
                continue
            
            for rank, doc_id in enumerate(matching_docs, 1):
                f_out.write(f"{query_id}\t{doc_id}\t{rank}\t1\n")
    print("Search complete. Output file generated.")


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print("Usage: python phrase_search.py <INDEX_DIR> <QUERY_FILE_PATH> <OUTPUT_DIR>")
        sys.exit(1)

    # Assign arguments to variables from the shell command.
    index_dir = sys.argv[1]
    query_file = sys.argv[2]
    output_dir = sys.argv[3]

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full path for the output file.
    output_file_path = os.path.join(output_dir, "phrase_search_docids.txt")
    start_time = time.time()
    # Call the main function to execute the entire process.
    phrase_search(index_dir, query_file, output_file_path)
    
    end_time = time.time()
    print("Execution time: {:.2f} seconds".format(end_time - start_time))
    
    # python phrase_search.py INDEX_DIR data\queries.json OUTPUT_DIR
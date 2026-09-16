import sys
sys.path.append("src")

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from extraction import extract_text, list_documents
from summarizer import summarize_long_text
from qa import answer_question, OllamaConnectionError


def process_document(file_path: str) -> str:
    """Extract text and generate a summary for a single document.
    Runs inside a Spark worker, so all imports must happen here.
    Never raises: returns an error message string instead, so one bad
    document doesn't kill the whole batch job."""
    sys.path.append("src")
    from extraction import extract_text
    from summarizer import summarize_long_text

    try:
        text = extract_text(file_path)
        if not text.strip():
            return "ERROR: empty document"
        return summarize_long_text(text)
    except Exception as e:
        return f"ERROR: {str(e)}"


def batch_summarize(data_folder: str = "data"):
    """Use Spark to summarize all documents in parallel."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = SparkSession.builder \
        .appName("DocumentBot") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()

    docs = list_documents(data_folder)
    if not docs:
        print("No documents found in data/")
        spark.stop()
        return None

    num_partitions = min(len(docs), 2)
    df = spark.createDataFrame([(d,) for d in docs], ["file_path"]) \
        .repartition(num_partitions)

    summarize_udf = udf(process_document, StringType())
    result_df = df.withColumn("summary", summarize_udf(df.file_path))

    results = result_df.collect()
    spark.stop()
    return results


def interactive_qa(selected_path: str):
    """Interactive Q&A loop for a single document (runs outside Spark)."""
    try:
        text = extract_text(selected_path)
    except ValueError as e:
        print(f"Error loading document: {e}")
        return

    print("\nAsk questions about this document (type 'exit' to quit).\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("exit", "salir", "quit"):
            break
        if not question:
            continue
        try:
            answer = answer_question(text, question)
            print(f"A: {answer}\n")
        except OllamaConnectionError as e:
            print(f"{e}\n")
            break


def main():
    print("Processing all documents with Spark...\n")
    results = batch_summarize("data")

    if not results:
        return

    valid_results = [r for r in results if not r.summary.startswith("ERROR:")]
    failed_results = [r for r in results if r.summary.startswith("ERROR:")]

    if failed_results:
        print(f"{len(failed_results)} document(s) failed to process:")
        for r in failed_results:
            print(f"    {r.file_path}: {r.summary}")
        print()

    if not valid_results:
        print("No documents could be processed successfully.")
        return

    print("Available documents:")
    for i, row in enumerate(valid_results):
        print(f"  [{i + 1}] {row.file_path}")
        print(f"      Summary: {row.summary[:200]}...\n")

    choice_input = input("Select a document to ask questions about (number): ").strip()
    if not choice_input.isdigit():
        print("Invalid selection: please enter a number.")
        return

    choice = int(choice_input) - 1
    if choice < 0 or choice >= len(valid_results):
        print(f"Invalid selection: choose a number between 1 and {len(valid_results)}.")
        return

    interactive_qa(valid_results[choice].file_path)


if __name__ == "__main__":
    main()
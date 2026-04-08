import os
import re

def count_sentences(text):
    """Count the number of sentences in text."""
    # Split on sentence-ending punctuation followed by space or end of string
    sentences = re.split(r'[.!?]+\s+|[.!?]+$', text.strip())
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def analyze_file(file_path):
    """Analyze a single file and return word count, character count, and sentence count."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()
        word_count = len(words)
        char_count = len(content)
        sentence_count = count_sentences(content)
        return word_count, char_count, sentence_count

def calculate_stats(total_words, total_chars, total_sentences, file_count):
    """Helper function to calculate statistics."""
    if file_count == 0:
        return {
            'file_count': 0,
            'avg_word_count': 0,
            'total_chars': 0,
            'total_sentences': 0,
            'avg_sentence_length': 0
        }

    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

    return {
        'file_count': file_count,
        'avg_word_count': total_words / file_count,
        'total_chars': total_chars,
        'total_sentences': total_sentences,
        'avg_sentence_length': avg_sentence_length
    }

def analyze_directory(directory_path):
    """Calculate statistics for all .txt files in a directory, overall and by domain."""
    total_words = 0
    total_chars = 0
    total_sentences = 0
    file_count = 0

    # Dictionary to store stats by domain
    domain_stats = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            word_count, char_count, sentence_count = analyze_file(file_path)

            # Overall totals
            total_words += word_count
            total_chars += char_count
            total_sentences += sentence_count
            file_count += 1

            # Extract domain (prefix before first underscore)
            domain = filename.split('_')[0]

            # Initialize domain if not seen before
            if domain not in domain_stats:
                domain_stats[domain] = {
                    'total_words': 0,
                    'total_chars': 0,
                    'total_sentences': 0,
                    'file_count': 0
                }

            # Update domain stats
            domain_stats[domain]['total_words'] += word_count
            domain_stats[domain]['total_chars'] += char_count
            domain_stats[domain]['total_sentences'] += sentence_count
            domain_stats[domain]['file_count'] += 1

    # Calculate overall stats
    overall_stats = calculate_stats(total_words, total_chars, total_sentences, file_count)

    # Calculate stats for each domain
    domain_results = {}
    for domain, stats in domain_stats.items():
        domain_results[domain] = calculate_stats(
            stats['total_words'],
            stats['total_chars'],
            stats['total_sentences'],
            stats['file_count']
        )

    return overall_stats, domain_results

def print_stats(stats, prefix=""):
    """Helper function to print statistics."""
    print(f"{prefix}Number of .txt files: {stats['file_count']}")
    print(f"{prefix}Total characters: {stats['total_chars']:,}")
    print(f"{prefix}Total sentences: {stats['total_sentences']}")
    print(f"{prefix}Average word count per file: {stats['avg_word_count']:.2f}")
    print(f"{prefix}Average sentence length (words): {stats['avg_sentence_length']:.2f}")

if __name__ == "__main__":
    directories = ["/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/human_edited_captions/all", "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/test/gt_captions_filteredToMatchHumanSet", "/home/ubuntu/thesis/data/samples/new_samples_no_overlap/train/gt_captions"]

    for directory in directories:
        if os.path.isdir(directory):
            overall_stats, domain_stats = analyze_directory(directory)
            dir_name = "/".join(directory.split("/")[-2:])

            print(f"\n{'='*60}")
            print(f"=== {dir_name} ===")
            print(f"{'='*60}")

            print("\n--- Overall Statistics ---")
            print_stats(overall_stats)

            if domain_stats:
                print("\n--- Statistics by Domain ---")
                for domain in sorted(domain_stats.keys()):
                    print(f"\n  Domain: {domain}")
                    print_stats(domain_stats[domain], prefix="    ")
        else:
            print(f"The provided path is not a valid directory: {directory}")
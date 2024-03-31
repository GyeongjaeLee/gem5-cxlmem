from collections import Counter, defaultdict

def analyze_page_frequencies_and_sequences(file_path):
    page_counter = Counter()
    threshold = 20
    sequence_threshold = 4
    current_page = None
    sequence_count = 1
    sequential_pages = defaultdict(list)  # Change to list to track all sequences

    # New: Track pages that meet both criteria
    pages_meeting_both_criteria = Counter()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            if len(parts) > 1:
                page_number = parts[-1].strip()
                page_counter[page_number] += 1

                if current_page == page_number:
                    sequence_count += 1
                else:
                    if sequence_count >= sequence_threshold:
                        sequential_pages[current_page].append(sequence_count)
                        # Check if the page also meets the overall threshold criterion
                        if page_counter[current_page] >= threshold:
                            pages_meeting_both_criteria[current_page] += 1
                    sequence_count = 1
                current_page = page_number

        # Check at the end of the file
        if sequence_count >= sequence_threshold:
            sequential_pages[current_page].append(sequence_count)
            if page_counter[current_page] >= threshold:
                pages_meeting_both_criteria[current_page] += 1

    frequent_pages = {page: count for page, count in page_counter.items() if count >= threshold}
    print(f"Total unique page numbers: {len(page_counter)}")
    print(f"Page numbers appearing {threshold} times or more: {len(frequent_pages)}")
    print(f"Page numbers appearing sequentially {sequence_threshold} times or more : {len(sequential_pages)}")
    print(f"Page numbers appearing {threshold} times or more && Page numbers appearing sequentially {sequence_threshold} times or more : {len(pages_meeting_both_criteria)}")
    print("Frequent Page Numbers:")
    for page, count in frequent_pages.items():
        print(f"Page Number: {page}, Appearances: {count}")



    # Print pages meeting both criteria
    print(f"\nPages meeting both {threshold} appearances and having multiple sequences of {sequence_threshold} sequential appearances:")
    for page,counts in sequential_pages.items():
           if page in pages_meeting_both_criteria :
            print(f"Page Number: {page}, Total Appearances: {page_counter[page]}, Sequential Counts: {counts}")


file_path = 'output.txt'
analyze_page_frequencies_and_sequences(file_path)
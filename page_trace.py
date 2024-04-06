import sys
from collections import (
    Counter,
    defaultdict,
)


def analyze_page_frequencies_and_sequences(file_path):
    page_counter = Counter()
    threshold = 20
    sequence_threshold = 4
    current_page = None
    sequence_count = 1
    sequential_pages = defaultdict(list)

    pages_meeting_both_criteria = Counter()

    with open(file_path) as file:
        for line in file:
            if "Timing access to addr" in line:
                parts = line.strip().split(",")
                if len(parts) > 1:
                    hex_address = parts[0].split()[-1]
                    # Convert to page number by discarding lower 12 bits and converting to decimal
                    page_number = (
                        int(hex_address[2:-3], 16) if hex_address[2:-3] else 0
                    )
                    page_counter[page_number] += 1

                    if current_page == page_number:
                        sequence_count += 1
                    else:
                        if sequence_count >= sequence_threshold:
                            sequential_pages[current_page].append(
                                sequence_count
                            )
                            if page_counter[current_page] >= threshold:
                                pages_meeting_both_criteria[current_page] += 1
                        sequence_count = 1
                    current_page = page_number

        if sequence_count >= sequence_threshold:
            sequential_pages[current_page].append(sequence_count)
            if page_counter[current_page] >= threshold:
                pages_meeting_both_criteria[current_page] += 1

    frequent_pages = {
        page: count
        for page, count in page_counter.items()
        if count >= threshold
    }
    print(f"Total unique page numbers: {len(page_counter)}")
    print(
        f"Page numbers appearing {threshold} times or more: {len(frequent_pages)}"
    )
    print(
        f"Page numbers appearing sequentially {sequence_threshold} times or more: {len(sequential_pages)}"
    )
    print(
        f"Page numbers appearing {threshold} times or more && Page numbers appearing sequentially {sequence_threshold} times or more: {len(pages_meeting_both_criteria)}"
    )
    print("Frequent Page Numbers:")
    for page, count in frequent_pages.items():
        print(f"Page Number: {page}, Appearances: {count}")

    # Print pages meeting both criteria
    print(
        f"\nPages meeting both {threshold} appearances and having multiple sequences of {sequence_threshold} sequential appearances:"
    )
    for page, counts in sequential_pages.items():
        if page in pages_meeting_both_criteria:
            print(
                f"Page Number: {page}, Total Appearances: {page_counter[page]}, Sequential Counts: {counts}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    analyze_page_frequencies_and_sequences(file_path)

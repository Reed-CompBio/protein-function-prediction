from colorama import Fore, Back, Style


def print_progress(current, total, bar_length=65):
    # Calculate the progress as a percentage
    percent = float(current) / total
    # Determine the number of hash marks in the progress bar
    arrow = "-" * int(round(percent * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))

    # Choose color based on completion
    if current < total:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    # Construct the progress bar string
    progress_bar = f"[{arrow + spaces}] {int(round(percent * 100))}%"

    # Print the progress bar with color, overwriting the previous line
    print(f"\r{color}{progress_bar}{Style.RESET_ALL}", end="")

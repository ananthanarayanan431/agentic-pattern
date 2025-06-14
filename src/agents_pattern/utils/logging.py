
import time 

from colorama import Fore
from colorama import Style

def fancy_print(message: str) -> None:
    """Prints a message with a timestamp and color

    Args:
        message (str): The message to print
    """
    # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # print(f"{timestamp} {Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}")

    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)

def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker for each iteration of the generation-reflection loop.

    Args:
        step (int): The current step in the loop.
        total_steps (int): The total number of steps in the loop.
    """
    fancy_print(f"STEP {step + 1}/{total_steps}")
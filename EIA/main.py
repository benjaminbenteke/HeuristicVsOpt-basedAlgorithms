import argparse

# Import your scripts as modules
import Ex1_EIA
import Ex2_EIA
import Ex3_EIA
import Ex4_EIA
import Ex5_EIA
import Ex6_EIA
import Ex7_EIA
import Ex8_EIA
import Ex9_EIA
import Ex10_EIA


def main():
    parser = argparse.ArgumentParser(description="Run EIA exercises")
    parser.add_argument(
        "--ex",
        nargs="+",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "all"],
        required=True,
        help="Which exercises to run (e.g. --ex 1 2 or --ex all)"
    )

    args = parser.parse_args()

    selected = args.ex

    if "all" in selected:
        selected = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    if "1" in selected:
        print("Running Exercise 1...")
        Ex1_EIA.main()

    if "2" in selected:
        print("Running Exercise 2...")
        Ex2_EIA.main()

    if "3" in selected:
        print("Running Exercise 3...")
        Ex3_EIA.main()
    if "4" in selected:
        print("Running Exercise 4...")
        Ex4_EIA.main()
    
    if "5" in selected:
        print("Running Exercise 5...")
        Ex5_EIA.main()

    if "6" in selected:
        print("Running Exercise 6...")
        Ex6_EIA.main()

    if "7" in selected:
        print("Running Exercise 7...")
        Ex7_EIA.main()

    if "8" in selected:
        print("Running Exercise 8...")
        Ex8_EIA.main()

    if "9" in selected:
        print("Running Exercise 9...")
        Ex9_EIA.main()

    if "10" in selected:
        print("Running Exercise 10...")
        Ex10_EIA.main()

if __name__ == "__main__":
    main()
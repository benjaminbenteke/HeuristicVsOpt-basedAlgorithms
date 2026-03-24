import argparse

# Import your scripts as modules
import Ex1_SGD
import Ex2_SGD
import Ex3_SGD
import Ex4_SGD
import Ex5_SGD
import Ex6_SGD
import Ex7_SGD
import Ex8_SGD
import Ex9_SGD


def main():
    parser = argparse.ArgumentParser(description="Run SGD exercises")
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
        Ex1_SGD.main()

    if "2" in selected:
        print("Running Exercise 2...")
        Ex2_SGD.main()

    if "3" in selected:
        print("Running Exercise 3...")
        Ex3_SGD.main()
    if "4" in selected:
        print("Running Exercise 4...")
        Ex4_SGD.main()
    
    if "5" in selected:
        print("Running Exercise 5...")
        Ex5_SGD.main()

    if "6" in selected:
        print("Running Exercise 6...")
        Ex6_SGD.main()

    if "7" in selected:
        print("Running Exercise 7...")
        Ex7_SGD.main()

    if "8" in selected:
        print("Running Exercise 8...")
        Ex8_SGD.main()

    if "9" in selected:
        print("Running Exercise 9...")
        Ex9_SGD.main()

if __name__ == "__main__":
    main()

import argparse

# Import your scripts as modules
import Ex1_SGM
import Ex2_SGM
import Ex3_SGM
import Ex4_SGM
import Ex5_SGM
import Ex6_SGM
import Ex7_SGM
import Ex8_SGM
import Ex9_SGM


def main():
    parser = argparse.ArgumentParser(description="Run SGM exercises")
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
        Ex1_SGM.main()

    if "2" in selected:
        print("Running Exercise 2...")
        Ex2_SGM.main()

    if "3" in selected:
        print("Running Exercise 3...")
        Ex3_SGM.main()
    if "4" in selected:
        print("Running Exercise 4...")
        Ex4_SGM.main()
    
    if "5" in selected:
        print("Running Exercise 5...")
        Ex5_SGM.main()

    if "6" in selected:
        print("Running Exercise 6...")
        Ex6_SGM.main()

    if "7" in selected:
        print("Running Exercise 7...")
        Ex7_SGM.main()

    if "8" in selected:
        print("Running Exercise 8...")
        Ex8_SGM.main()

    if "9" in selected:
        print("Running Exercise 9...")
        Ex9_SGM.main()

if __name__ == "__main__":
    main()
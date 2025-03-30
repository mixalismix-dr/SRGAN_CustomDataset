def run_iteration_1():
    print("Running Iteration 1: Training with synthetic LR images")
    # your logic here

def run_iteration_2():
    print("Running Iteration 2: Inference with real LR images")
    # your logic here

def main():
    print("Which iteration do you want to perform?")
    print("1: Iteration 1 (Train on synthetic LR)")
    print("2: Iteration 2 (Apply to real LR)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_iteration_1()
    elif choice == "2":
        run_iteration_2()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()

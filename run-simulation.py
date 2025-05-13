from orchestrator import run_orchestration

def main():
    print("🧠 Welcome to the ROS-CVD Multi-Agent Explorer!")
    print("Enter your biomedical query below.\n")

    user_query = input("🔍 Your query: ").strip()

    if not user_query:
        print("⚠️ No query provided. Exiting.")
        return

    print("\n🔄 Processing...\n")
    final_report = run_orchestration(user_query)

    print("\n📄 Final Report:\n")
    print(final_report)

if __name__ == "__main__":
    main()

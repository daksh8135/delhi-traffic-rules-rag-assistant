# main.py

from generator import Generator

def main():
    print("\n🤖 Traffic Rules Assistant (DELHI RAG Chatbot)")
    print("Ask any question about traffic rules, fines, procedures, etc.")
    print("Type 'exit' to quit.\n")

    generator = Generator()

    while True:
        query = input("🧑 You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting. Stay safe on the roads!\n")
            break

        if not query:
            print("⚠️ Please enter a valid question.\n")
            continue

        print("\n🤖 Assistant is thinking...\n")
        try:
            result = generator.ask(query)
            print("🤖 Answer:\n")
            print(result["answer"])
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()

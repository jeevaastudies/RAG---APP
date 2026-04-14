import sys

print("1. Python Executable being used:")
print(sys.executable)
print("\n2. Where Python is looking for packages:")
for path in sys.path:
    print("-", path)

print("\n3. Testing LangChain:")
try:
    import langchain
    print("SUCCESS: LangChain found at:", langchain.__file__)
    from langchain.chains import RetrievalQA
    print("SUCCESS: langchain.chains imported perfectly.")
except Exception as e:
    print("FAILED:", e)
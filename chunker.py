
def read_data() -> str:
    with open("data.md", "r", encoding="utf-8") as f:
        return f.read()
    
def get_chunks() -> list[str]:
    """
    design your text chunk algorithm here
    langchain -> recursive charactoer text splitter
    https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html
    """
    content = read_data()
    chunks = content.split('\n\n')
    
    result = []
    header = ""
    for c in chunks:
        if c.startswith("#"):
            header += f"{c}\n"
        else:
            result.append(f"{header}{c}")
            header = ""

    return result

if __name__ == '__main__':
    chunks = get_chunks()
    for c in chunks:
        print(c)
        print("--------------")

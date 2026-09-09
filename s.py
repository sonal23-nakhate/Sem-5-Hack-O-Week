from groq import Groq

client=Groq(api_key="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")

sys_msg="""You are a travel agent AI.
      Your job is to:  
      -answer travel questions.
      -suggest itineraries.
      -recommend hotels and transport.
      """

while True:
    query = input("User: ")
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": query}
        ]
    )
    
    print("Agent: ", response.choices[0].message.content)
    print("\n")

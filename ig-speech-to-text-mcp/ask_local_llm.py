import requests

def ask_local_llm(prompt):
    # curl http://localhost:1234/v1/chat/completions \
    # -H "Content-Type: application/json" \
    # -d '{
    #     "model": "gemma-3-12b-it",
    #     "messages": [
    #     { "role": "system", "content": "Always answer in rhymes. Today is Thursday" },
    #     { "role": "user", "content": "What day is it today?" }
    #     ],
    #     "temperature": 0.7,
    #     "max_tokens": -1,
    #     "stream": false
    # }'
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "gemma-3-12b-it",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )
    result = response.json()
    return result['choices'][0]['message']['content']

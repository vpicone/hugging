import cursor
from transformers import BloomTokenizerFast, BloomForCausalLM

print("")
cursor.hide()
print("Loading BLOOM model...", end="", flush=True)

tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b7")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b7")

print("Done! Enter your prompt below. Type 'exit' to quit.\n\n")

# result_length = int(input("Max words generated:"))
result_length = 20
counter = 0

# It was the patient, cut-flower sound of...

# Greedy (1b7)...the wind, the sound of the wind in the trees, the sound of the wind in the leaves (1b7)
# Sample (1b7)...a great man’s mind.


while True:
    cursor.show()
    counter += 1
    prompt = input("Prompt #" + str(counter) + ": ")
    if prompt == "exit":
        exit()

    cursor.hide()

    print("Generating...", end="", flush=True)

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    result = tokenizer.decode(model.generate(
        inputs["input_ids"], max_new_tokens=result_length, do_sample=True)[0])
    print("\r" + "Result: " + result)
    print("\n\n–––\n\n")

    cursor.show()

# The Waystone was his, just as the third silence was his. This was appropriate, as it was the greatest silence of the three, wrapping the others inside itself. It was deep and wide as autumn’s ending. It was heavy as a great river-smooth stone. It was the patient, cut-flower sound of the wind, the sound of the wind in the trees, the sound of the wind in the leaves.

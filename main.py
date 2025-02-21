import base64
import config

from utils import get_api_key
from chatbot import NDII

def main():

    # Load API Key
    api_key = get_api_key()

    # Initialize ND II
    nd_ii = NDII(api_key)

    print("ND II: Hello! I'm your autonomous vehicle assistant. How can I help you today?")
    print(f"User Input: {config.USER_INPUT}")

    # Send message with additional parameters
    response = nd_ii.send_message(
        user_input=config.USER_INPUT,
        use_cot=config.USE_COT,
        model=config.MODEL,
        modalities=config.MODALITIES,
        audio=config.AUDIO
    )

    if response:
        print(f"ND II: {response}")
    else:
        print(f"ND II: I apologize, but I encountered an error. Please try again.")

    wav_bytes = base64.b64decode(response.audio.data)
    file_path = config.FILE_NAME + config.FILE_EXT
    with open(file_path, "wb") as f:
        f.write(wav_bytes)

if __name__ == "__main__":
    main()
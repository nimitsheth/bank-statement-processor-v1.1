# #from google import genai
# import google.generativeai as genai
from config import GEMINI_PROMPT, GEMINI_API_KEY

# client = genai.Client()

# myfile = client.files.upload(file="c:/Users/mansi/Downloads/Bank Statements/Bank Statements/BOB CC JULY 25 - PDF.pdf")

# response = client.models.generate_content(
#     model="gemini-2.5-flash", contents=[GEMINI_PROMPT, myfile]
# )

# print(response.text)

from google import genai
import pandas as pd
from io import StringIO
from google.genai import types
import pathlib

client = genai.Client(api_key=GEMINI_API_KEY)
# media = pathlib.Path(__file__).parents[1] / "third_party"

media = pathlib.Path("c:/Users/mansi/Downloads/Bank Statements/Bank Statements/BOB CC JULY 25 - PDF.pdf")
sample_pdf = client.files.upload(file=media, config=dict(
    mime_type='application/pdf'))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[GEMINI_PROMPT, sample_pdf]
)
print(f"{response.text=}")
csv_text = response.text
df = pd.read_csv(
        StringIO(csv_text),
        on_bad_lines=lambda x: print(f"Bad line: {x}"),  # Print bad lines and skip them
        engine='python'  # Use python engine for better error handling
    )
print(df.shape)
print(df)
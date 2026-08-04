"""
The agent loop: sends the user's question to Gemini along with tool definitions,
executes whatever tools Gemini decides to call, feeds the results back, and returns
the final answer along with any generated map image.

Requires GEMINI_API_KEY environment variable (or loaded from .env).
"""
import os
import json
from google import genai
from google.genai import types
from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS, SEMANTIC_LAYER


def load_dotenv():
    # Try common locations for .env relative to current working directory
    for path in [".env", "src/.env", "../.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip("'\"")


load_dotenv()

MODEL = "gemini-3.5-flash"

SYSTEM_PROMPT = f"""You are an advanced geospatial AI assistant for the Barnes Ice Cap research dataset (Baffin Island, Nunavut, Canada).
You have access to real radar measurements collected during the 2015 NASA Operation IceBridge survey (using the MCoRDS radar instrument).

Dataset region information:
{json.dumps(SEMANTIC_LAYER['region'], indent=2)}

You can answer questions about ice thickness, surface elevation, and subglacial bedrock elevation along the flight lines.
Always ground your answers in actual tool results. If you need to check what variables are available, call list_datasets.
If the user's question mentions a location but not coordinates, look for coordinates in your knowledge base or suggest looking at typical ranges.
If a user's question implies a bounding box but doesn't give coordinates, use coordinates from the region's bounding box and state them explicitly.

You also have access to Sentinel-2 satellite imagery of the region for these specific dates:
- 2022-08-07 (L2A, WWC tile covering the southern portion of the ice cap)
- 2022-08-09-WWC (L1C, WWC tile covering the southern portion)
- 2022-08-09-WWD (L2A, WWD tile covering the northern portion of the ice cap)
- 2022-08-29 (L1C, WXC tile covering the eastern portion of the ice cap)

You can generate True Color RGB maps using `generate_satellite_rgb_map` and Normalized Difference Snow Index (NDSI) maps using `generate_satellite_ndsi_map` to analyze snow/ice coverage and surface changes over time.

Note:
- Elevation is reference to WGS-84 ellipsoid.
- Ice thickness is in meters.
- Bedrock elevation = Actual ice bottom (ELEVATION - BOTTOM range).
- Surface elevation = Actual surface (ELEVATION - SURFACE range).
- The flight segment is from May 7, 2015.
"""


def run_agent(user_question: str, verbose: bool = True) -> dict:
    client = genai.Client()  # reads GEMINI_API_KEY from environment
    
    # Map the tool definitions to Gemini FunctionDeclarations
    gemini_tools = [
        types.FunctionDeclaration(
            name=defn["name"],
            description=defn["description"],
            parameters=defn.get("input_schema"),
        )
        for defn in TOOL_DEFINITIONS
    ]
    
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)]
        )
    ]
    
    generated_image_b64 = None

    for iteration in range(6):  # tool-use loop, capped to avoid runaway calls
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[types.Tool(function_declarations=gemini_tools)],
                # Disable automatic calling to handle mapping manually and strip images from context
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        # If the model didn't call any more tools, it has a final answer
        if not response.function_calls:
            answer = response.text or ""
            return {"answer": answer, "image_base64": generated_image_b64}

        # Model requested function calls. We must add the model's response to the message history.
        if response.candidates and response.candidates[0].content:
            messages.append(response.candidates[0].content)
        else:
            model_parts = []
            if response.text:
                model_parts.append(types.Part.from_text(text=response.text))
            for call in response.function_calls:
                model_parts.append(types.Part.from_function_call(
                    name=call.name,
                    args=call.args
                ))
            messages.append(types.Content(role="model", parts=model_parts))

        tool_parts = []
        for call in response.function_calls:
            if call.name not in TOOL_FUNCTIONS:
                result = {
                    "error": f"Tool '{call.name}' is not recognized. Available tools: {list(TOOL_FUNCTIONS.keys())}."
                }
                if verbose:
                    print(f"  [unrecognized tool call] {call.name}({call.args})")
            else:
                fn = TOOL_FUNCTIONS[call.name]
                if verbose:
                    print(f"  [tool call] {call.name}({call.args})")
                try:
                    result = fn(**call.args)
                except Exception as e:
                    result = {"error": f"Error executing tool '{call.name}': {str(e)}"}
            
            # Extract generated map images to return separately and prevent token bloat
            if isinstance(result, dict) and "image_base64" in result:
                generated_image_b64 = result["image_base64"]
                # Replace the huge base64 string with a placeholder in the LLM context
                result = {**result, "image_base64": "<omitted from context, returned separately in the API response>"}
            
            tool_parts.append(types.Part.from_function_response(
                name=call.name,
                response=result
            ))

        messages.append(types.Content(role="tool", parts=tool_parts))

    return {
        "answer": "Reached maximum tool-call iterations without getting a final response from the agent.",
        "image_base64": None
    }


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or (
        "What is the average ice thickness and surface elevation of the Barnes Ice Cap "
        "in the bounding box [-73.5, 69.8, -73.0, 70.3]?"
    )
    print(f"User: {question}\n")
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable is not set. API calls will fail.")
        
    try:
        result = run_agent(question)
        print("\nAgent:", result["answer"])
        if result["image_base64"]:
            print(f"(Map image generated, base64 length: {len(result['image_base64'])} chars)")
    except Exception as e:
        print("\nError running agent:", e)

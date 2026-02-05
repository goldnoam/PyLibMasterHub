
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { LibraryDetails } from "../types";

// Removed global API_KEY constant to follow guidelines of using process.env directly

const validateLibraryDetails = (data: any): LibraryDetails => {
  // Defensive normalization to prevent React rendering objects as children (Error #31)
  const ensureString = (val: any, fallback: string = ""): string => {
    if (typeof val === 'string') return val;
    if (val === null || val === undefined) return fallback;
    try {
      return JSON.stringify(val);
    } catch {
      return fallback;
    }
  };

  const ensureArrayOfStrings = (val: any): string[] => {
    if (!Array.isArray(val)) return [];
    return val.map(item => ensureString(item));
  };

  return {
    name: ensureString(data.name, "Unknown Library"),
    description: ensureString(data.description),
    useCases: ensureArrayOfStrings(data.useCases),
    codeExample: ensureString(data.codeExample),
    keyFeatures: ensureArrayOfStrings(data.keyFeatures),
  };
};

export const fetchLibraryDetails = async (libraryName: string): Promise<LibraryDetails> => {
  // Guidelines: Create a new GoogleGenAI instance right before making an API call
  // Guidelines: Use process.env.API_KEY directly in named parameter
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  
  // Upgraded to gemini-3-pro-preview for tasks involving coding and technical documentation
  const response: GenerateContentResponse = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: `Provide a detailed educational overview of the Python library: "${libraryName}". Include its primary purpose, common use cases, 3 key features, and a high-quality "Hello World" or starter code example.`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          description: { type: Type.STRING },
          useCases: { 
            type: Type.ARRAY, 
            items: { type: Type.STRING } 
          },
          codeExample: { type: Type.STRING },
          keyFeatures: { 
            type: Type.ARRAY, 
            items: { type: Type.STRING } 
          },
        },
        required: ["name", "description", "useCases", "codeExample", "keyFeatures"]
      }
    }
  });

  // Guidelines: Use .text property directly (not a method)
  const text = response.text;
  if (!text) {
    throw new Error("No response text received from documentation service");
  }

  try {
    const rawData = JSON.parse(text);
    return validateLibraryDetails(rawData);
  } catch (e) {
    console.error("Failed to parse response:", text);
    throw new Error("Invalid response format from service");
  }
};

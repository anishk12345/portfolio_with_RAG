Personal Portfolio with RAG based chatbot using llama 3.2

This project implements a customizable AI Chatbot API using the Llama model and LangChain to retrieve relevant data from predefined knowledge chunks.

Arguments:
--model_name:	Path or name of the model folder to use	"Llama-3.2-3B-Instruct"
--api_key:	Static api key to route to localhost ( You can use ngrok)

Ex:

python api.py --model_name "Llama-3.2-3B-Instruct" --api_key "https://my-ngrok-url.app" 

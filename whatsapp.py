import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
import time
import random
import string
import os
import json
import uuid
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import os
import re
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from collections import defaultdict
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import string
import time
import threading
import win32clipboard
from PIL import Image
import io
from fpdf import FPDF
import base64

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

openai_api_key = st.secrets["api"]["key"]  # or st.secrets.api.key

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

USER_SESSIONS = defaultdict(lambda: {
    "memory": None,  # This should be set to the user-specific memory object somewhere in your code
    "order_in_progress": False,
    "order_details": {},
    "order_update_in_progress": False,
    "order_update_details": {}
})

DOWNLOAD_DIR = "C:\\Users\\rafay\\Downloads"  # Change as needed

def load_names():
    try:
        with open("stored_names.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

stored_names = load_names()

class RagCode:

    def process_user_query(self, question: str, person_name: str) -> str:
        # 1. Make sure memory is initialized
        if USER_SESSIONS[person_name]["memory"] is None:
            USER_SESSIONS[person_name]["memory"] = ConversationSummaryBufferMemory(
                llm=ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0.5,
                    openai_api_key=openai_api_key
                ),
                memory_key="chat_history"
            )

        # 2. Classify the user’s question
        category = self.classify_question(question,person_name)
        
        # 3. Generate a response using your specialized function
        if category == "General Query":
            response = self.process_general_query(question, person_name)
        elif category == "Product Query":
            response = self.process_product_query(question, person_name)
        elif category == "Order Request":
            response = self.process_order_request(question, person_name)
        elif category == "Order Update":
            response = self.process_order_update(question, person_name)
        else:
            response = f"Unable to classify the question. Received classification: {category}"

        if isinstance(response, dict):
            newly = f"Your order Details:\n{response}"
            # 4. **Save** the user’s question + the assistant’s response to memory
            USER_SESSIONS[person_name]["memory"].save_context(
                {"input": question}, 
                {"output": newly}
            )
            return response
        else:
            # 4. **Save** the user’s question + the assistant’s response to memory
            USER_SESSIONS[person_name]["memory"].save_context(
                {"input": question}, 
                {"output": response}
            )

            # 5. Return the response
            return response

    def classify_question(self,question: str,person_name) -> str:
        # Retrieve conversation history from memory.
        memory_vars = USER_SESSIONS[person_name]["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        prompt_template = """
        You are an assistant that categorizes user queries regarding our ecommerce platform into one of the following categories:
        1. General Query
        Definition: The user seeks specific information about a product, such as:
            Features
            Pricing
            Specifications and in-depth features
            Availability
        Examples:
            "What is the price of Product X?"
            "Is Product Y available in stock?"
            "What are the specifications of Product Y?"
            "Does Product Z have Bluetooth support?"
            "List the names of available laptops."
        2. Product Query
        Definition: The user requests a broader set of details about a product, including:
            A complete product overview
            Specifications and in-depth features
            A list of products within a category
        Examples:
            "Tell me everything about Product X."
            "Can you list all the smartphones under $500?"
            "Show me all available gaming laptops."
        3. Order Request: Requests to place an order.
        4. Order Update: Inquiries regarding the status of an existing order.
        Based on the user question provided below, reply with exactly one of the following words: "General Query","Product Query", "Order Request", or "Order Update". Do not include any extra text.

        Use this conversation history to get the context of conversation and the user question to decide which category the question falls into.:
        {history}

        User question: {question}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key)
        response = model.call_as_llm(prompt.format(history=history,question=question))
        return response.strip()

    def process_general_query(self, question: str,person_name) -> str:
        vectorstore = self.load_products_vectorstore()
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        
        # Load CSV to get all unique product categories.
        df = pd.read_csv("products.csv")
        categories = df["Category"].unique().tolist()
        categories_str = ", ".join(str(category) for category in categories)

        # Retrieve conversation history from memory.
        memory_vars = USER_SESSIONS[person_name]["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")

        print('GENERAL QUERY')
        
        prompt_template = """
        You are a helpful E-commerce chat assistant that uses the following conversation context along with product details to answer the customer's query.
        Genral Info: 
        Mashallah :An online home decor ecommerce brand
        We offer high quality luxury lighting products.We have 30 days replacement guarantee. We deliver in 1-3 days and delivery cost is 80Taka inside Dhaka, 120Taka outside dhaka.

        If a product is available with us, please fetch its specifications from the vector database and give them to the client if the client asks for them. 
        But if a product is not in the context,DO NOT SHARE ITS SPECIFICATIONS; just say that it's not available.
        If user acquire specific product information then provide the details from the internet seemlessly.  
        If the user generally asks for products or categories, you should provide all product categories.
        Donot ever return image path in data.
        Language: The bot should always reply in English spoken in English.
        Conversation History:
        {history}

        Product Details:
        {context}

        All Product Categories: {categories}

        Customer Query: {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question", "categories"])
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({
            "input_documents": docs, 
            "question": question, 
            "categories": categories_str,
            "history": history
        }, return_only_outputs=True)
        
        return response["output_text"]
    
    def process_product_query(self, question: str,person_name) -> str:
        vectorstore = self.load_products_vectorstore()
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        #print('DOCUMENT PRINTTTT:',docs)
        print('PRODUCT QUERY')
        
        # Load CSV to get all unique product categories.
        df = pd.read_csv("products.csv")
        categories = df["Category"].unique().tolist()
        categories_str = ", ".join(str(category) for category in categories)

        # Retrieve conversation history from memory.
        memory_vars = USER_SESSIONS[person_name]["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        
        prompt_template = """
        You are a helpful E-commerce Chatbot answering customer queries based on conversation history and product details.
        
        - If a product is available, fetch its specifications from the vector database. If unavailable, inform the user without providing details.
        - For general product queries, return all relevant available products with full details.
        - Include the image path at the end (formatted as "Image: [image_path]$#$") only when responding to queries about a specific product or product category and returning whole data of product when asked about specific thing of product then donot return.
        - If asked generally about products or categories, list all product categories.
        - Respond in English spoken in English.

        Conversation History:
        {history}

        Product Details:
        {context}

        All Product Categories:
        {categories}

        Customer Query:
        {question}

        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question", "categories"])
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({
            "input_documents": docs, 
            "question": question, 
            "categories": categories_str,
            "history": history
        }, return_only_outputs=True)
        response = response["output_text"]
        print(response)
        if "$#$" in response:
            response = self.parse_product_response(response)

        print('RESPONSE',response)
        
        return response

    def process_order_request(self, question: str, person_name: str) -> str:

        user_data = USER_SESSIONS[person_name]
        print('ORDER QUERY')

        # Helper: Verify the product via vector store.
        def verify_product(product_name: str) -> (bool, str):
            vectorstore = self.load_products_vectorstore()
            results = vectorstore.similarity_search(product_name, k=3)
            suggestions = []
            for r in results:
                for line in r.page_content.splitlines():
                    if line.startswith("Product:"):
                        suggestions.append(line.replace("Product:", "").strip())
            if any(product_name.lower() == s.lower() for s in suggestions):
                return True, ""
            else:
                suggestions_str = ", ".join(suggestions) if suggestions else "No similar products found"
                return False, (
                    f"Sorry, I couldn't find an exact match for '{product_name}'. "
                    f"Did you mean one of these? {suggestions_str}\n\n"
                    "Please confirm the correct product name or type 'cancel' to stop."
                )

        # Helper: Get product price from CSV.
        # Returns an integer price; if not found or conversion fails, returns 0.
        def get_product_price(product_name: str) -> int:
            try:
                df = pd.read_csv("products.csv")
                # Clean column names by stripping extra characters and whitespace.
                df.columns = df.columns.str.strip().str.replace(r"^\[", "", regex=True)
                df.columns = df.columns.str.strip().str.replace(r"\]$", "", regex=True)
                # Compare names ignoring extra spaces and case.
                row = df[df['Name'].str.strip().str.lower() == product_name.lower()]
                if not row.empty:
                    price_val = row.iloc[0]['Price']
                    try:
                        return int(price_val)
                    except:
                        return 0
            except Exception as e:
                print("Error in get_product_price:", e)
            return 0

        # Helper: Fill missing prices based on the product list.
        # If the provided price is not a valid integer (or is a placeholder),
        # it replaces it with the price fetched from the CSV.
        def fill_missing_prices(product_list, price_list):
            if len(price_list) < len(product_list):
                price_list.extend([""] * (len(product_list) - len(price_list)))
            filled = []
            for product, price in zip(product_list, price_list):
                try:
                    # Try converting the provided price to int.
                    numeric_price = int(price)
                    filled.append(numeric_price)
                except:
                    # If conversion fails (e.g. placeholder text), get price from CSV.
                    filled.append(get_product_price(product))
            return filled

        # Helper: Write order info to CSV (without the Price key).
        def write_order_to_csv(order_info: dict):
            csv_order_info = order_info.copy()
            csv_order_info.pop("Price", None)
            order_df = pd.DataFrame([csv_order_info])
            if os.path.exists("orders.csv"):
                order_df.to_csv("orders.csv", mode='a', header=False, index=False)
            else:
                order_df.to_csv("orders.csv", index=False)

        # Helper: Remove markdown formatting from LLM responses.
        def sanitize_llm_response(response_str: str) -> str:
            response_str = response_str.strip()
            if response_str.startswith("```"):
                lines = response_str.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_str = "\n".join(lines).strip()
            return response_str

        # Helper: Extract JSON block from a response.
        def extract_json_from_response(response_str: str) -> str:
            start = response_str.find('{')
            end = response_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                return response_str[start:end+1]
            return response_str

        # 1. Cancellation check.
        if question.strip().lower() in ["cancel", "cancel order", "stop"]:
            user_data["order_in_progress"] = False
            user_data["order_details"] = {}
            return "Order has been cancelled."

        # 2. If an order is mentioned, try to extract a product from the request and add it to the product list.
        if "order" in question.lower():
            match = re.search(r'order\s+(?:an|a)\s+([A-Za-z0-9\s\-]+)', question, re.IGNORECASE)
            if match:
                extracted_product = match.group(1).strip()
                valid, message = verify_product(extracted_product)
                if not valid:
                    return message
                order_details = user_data.setdefault("order_details", {})
                if "Product" in order_details:
                    if not isinstance(order_details["Product"], list):
                        order_details["Product"] = [order_details["Product"]]
                    if extracted_product not in order_details["Product"]:
                        order_details["Product"].append(extracted_product)
                else:
                    order_details["Product"] = [extracted_product]
                user_data["order_in_progress"] = True

        # 3. Try parsing the entire input as JSON (final confirmation case).
        try:
            input_data = json.loads(question)
            if isinstance(input_data, dict) and input_data.get("order_status") == "confirmed":
                prod_val = input_data.get("Product", "")
                product_list = prod_val if isinstance(prod_val, list) else [prod_val.strip()]
                for product in product_list:
                    valid, message = verify_product(product)
                    if not valid:
                        return message
                price_val = input_data.get("Price", "")
                price_list = price_val if isinstance(price_val, list) else [price_val.strip()]
                quantity_val = input_data.get("Quantity", "")
                quantity_list = quantity_val if isinstance(quantity_val, list) else [quantity_val.strip()]
                
                # Fill missing prices, ensuring each price is an integer (or 0 if not found).
                price_list = fill_missing_prices(product_list, price_list)
                
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": input_data.get("Name", ""),
                    "Address": input_data.get("Address", ""),
                    "Product": product_list,
                    "Price": price_list,
                    "Quantity": quantity_list,
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
        except json.JSONDecodeError:
            pass

        # 4. If the user sends a simple confirmation phrase.
        if question.strip().lower() in ["go ahead", "confirm", "okay thanks"]:
            required_keys = ["Name", "Address", "Product", "Quantity"]
            details = user_data.get("order_details", {})
            if all(k in details for k in required_keys):
                prod_val = details.get("Product", "")
                product_list = prod_val if isinstance(prod_val, list) else [prod_val.strip()]
                for product in product_list:
                    valid, message = verify_product(product)
                    if not valid:
                        return message
                price_val = details.get("Price", "")
                price_list = price_val if isinstance(price_val, list) else [price_val.strip()]
                quantity_val = details.get("Quantity", "")
                quantity_list = quantity_val if isinstance(quantity_val, list) else [quantity_val.strip()]
                
                # Fill missing prices.
                price_list = fill_missing_prices(product_list, price_list)
                
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": details.get("Name", ""),
                    "Address": details.get("Address", ""),
                    "Product": product_list,
                    "Price": price_list,
                    "Quantity": quantity_list,
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
            else:
                user_data["order_in_progress"] = True

        # 5. If session order details are already flagged as confirmed.
        if user_data.get("order_details", {}).get("order_status") == "confirmed":
            confirmed_details = user_data["order_details"]
            prod_val = confirmed_details.get("Product", "")
            product_list = prod_val if isinstance(prod_val, list) else [prod_val.strip()]
            for product in product_list:
                valid, message = verify_product(product)
                if not valid:
                    return message
            price_val = confirmed_details.get("Price", "")
            price_list = price_val if isinstance(price_val, list) else [price_val.strip()]
            quantity_val = confirmed_details.get("Quantity", "")
            quantity_list = quantity_val if isinstance(quantity_val, list) else [quantity_val.strip()]
            
            # Fill missing prices.
            price_list = fill_missing_prices(product_list, price_list)
            
            order_id = str(uuid.uuid4())
            order_info = {
                "Order ID": order_id,
                "Name": confirmed_details.get("Name", ""),
                "Address": confirmed_details.get("Address", ""),
                "Product": product_list,
                "Price": price_list,
                "Quantity": quantity_list,
                "Status": "received",
            }
            write_order_to_csv(order_info)
            user_data["order_in_progress"] = False
            user_data["order_details"] = {}
            return order_info

        # 6. Otherwise, call the LLM to get missing details.
        user_data["order_in_progress"] = True
        memory_vars = user_data["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        existing_order_details = user_data.get("order_details", {})

        prompt_template = """
        You are an E-commerce Chatbot that helps complete a customer's product order step by step.
        Genral Info: 
        Mashallah :An online home decor ecommerce brand
        We offer high quality luxury lighting products. We have 30 days replacement guarantee. We deliver in 1-3 days and delivery cost is 80Taka inside Dhaka, 120Taka outside Dhaka.

        Review the conversation history and the user's latest input. We need the following keys:
        Name, Address, Product, Quantity.
        Note Do not explicitily tells the amount of products, assume that the quantity of the product is 1.

        Steps:
        1) If the user is missing any of these details, ask for them naturally (Except Quantity).
        2) If the user has provided them all, respond with a JSON ONLY.

        The JSON must look like:
        {{
            "Name": ...,
            "Address": ...,
            "Product": [...],
            "Quantity": [...],
            "Price": [...],
            "order_status": "confirmed"
        }}

        3) Do NOT ask for payment or other info.
        4) If the user ever says 'cancel', the order is cancelled.
        Language: The bot should always reply in English spoken in English.
        Conversation History:
        {history}

        Latest User Input:
        {question}

        Current Collected Order Details (if any):
        {order_details}

        Now please respond with either:
        - A question to fill in missing info, OR
        - A final JSON if all details are known.
            """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["history", "question", "order_details"],
        )
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        llm_response = model.call_as_llm(
            prompt.format(
                history=history,
                question=question,
                order_details=json.dumps(existing_order_details),
            )
        )

        # 7. Try to extract and parse the JSON block from the LLM response.
        try:
            sanitized_response = sanitize_llm_response(llm_response)
            json_text = extract_json_from_response(sanitized_response)
            response_json = json.loads(json_text)
            if response_json.get("order_status") == "confirmed":
                prod_val = response_json.get("Product", "")
                product_list = prod_val if isinstance(prod_val, list) else [prod_val.strip()]
                for product in product_list:
                    valid, message = verify_product(product)
                    if not valid:
                        return message
                price_val = response_json.get("Price", "")
                price_list = price_val if isinstance(price_val, list) else [price_val.strip()]
                quantity_val = response_json.get("Quantity", "")
                quantity_list = quantity_val if isinstance(quantity_val, list) else [quantity_val.strip()]
                
                # Fill missing prices.
                price_list = fill_missing_prices(product_list, price_list)
                
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": response_json.get("Name", ""),
                    "Address": response_json.get("Address", ""),
                    "Product": product_list,
                    "Price": price_list,
                    "Quantity": quantity_list,
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
            else:
                user_data["order_details"].update(response_json)
                return llm_response
        except json.JSONDecodeError:
            return llm_response

    def process_order_requestold(self, question: str, person_name: str) -> str:
        """
    Orchestrates the user's order request by gathering details, verifying product availability,
    and confirming the order. If all details are provided and confirmed, it stores the order in 'orders.csv'
    and returns a confirmation message including the order ID. Otherwise, it returns a natural language response
    prompting for more information.
    
    Returns:
        str: A natural language response or a confirmation message with order details.
    """
        user_data = USER_SESSIONS[person_name]
        print('ORDER QUERY')

        # Helper: Verify the product via vector store.
        def verify_product(product_name: str) -> (bool, str):
            vectorstore = self.load_products_vectorstore()
            results = vectorstore.similarity_search(product_name, k=3)
            suggestions = []
            for r in results:
                for line in r.page_content.splitlines():
                    if line.startswith("Product:"):
                        suggestions.append(line.replace("Product:", "").strip())
            if any(product_name.lower() == s.lower() for s in suggestions):
                return True, ""
            else:
                suggestions_str = ", ".join(suggestions) if suggestions else "No similar products found"
                return False, (
                    f"Sorry, I couldn't find an exact match for '{product_name}'. "
                    f"Did you mean one of these? {suggestions_str}\n\n"
                    "Please confirm the correct product name or type 'cancel' to stop."
                )

        # Helper: Write order info to CSV.
        def write_order_to_csv(order_info: dict):
            order_df = pd.DataFrame([order_info])
            if os.path.exists("orders.csv"):
                order_df.to_csv("orders.csv", mode='a', header=False, index=False)
            else:
                order_df.to_csv("orders.csv", index=False)

        # Helper: Remove markdown formatting from LLM responses.
        def sanitize_llm_response(response_str: str) -> str:
            response_str = response_str.strip()
            if response_str.startswith("```"):
                lines = response_str.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_str = "\n".join(lines).strip()
            return response_str

        # Helper: Extract JSON block from a response.
        def extract_json_from_response(response_str: str) -> str:
            start = response_str.find('{')
            end = response_str.rfind('}')
            if start != -1 and end != -1 and end > start:
                return response_str[start:end+1]
            return response_str

        # 1. Cancellation check.
        if question.strip().lower() in ["cancel", "cancel order", "stop"]:
            user_data["order_in_progress"] = False
            user_data["order_details"] = {}
            return "Order has been cancelled."

        # 2. If no product tracked yet, try to extract from the order request.
        if "order" in question.lower() and "Product" not in user_data.get("order_details", {}):
            match = re.search(r'order\s+(?:an|a)\s+([A-Za-z0-9\s\-]+)', question, re.IGNORECASE)
            if match:
                extracted_product = match.group(1).strip()
                valid, message = verify_product(extracted_product)
                if not valid:
                    return message
                user_data.setdefault("order_details", {})["Product"] = extracted_product
                user_data["order_in_progress"] = True

        # 3. Try parsing the entire input as JSON (final confirmation case).
        try:
            input_data = json.loads(question)
            if isinstance(input_data, dict) and input_data.get("order_status") == "confirmed":
                product_name = input_data.get("Product", "").strip()
                valid, message = verify_product(product_name)
                if not valid:
                    return message
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": input_data.get("Name", ""),
                    "Address": input_data.get("Address", ""),
                    "Product": product_name,
                    "Price": input_data.get("Price", ""),
                    "Quantity": input_data.get("Quantity", ""),
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
        except json.JSONDecodeError:
            # Not valid JSON; continue with LLM flow.
            pass

        # 4. If the user sends a simple confirmation phrase.
        if question.strip().lower() in ["go ahead", "confirm", "okay thanks"]:
            required_keys = ["Name", "Address", "Product", "Quantity"]
            details = user_data.get("order_details", {})
            if all(k in details for k in required_keys):
                product_name = details["Product"].strip()
                valid, message = verify_product(product_name)
                if not valid:
                    return message
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": details.get("Name", ""),
                    "Address": details.get("Address", ""),
                    "Product": product_name,
                    "Price": details.get("Price", ""),
                    "Quantity": details.get("Quantity", ""),
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
            else:
                user_data["order_in_progress"] = True

    # 5. If session order details are already flagged as confirmed.
        if user_data.get("order_details", {}).get("order_status") == "confirmed":
            confirmed_details = user_data["order_details"]
            product_name = confirmed_details.get("Product", "").strip()
            valid, message = verify_product(product_name)
            if not valid:
                return message
            order_id = str(uuid.uuid4())
            order_info = {
                "Order ID": order_id,
                "Name": confirmed_details.get("Name", ""),
                "Address": confirmed_details.get("Address", ""),
                "Product": product_name,
                "Price": confirmed_details.get("Price", ""),
                "Quantity": confirmed_details.get("Quantity", ""),
                "Status": "received",
            }
            write_order_to_csv(order_info)
            user_data["order_in_progress"] = False
            user_data["order_details"] = {}
            return order_info

        # 6. Otherwise, call the LLM to get missing details.
        user_data["order_in_progress"] = True
        memory_vars = user_data["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        existing_order_details = user_data.get("order_details", {})

        prompt_template = """
You are an E-commerce Chatbot that helps complete a customer's product order step by step.
Genral Info: 
Mashallah :An online home decor ecommerce brand
We offer high quality luxury lighting products.We have 30 days replacement guarantee. We deliver in 1-3 days and delivery cost is 80Taka inside Dhaka, 120Taka outside dhaka.

Review the conversation history and the user's latest input. We need the following keys:
Name, Address, Product, Quantity.

Steps:
1) If the user is missing any of these details, ask for them naturally.
2) If the user has provided them all, respond with a JSON ONLY.

The JSON must look like:
{{
    "Name": ...,
    "Address": ...,
    "Product": ...,
    "Quantity": ...,
    "Price": ...,
    "order_status": "confirmed"
}}

3) Do NOT ask for payment or other info.
4) If the user ever says 'cancel', the order is cancelled.
Language: The bot should always reply in English spoken in English.
Conversation History:
{history}

Latest User Input:
{question}

Current Collected Order Details (if any):
{order_details}

Now please respond with either:
- A question to fill in missing info, OR
- A final JSON if all details are known.
    """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["history", "question", "order_details"],
        )
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        llm_response = model.call_as_llm(
            prompt.format(
                history=history,
                question=question,
                order_details=json.dumps(existing_order_details),
            )
        )

        # 7. Try to extract and parse the JSON block from the LLM response.
        try:
            sanitized_response = sanitize_llm_response(llm_response)
            json_text = extract_json_from_response(sanitized_response)
            response_json = json.loads(json_text)
            if response_json.get("order_status") == "confirmed":
                product_name = response_json.get("Product", "").strip()
                valid, message = verify_product(product_name)
                if not valid:
                    return message
                order_id = str(uuid.uuid4())
                order_info = {
                    "Order ID": order_id,
                    "Name": response_json.get("Name", ""),
                    "Address": response_json.get("Address", ""),
                    "Product": product_name,
                    "Price": response_json.get("Price", ""),
                    "Quantity": response_json.get("Quantity", ""),
                    "Status": "received",
                }
                write_order_to_csv(order_info)
                user_data["order_in_progress"] = False
                user_data["order_details"] = {}
                return order_info
            else:
                # If the JSON is partial, update session and return the LLM's conversational reply.
                user_data["order_details"].update(response_json)
                return llm_response
        except json.JSONDecodeError:
            # If no valid JSON is extracted, return the LLM's natural language reply.
            return llm_response

    def process_order_update(self,question: str,person_name) -> str:
        user_data = USER_SESSIONS[person_name]
        print('ORDER UPDATE QUERY')
        # Helper function to extract an Order ID from the text.
        def extract_order_id(text: str) -> str:
            # Try to match a UUID pattern.
            uuid_pattern = r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}"
            m = re.search(uuid_pattern, text)
            if m:
                return m.group(0)
            # Otherwise, try to match a sequence of digits (at least 5 digits).
            num_pattern = r"\b\d{5,}\b"
            m = re.search(num_pattern, text)
            if m:
                return m.group(0)
            return None

        # Check for cancellation command.
        if question.strip().lower() in ["cancel", "cancel update", "stop"]:
            user_data["order_update_in_progress"] = False
            user_data["order_update_details"] = {}
            return "Order update process has been cancelled."

        # Attempt to extract an Order ID from the user's message.
        extracted_order_id = extract_order_id(question)
        if extracted_order_id is not None:
            # If an Order ID is found, try to look it up in the CSV.
            if not os.path.exists("orders.csv"):
                user_data["order_update_in_progress"] = False
                user_data["order_update_details"] = {}
                return "No orders found in our records."
            orders_df = pd.read_csv("orders.csv")
            # Use the exact CSV header "Order Id"
            matched_order = orders_df[orders_df["Order ID"].astype(str) == extracted_order_id]
            if matched_order.empty:
                user_data["order_update_in_progress"] = False
                user_data["order_update_details"] = {}
                return f"No order found for the Order ID {extracted_order_id}."
            else:
                order_info = matched_order.iloc[0]
                name = order_info.get("Name", "N/A")
                product = order_info.get("Product", "N/A")
                status = order_info.get("Status", "N/A")
                user_data["order_update_in_progress"] = False
                user_data["order_update_details"] = {}
                return f"Order Details: Name: {name}, Product: {product}, Status: {status}"
        
        # If no Order ID is present in the message, then call the LLM to extract it.
        memory_vars = user_data["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        update_details = user_data.get("order_update_details", {})

        prompt_template = """
        You are an E-Commerce assistant that helps customers check the status of their orders.

        Based on the conversation history below and the customer's latest input determine if user is asking for his order information or is generally asking for delievery time and information.
        If user is generally asking about delievery details than use this information to give the client the info.
        - We offer high quality luxury lighting products.We have 30 days replacement guarantee. We deliver in 1-3 days and delivery cost is 80Taka inside Dhaka, 120Taka outside dhaka.
        If user is asking for his order details extract the following detail if provided:
        - Order ID

        If the Order ID is missing, ask a follow-up question in a natural conversational manner to obtain the missing Order ID.
        If the Order ID is provided, output a confirmation in JSON format with the following keys:
        "Order ID": customer's order id,
        "lookup_status": "ready"

        If the customer types "cancel" at any point, cancel the update process.

        Language: The bot should always reply in English spoken in English.

        Conversation History:
        {history}

        Current Customer Input:
        {question}

        Existing Order Update Details (if any):
        {update_details}

        Respond:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question", "update_details"])
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        llm_response = model.call_as_llm(prompt.format(
            history=history,
            question=question,
            update_details=json.dumps(update_details)
        ))
        
        try:
            response_json = json.loads(llm_response)
            if response_json.get("lookup_status") == "ready" and response_json.get("Order ID"):
                order_id = response_json.get("Order ID").strip()
                if not os.path.exists("orders.csv"):
                    user_data["order_update_in_progress"] = False
                    user_data["order_update_details"] = {}
                    return "No orders found in our records."
                orders_df = pd.read_csv("orders.csv")
                matched_order = orders_df[orders_df["Order Id"].astype(str) == order_id]
                if matched_order.empty:
                    user_data["order_update_in_progress"] = False
                    user_data["order_update_details"] = {}
                    return f"No order found for the Order ID {order_id}."
                else:
                    order_info = matched_order.iloc[0]
                    name = order_info.get("Name", "N/A")
                    product = order_info.get("Product", "N/A")
                    status = order_info.get("Status", "N/A")
                    user_data["order_update_in_progress"] = False
                    user_data["order_update_details"] = {}
                    return f"Order Details: Name: {name}, Product: {product}, Status: {status}"
            else:
                user_data["order_update_details"].update(response_json)
                user_data["order_update_in_progress"] = True
                return llm_response
        except json.JSONDecodeError:
            user_data["order_update_in_progress"] = True
            return llm_response

    def parse_product_response(self, response: str):
        """
        Parses the product information from the response text.
        Splits multiple products on '$#$' and returns a list of dicts.
        """
        # 1) Split the full text on "$#$"
        raw_products = response.split("$#$")
        parsed_products = []

        for raw_product in raw_products:
            product = raw_product.strip()
            if not product:
                # Skip empty product segments (sometimes happens if text ends with $#$)
                continue

            product_info = {}

            # 2) Extract the product name
            #    Example: 1. **Diamond Table Lamp**
            #    We'll match the text between '**' pairs after a digit and a dot.
            name_match = re.search(r"\d+\.\s*\*\*(.*?)\*\*", product)

            product = product.replace("*", "")

            # 3) Extract Price, e.g. "- Price: 40"
            price_match = re.search(r"Price:\s*(\d+)", product)

            # 4) Extract Description, e.g. "- Description: (some text)..."
            description_match = re.search(r"Description:\s*(.*?)(?:\n|$)", product)

            # 5) Extract Availability, e.g. "- Availability: In Stock"
            availability_match = re.search(r"Availability:\s*(.*?)(?:\n|$)", product)

            # 6) Extract Image (manual slicing, no regex)
            #    We look for the substring after "Image:" until the next "$#$" or end of string.
            image_val = None
            if "Image:" in product:
                # Find start position right after "Image:"
                start_idx = product.index("Image:") + len("Image:")
                # Slice out everything to the end (we'll trim if "$#$" is found)
                temp = product[start_idx:].strip()

                # If there's still a "$#$" leftover inside this chunk, cut it off
                # (in case the product text has "$#$" inside)
                dollar_sep_index = temp.find("$#$")
                if dollar_sep_index != -1:
                    temp = temp[:dollar_sep_index].strip()

                # Remove any leading '.' (e.g. ".pic_1" → "pic_1")
                temp = temp.lstrip('.')

                image_val = temp

            # 7) Assign to product_info if found
            if name_match:
                product_info["Name"] = name_match.group(1).strip()
            if price_match:
                product_info["Price"] = int(price_match.group(1))
            if description_match:
                product_info["Description"] = description_match.group(1).strip()
            if availability_match:
                product_info["Availability"] = availability_match.group(1).strip()
            if image_val:
                product_info["Image"] = image_val

            # 8) Only add if we found relevant product details
            if product_info:
                parsed_products.append(product_info)

        return parsed_products   

    def load_products_vectorstore(
        self,
        csv_path: str = "products.csv", 
        vectorstore_path: str = "products_faiss_index"
    ):
        # Step 1: Instantiate embeddings.
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Step 2: Check if our FAISS index already exists.
        if os.path.exists(vectorstore_path):
            # Load existing vectorstore
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        else:
            # Create a new vectorstore
            vectorstore = None

        # Step 3: Read the CSV to a DataFrame
        df = pd.read_csv(csv_path)

        # Create a list to hold documents for new rows
        new_docs = []

        if vectorstore:
            # We have an existing vectorstore
            existing_docstore = vectorstore.docstore._dict
            existing_ids = set()
            for doc_id, stored_doc in existing_docstore.items():
                product_id = stored_doc.metadata.get("product_id")
                if product_id:
                    existing_ids.add(product_id)

            # Filter for new rows
            df_new = df[~df["ID"].isin(existing_ids)]
        else:
            # Vectorstore does not exist, so everything is "new"
            df_new = df

        # Step 4: Create Documents for newly added rows
        for _, row in df_new.iterrows():
            content = (
                f"ID: {row['ID']}\n"
                f"Product: {row['Name']}\n"
                f"Price: {row['Price']}\n"
                f"Category: {row['Category']}\n"
                f"image_path: {row['image']}\n"
                f"Description: {row['Description']}\n"
                f"Availability: {row['Availability']}\n"
            )
            doc = Document(
                page_content=content,
                metadata={"product_id": row["ID"]}  # store unique ID in metadata
            )
            new_docs.append(doc)

        # Step 5: If there are new documents, embed them and add them to vectorstore
        if new_docs:
            if vectorstore is None:
                # No existing vectorstore: create it
                vectorstore = FAISS.from_documents(new_docs, embedding=embeddings)
            else:
                # Add new docs to existing vectorstore
                vectorstore.add_documents(new_docs)

            # Save vectorstore to local path
            vectorstore.save_local(vectorstore_path)

        # Finally, return the (possibly updated) vectorstore
        return vectorstore

    def get_product_details_from_picture(self, image, person_name, csv_path="products.csv"):
        """
        Extracts product details from a given image by performing OCR to detect a product ID and 
        then looking up the corresponding product details in a CSV file.
        
        The CSV is expected to have the following columns:
        ID, Name, Price, Category, Description, Photo
        
        Parameters:
            image: A PIL image.
            csv_path (str): The path to the CSV file containing product details.
            
        Returns:
            dict or None: A dictionary containing product details if a matching ID is found, otherwise None.
        """
        # 1. Make sure memory is initialized
        if USER_SESSIONS[person_name]["memory"] is None:
            USER_SESSIONS[person_name]["memory"] = ConversationSummaryBufferMemory(
                llm=ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0.5,
                    openai_api_key=openai_api_key
                ),
                memory_key="chat_history"
            )
        # Get the user-specific session data
        user_data = USER_SESSIONS[person_name]
        
        # Open the image using PIL.
        images = Image.open(image)
        
        # Convert the image to a numpy array.
        image_np = np.array(images)
        
        # Initialize the PaddleOCR reader and process the image.
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        results = ocr.ocr(image_np, cls=True)
        
        # Extract text strings from the OCR results.
        # PaddleOCR returns a list of lists. Each sublist contains detected texts as tuples.
        extracted_texts = []
        for line in results:
            for box_info in line:
                # box_info[1] is a tuple: (recognized_text, confidence)
                text = box_info[1][0]
                extracted_texts.append(text)
        
        # Look for a text starting with "ID:" and extract the number after it.
        product_id = None
        for text in extracted_texts:
            if text.startswith("ID:"):
                id_text = text[3:].strip()  # Extract only the number after "ID:"
                try:
                    product_id = int(id_text)
                except ValueError:
                    return f"Product Not Found!"
                break

        print("it continues here")

        if product_id is None:
            return "No product ID found in the image."

        # Load the CSV file.
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            return f"CSV file not found at: {csv_path}"

        # Look for a matching product ID in the CSV.
        product_row = df[df["ID"] == product_id]
        if product_row.empty:
            return f"Product with ID {product_id} not found in the CSV."

        # Convert the found product details to a dictionary.
        product_details = product_row.to_dict(orient="records")[0]
        print("Product found:")
        for key, value in product_details.items():
            print(f"{key}: {value}")
        
        # Retrieve conversation context from memory.
        memory_vars = user_data["memory"].load_memory_variables({})
        history = memory_vars.get("chat_history", "")
        
        # Format the product details into a multiline string.
        product_info_str = "\n".join([f"{key}: {value}" for key, value in product_details.items()])
        
        prompt_template = """
            You are a helpful Ecommerce assistant that uses the conversation context along with product details 
            to generate a concise description of a product. It should include the product name and price.
            Based on the following product details, please provide a natural language response 
            that describes the product, including its name, price. (Dont use the introducing word).

            Language: The bot should always reply in English spoken in English.

            Conversation History:
            {history}

            Product Details:
            {product_details}

            Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["history", "product_details"])
        model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        response = model.call_as_llm(prompt.format(history=history, product_details=product_info_str))
        
        # Save this interaction into conversation memory for follow-up queries.
        user_data["memory"].save_context(
            {"input": "Shared image for product details!"},
            {"output": response}
        )
        print("printer: ", response)
        return response


ragbot = RagCode()

# Create chatbot class
class EnhancedChatbot:
    def parse_input_to_list(self,input_data):
        """
        Merged function:
        1) If input_data is a dictionary, return a list of "Key: Value" strings.
        2) Otherwise, treat input_data as text and return a list of non-empty lines 
        (ignoring blank lines).
        """
        if isinstance(input_data, dict):
            return [f"{k}: {v}" for k, v in input_data.items()]
        else:
            # Convert to string in case it's something other than str
            text = str(input_data)
            # Split into lines, remove empty lines, strip whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return lines

    def get_latest_file(self,directory):
        """Returns the latest file in the given directory."""
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        return max(files, key=os.path.getctime) if files else None

    def simulate_typing(self,driver, stop_event):
        """
        Continuously types random text in the WhatsApp message box until stop_event is set.
        """
        while not stop_event.is_set():
            try:
                # Locate the message box
                message_box = driver.find_element(By.XPATH, "//div[@aria-placeholder='Type a message']")
                
                # Generate some random text
                random_text = ''.join(
                    random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 12))
                )
                message_box.send_keys(random_text)
                
                # Sleep a short random interval to mimic human typing gaps
                time.sleep(random.uniform(0.3, 1.2))

            except NoSuchElementException:
                print("Could not find the message input field while simulating typing.")
                break
            except Exception as e:
                print(f"Typing simulation encountered an error: {str(e)}")
                break
    
    def send_image_headless(self, image_path_dict, driver):
        """
        Sends a regular image (not sticker) WITHOUT any caption in headless browser mode to WhatsApp Web.
        Then follows up with a description message if needed.
        
        Args:
            image_path_dict: Dictionary containing image path information
            driver: Selenium WebDriver instance
        """
        
        # Validate image exists in dictionary
        if not image_path_dict.get("Image"):
            print("No image specified.")
            return False
            
        # Process the image file path
        image_file = image_path_dict["Image"].replace("**", "").replace(" ", "")
        current_dir = os.getcwd()
        images_dir = os.path.join(current_dir, "images")
        file_path = os.path.abspath(os.path.join(images_dir, image_file))
        
        # Retrieve and clean up each value if present.
        description = name = price = availability = None

        if image_path_dict.get("Description"):
            text_desc = image_path_dict["Description"].replace("**", "")
            description = f"Description : {text_desc}"

        if image_path_dict.get("Name"):
            text_name = image_path_dict["Name"].replace("**", "")
            name = f"Name : {text_name}"

        if image_path_dict.get("Price"):
            text_price = str(image_path_dict["Price"]).replace("**", "")
            price = f"Price : TK{text_price}"

        if image_path_dict.get("Availability"):
            text_avail = image_path_dict["Availability"].replace("**", "")
            availability = f"Availability : {text_avail}"
        
        # Check if the image file exists
        if not os.path.exists(file_path):
            print(f"Image file not found: {file_path}")
            return False
        
        print(f"Attempting to send image (no caption): {file_path}")
        
        # First try to clear any existing dialogs
        try:
            driver.execute_script("""
                // Close any open dialogs
                var dialogs = document.querySelectorAll('div[role="dialog"]');
                for(var i=0; i<dialogs.length; i++) {
                    var closeButton = dialogs[i].querySelector('span[data-icon="x"]');
                    if(closeButton) closeButton.click();
                }
            """)
            time.sleep(1)
        except:
            pass

        try:
            print("Starting image upload process with specified locators")
            
            # First find the proper message input field to ensure we're in the right chat
            try:
                message_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[@aria-placeholder='Type a message']"))
                )
                message_box.click()
                print("Message box found and clicked")
            except Exception as e:
                print(f"Warning: Could not find message box: {str(e)}")
            
            # Find the attach button using only the specified locator
            try:
                attach_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@title='Attach']"))
                )
                attach_button.click()
                print("Attach button clicked")
                time.sleep(1)
            except Exception as e:
                print(f"Could not find attach button: {str(e)}")
                return False
            
            # Look specifically for the file input using the specified locator
            try:
                file_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//input[@accept='image/*,video/mp4,video/3gpp,video/quicktime']"))
                )
                file_input.send_keys(file_path)
                print("Sent file path directly to input element")
            except Exception as e:
                print(f"Failed to interact with Photos & videos button: {str(e)}")
                return False
            
            # Wait for the image preview to appear
            time.sleep(2)
            
            # Use only the specified locator for the send button
            try:
                send_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(@data-icon,'send')]"))
                )
                send_button.click()
                print("Send button clicked - image sent successfully!")
            except Exception as e:
                print(f"Failed to click send button: {str(e)}")
                return False
            
            try:
                message_box = driver.find_element(By.XPATH, "//div[@aria-placeholder='Type a message']")
                if name:
                    message_box.send_keys(name)
                    print("Name added")
                    message_box.send_keys(Keys.SHIFT, Keys.ENTER)

                if price:
                    message_box.send_keys(price)
                    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
                if availability:
                    message_box.send_keys(availability)
                    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
                if description:
                    message_box.send_keys(description)

                time.sleep(2)
                message_box.send_keys(Keys.ENTER)
                print("Description message sent")
                return True
            except Exception as e:
                print(f"Failed to send message: {str(e)}")
                return False
                        
        except Exception as e:
            print(f"Image upload process failed: {str(e)}")
        
        print("Failed to send the image")
        return False
                
    def save_to_pdf(self, Dict, logo_path, company_name, driver, file_name="Invoice.pdf"):
        """Save formatted invoice data to a PDF with a top-left logo, company name, and a table.
        Supports multiple products: displays each product with its unit price, quantity, subtotal,
        and calculates a final total. This PDF is intended for clients.
        """
        # Process product details from Dict
        products = Dict.get('Product', [])
        prices = Dict.get('Price', [])
        quantities = Dict.get('Quantity', [])
        
        # Ensure they are lists (if not, convert them)
        if not isinstance(products, list):
            products = [products]
        if not isinstance(prices, list):
            prices = [prices]
        if not isinstance(quantities, list):
            quantities = [quantities]
        
        # Calculate per-product subtotals and final total
        sub_totals = []
        final_total = 0
        for pr, q in zip(prices, quantities):
            try:
                unit_price = float(pr)
            except (ValueError, TypeError):
                unit_price = 0
            try:
                qty = int(q)
            except (ValueError, TypeError):
                qty = 0
            subtotal = unit_price * qty
            sub_totals.append(subtotal)
            final_total += subtotal
        
        # Update Dict with a summary Total (for reference, not used in PDF table)
        Dict['Total'] = f"Subtotals: {sub_totals}, Final Total: {final_total}"
        
        # Create the PDF invoice
        pdf = FPDF()
        pdf.add_page()
        
        # Add logo (if exists)
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=10, w=30)
        
        pdf.ln(20)
        
        # Add the "INVOICE" title
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, txt="INVOICE", ln=True, align="C")
        pdf.ln(5)
        
        # Add company name below the title
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, txt=f"Company: {company_name}", ln=True, align="L")
        pdf.ln(10)
        
        # Display header details (all keys except product details)
        header_keys = [key for key in Dict.keys() if key not in ['Product', 'Price', 'Quantity', 'Total', 'Status']]
        row_height = 10
        col_width_1 = 50
        col_width_2 = 130
        for key in header_keys:
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(col_width_1, row_height, key, border=1, align="C")
            pdf.set_font("Arial", size=12)
            pdf.cell(col_width_2, row_height, str(Dict.get(key)), border=1)
            pdf.ln(row_height)
        
        pdf.ln(10)
        
        # Create a table for product details
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt="Products", ln=True, align="C")
        pdf.ln(5)
        
        # Define column widths for the product table:
        # Columns: Product, Unit Price, Quantity, Subtotal
        col_widths = [60, 40, 40, 40]
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(col_widths[0], row_height, "Product", border=1, align="C")
        pdf.cell(col_widths[1], row_height, "Unit Price", border=1, align="C")
        pdf.cell(col_widths[2], row_height, "Quantity", border=1, align="C")
        pdf.cell(col_widths[3], row_height, "Subtotal", border=1, align="C")
        pdf.ln(row_height)
        
        pdf.set_font("Arial", size=12)
        # Draw each product row
        for i in range(len(products)):
            pdf.cell(col_widths[0], row_height, str(products[i]), border=1)
            pdf.cell(col_widths[1], row_height, str(prices[i]), border=1, align="C")
            pdf.cell(col_widths[2], row_height, str(quantities[i]), border=1, align="C")
            pdf.cell(col_widths[3], row_height, str(sub_totals[i]), border=1, align="C")
            pdf.ln(row_height)
        
        # Add a final row for the overall total
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(col_widths[0] + col_widths[1] + col_widths[2], row_height, "Final Total", border=1, align="C")
        pdf.cell(col_widths[3], row_height, str(final_total), border=1, align="C")
        pdf.ln(row_height)
        
        # Save the PDF (for the client, full details including prices)
        pdf.output(file_name)
        print(f"Saved Invoice to {file_name} with Final Total: {final_total}.")
        
        # Send the PDF via Selenium
        try:
            attach_button = driver.find_element(By.XPATH, "//button[@title='Attach']")
            attach_button.click()
            file_input = driver.find_element(By.XPATH, "//input[@type='file']")
            file_input.send_keys(os.path.abspath(file_name))
            time.sleep(1)  # Allow time for the file upload
            send_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[@data-icon='send']"))
            )
            send_button.click()
            print(f"Sent PDF: {os.path.abspath(file_name)}")
        except NoSuchElementException as e:
            print(f"Could not send PDF: {str(e)}")

    def process_text(self,text):
        # Escape potential bullet markers or special chars:
        text = text.replace('*', '')
        text = text.replace('-', '')
        text = text.replace('1.', '')
        text = text.replace('2.', '')
        text = text.replace('3.', '')
        text = text.replace('4.', '')
        text = text.replace('5.', '')
        text = text.replace('6.', '')
        text = text.replace('7.', '')
        text = text.strip()
        if "\n" in text:
            return text.split("\n")  # Return list if multiline
        return text  # Return string if single line

    def run_browser(self):
        chrome_options = webdriver.ChromeOptions()
        service = Service('C:\\Users\\umarf\\Downloads\\chromedriver-win64\\chromedriver.exe')
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-browser-side-navigation")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--user-data-dir=C:\\Users\\umarf\\AppData\\Local\\Google\\Chrome\\User_Data_2\\") #e.g. C:\Users\You\AppData\Local\Google\Chrome\User Data
        chrome_options.add_argument('--profile-directory=Profile 2') #e.g. Profile 3

        print("Initializing Chrome Driver!")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.maximize_window()
        wait = WebDriverWait(driver, 10)
        actions = ActionChains(driver)

        try:
            driver.get("https://web.whatsapp.com/")
            print("Waiting for WhatsApp Web to load...")
            time.sleep(15)
            print("Page loaded successfully.")

            # Take screenshot of QR code
            driver.save_screenshot("whatsapp_qr.png")
            print("QR code saved as 'whatsapp_qr.png'. Please scan it with your WhatsApp app.")

            # Wait for user to scan QR code
            input("Press Enter after scanning the QR code...")    

        except (WebDriverException, NoSuchElementException) as e:
            print(f"An error occurred: {str(e)}")
        finally:
            try:
                driver.quit()
                print("Driver quit successfully.")
            except WebDriverException as e:
                print(f"Error during driver quit: {str(e)}")

if __name__ == "__main__":
    chatbot = EnhancedChatbot()
    chatbot.run_browser()

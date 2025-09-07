
import os
import json
import time
import csv
import pandas as pd
import requests

def fetch_data():
    
    url = "http://localhost:5555/vericore_information"

    payload = {
        "token_name": "hiccup_token",
        "token_value": "hiccup_1234567890",
        "page_size": 8000
    }

    response = requests.post(url, json=payload)
    
    statement_array1 = []
    excerpt_array1 = []
    relatetype_array1 = [] # 0 - support, 1 - unrelated
    
    statement_array2 = []
    excerpt_array2 = []
    relatetype_array2 = [] # 0 - support, 1 - unrelated
    
    if response.status_code == 200:
        data = response.json()
        print("Fetched Data------------")
        
        for item in data:
            statement = item.get("statement")
            results = item.get("results", [])
            
            for result in results:
                responses = result.get("vericore_responses", [])
                
                for response in responses:
                    excerpt = response['excerpt']
                    snippet_score_reason = response['snippet_score_reason']  # "unrelated_page_snippet"
                    local_score = response['local_score']
                    
                    # print("----------------------")
                    # print(excerpt)
                    # print(snippet_score_reason)
                    # print(local_score)
                    
                    if snippet_score_reason == "unrelated_page_snippet":
                        statement_array2.append(statement)
                        excerpt_array2.append(excerpt)
                        relatetype_array2.append(1)
                    elif local_score > 0:
                        statement_array1.append(statement)
                        excerpt_array1.append(excerpt)
                        relatetype_array1.append(0)
                        pass
            
        print(f"Unrelate Count: {len(statement_array2)}")
        print(f"Support Count: {len(statement_array1)}")
        len_statement = len(statement_array2) if len(statement_array2) < len(statement_array1) else len(statement_array1) 
         
        statement_array = []
        excerpt_array = []
        relatetype_array = []
        
        statement_array.extend(statement_array1[0:len_statement])
        statement_array.extend(statement_array2[0:len_statement])
        
        excerpt_array.extend(excerpt_array1[0:len_statement])
        excerpt_array.extend(excerpt_array2[0:len_statement])
        
        relatetype_array.extend(relatetype_array1[0:len_statement])
        relatetype_array.extend(relatetype_array2[0:len_statement])
                    
        data = {
            "statement": statement_array,
            "excerpt": excerpt_array,
            "relate": relatetype_array
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv("dataset.csv", index=False)
            
        print("Finished")
        print(len(statement_array))
        print(len(excerpt_array))
        print(len(relatetype_array))
        
    else:
        print(f"❌ Failed: {response.status_code}")

fetch_data()
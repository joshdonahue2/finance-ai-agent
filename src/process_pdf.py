#!/usr/bin/env python3
"""
PDF Processing Pipeline: OCR → Extract → Embed → Store
Supports batch mode for initial 7-year data load
"""

import os
import re
import glob
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from pdf2image import convert_from_path

from PIL import Image
from supabase import create_client, Client
import ollama
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import tempfile
from mistralai.client import MistralClient
from mistralai.models.files import FileObject
from mistralai.models.jobs import Job

class PDFProcessor:
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        
    def detect_bank_from_filename(self, filename: str) -> str:
        """Simple bank detection from filename patterns"""
        filename_lower = filename.lower()
        if 'usaa' in filename_lower:
            return 'USAA'
        elif 'chase' in filename_lower:
            return 'Chase'
        else:
            # For AI extraction, we can be more generic
            return os.path.splitext(filename)[0].capitalize()

    def extract_transactions_with_mistral_api(self, pdf_path: str) -> pd.DataFrame:
        """
        Extracts transactions from a PDF using the Mistral API.
        """
        import json
        import time

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file")

        client = MistralClient(api_key=api_key)

        try:
            with open(pdf_path, "rb") as f:
                uploaded_file: FileObject = client.files.create(
                    file=(os.path.basename(pdf_path), f.read()),
                    purpose="ocr"
                )
            
            job: Job = client.jobs.create(
                model="mistral-ocr-latest",
                tasks=[{"tool": "ocr", "document": {"type": "file_id", "file_id": uploaded_file.id}}]
            )

            job = client.jobs.retrieve(job.id)
            while job.status not in ["COMPLETED", "FAILED"]:
                time.sleep(1)
                job = client.jobs.retrieve(job.id)

            if job.status == "FAILED":
                logger.error(f"Mistral API job failed for {pdf_path}")
                return pd.DataFrame()

            json_response = job.result.outputs[0].markdown
            
            json_start = json_response.find('[')
            json_end = json_response.rfind(']') + 1
            json_str = json_response[json_start:json_end]

            if json_str:
                transactions = json.loads(json_str)
                df = pd.DataFrame(transactions)
                df['transaction_date'] = pd.to_datetime(df['date'])
                df['amount'] = pd.to_numeric(df['amount'])
                logger.info(f"✅ Extracted {len(df)} transactions using Mistral API from {pdf_path}")
                return df

        except Exception as e:
            logger.error(f"Mistral API processing failed for {pdf_path}: {e}")

        logger.warning(f"No transactions extracted from {pdf_path} using Mistral API.")
        return pd.DataFrame()

    def extract_transactions_with_ai(self, pdf_path: str, ocr_model: str) -> pd.DataFrame:
        """
        Extracts transactions from a PDF using a multimodal AI model.
        This approach is bank-agnostic.
        """
        if ocr_model == "Mistral API":
            return self.extract_transactions_with_mistral_api(pdf_path)

        import base64
        import json
        from io import BytesIO

        try:
            images = convert_from_path(pdf_path, dpi=200)  # Use 200 DPI for faster processing
        except Exception as e:
            logger.error(f"PDF to image conversion failed for {pdf_path}: {e}")
            return pd.DataFrame()

        all_transactions = []

        # Determine host and model name
        if "::" in ocr_model:
            instance, model_name = ocr_model.split("::")
            if instance == "ocr":
                host = os.getenv("OLLAMA_HOST_OCR")
            else: # Should not happen based on UI filtering
                host = os.getenv("OLLAMA_HOST_CHAT")
        else: # No prefix, use default host
            model_name = ocr_model
            host = os.getenv("OLLAMA_HOST")

        ollama_client = ollama.Client(host=host)

        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with {model_name} on host {host}...")
            
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            prompt = """
            You are an expert financial analyst. Analyze the provided bank statement image and extract all transactions.
            Return the transactions as a valid JSON array. Each object in the array should have the following keys:
            "date" (in YYYY-MM-DD format), "description" (a string), and "amount" (a float, negative for debits/withdrawals, positive for credits/deposits).
            If you cannot determine the year, assume the current year. Ignore summary sections.
            """

            try:
                response = ollama_client.generate(
                    model=model_name,
                    prompt=prompt,
                    images=[img_base64],
                    stream=False
                )
                
                # Extract the JSON part of the response
                json_response = response['response']
                json_start = json_response.find('[')
                json_end = json_response.rfind(']') + 1
                json_str = json_response[json_start:json_end]

                if json_str:
                    transactions = json.loads(json_str)
                    all_transactions.extend(transactions)
                    logger.info(f"Extracted {len(transactions)} transactions from page {i+1}")

            except Exception as e:
                logger.error(f"AI processing failed for page {i+1} of {pdf_path}: {e}")
                continue

        if not all_transactions:
            logger.warning(f"No transactions extracted from {pdf_path} using AI.")
            return pd.DataFrame()

        df = pd.DataFrame(all_transactions)
        df['transaction_date'] = pd.to_datetime(df['date'])
        df['amount'] = pd.to_numeric(df['amount'])
        
        logger.info(f"✅ Extracted {len(df)} transactions using AI from {pdf_path}")
        return df

    def create_transaction_chunk(self, row: pd.Series) -> str:
        """Create embeddable text chunk for a single transaction"""
        return (
            f"Date: {row['transaction_date'].strftime('%Y-%m-%d')}, "
            f"Description: {row['description']}, "
            f"Amount: ${row['amount']:.2f}, "
            f"Bank: {row['bank_name']}, "
            f"PDF: {row['pdf_filename']}"
        )
    
    def embed_text(self, text: str, chat_model: str) -> List[float]:
        """Generate embedding using Ollama"""
        # Determine host and model name
        if "::" in chat_model:
            instance, model_name = chat_model.split("::")
            if instance == "chat":
                host = os.getenv("OLLAMA_HOST_CHAT")
            else: # Should not happen based on UI filtering
                host = os.getenv("OLLAMA_HOST_OCR")
        else: # No prefix, use default host
            model_name = chat_model
            host = os.getenv("OLLAMA_HOST")

        ollama_client = ollama.Client(host=host)

        try:
            response = ollama_client.embeddings(
                model='nomic-embed-text', # embedding model is not selectable for now
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return [0.0] * 384  # Return zero vector on error
    
    def store_transaction(self, row: pd.Series, chat_model: str) -> bool:
        """Store a single transaction in Supabase"""
        try:
            chunk_text = self.create_transaction_chunk(row)
            embedding = self.embed_text(chunk_text, chat_model=chat_model)
            
            data = {
                'pdf_filename': row['pdf_filename'],
                'bank_name': row['bank_name'],
                'transaction_date': row['transaction_date'].isoformat(),
                'description': str(row['description']),
                'amount': float(row['amount']),
                'full_chunk': chunk_text,
                'embedding': embedding
            }
            
            result = self.client.table('transactions').insert(data).execute()
            if result.data:
                logger.info(f"Stored transaction: {row['transaction_date']} - {row['description'][:30]}")
                return True
            else:
                logger.error(f"Failed to insert: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Store failed for {row['transaction_date']}: {e}")
            return False
    
    def process_single_pdf(self, pdf_path: str, ocr_model: str, chat_model: str) -> int:
        """Process a single PDF and return number of transactions stored"""
        logger.info(f"Processing {pdf_path} with AI...")
        
        bank_name = self.detect_bank_from_filename(os.path.basename(pdf_path))
        pdf_filename = os.path.basename(pdf_path)
        
        # Extract transactions using AI
        df = self.extract_transactions_with_ai(pdf_path, ocr_model=ocr_model)
        if df.empty:
            logger.warning(f"No transactions extracted from {pdf_path} with AI.")
            return 0
        
        # Add metadata
        df['bank_name'] = bank_name
        df['pdf_filename'] = pdf_filename
        
        # Store each transaction
        successful = 0
        for _, row in df.iterrows():
            if self.store_transaction(row, chat_model=chat_model):
                successful += 1
        
        logger.info(f"Processed {pdf_path}: {successful}/{len(df)} transactions stored")
        return successful

    def process_batch(self, folder_path: str, ocr_model: str, chat_model: str) -> int:
        """Process all PDFs in a folder (for initial data load)"""
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return 0
        
        total_processed = 0
        for pdf_path in pdf_files:
            try:
                count = self.process_single_pdf(pdf_path, ocr_model=ocr_model, chat_model=chat_model)
                total_processed += count
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
        
        logger.info(f"Batch complete: {total_processed} total transactions processed")
        return total_processed

def main():
    parser = argparse.ArgumentParser(description="Process bank statement PDFs")
    parser.add_argument('--pdf', type=str, help="Single PDF path")
    parser.add_argument('--batch', type=str, help="Folder containing multiple PDFs")
    parser.add_argument('--ocr_model', type=str, default="llava", help="Ollama model to use for OCR")
    parser.add_argument('--chat_model', type=str, default="llama3", help="Ollama model to use for chat")
    args = parser.parse_args()
    
    if not args.pdf and not args.batch:
        parser.error("Must specify --pdf or --batch")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    
    supabase_client = create_client(supabase_url, supabase_key)
    processor = PDFProcessor(supabase_client)
    
    if args.pdf:
        processor.process_single_pdf(args.pdf, ocr_model=args.ocr_model, chat_model=args.chat_model)
    elif args.batch:
        processor.process_batch(args.batch, ocr_model=args.ocr_model, chat_model=args.chat_model)
if __name__ == "__main__":
    main()
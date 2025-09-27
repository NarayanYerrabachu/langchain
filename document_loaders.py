import logging
import tempfile
import uuid
from io import BytesIO
from typing import List, Dict, Any
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)

from models import DocumentType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle processing of different document types"""

    def __init__(self):
        self.supported_types = {
            DocumentType.TEXT: self._process_text,
            DocumentType.PDF: self._process_pdf,
            DocumentType.DOCX: self._process_docx,
            DocumentType.HTML: self._process_html,
            DocumentType.WEB_URL: self._process_web_url,
            DocumentType.MARKDOWN: self._process_markdown,
            DocumentType.CSV: self._process_csv,
            DocumentType.EXCEL: self._process_excel,
        }

    def process_document(
            self,
            content: str = None,
            file_content: bytes = None,
            url: str = None,
            document_type: DocumentType = DocumentType.TEXT,
            metadata: Dict[str, Any] = None,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ) -> List[Document]:
        """Process document based on type and return chunks"""

        if metadata is None:
            metadata = {}

        # Add document ID for tracking
        doc_id = str(uuid.uuid4())
        metadata["document_id"] = doc_id
        metadata["document_type"] = document_type.value

        try:
            processor = self.supported_types.get(document_type)
            if not processor:
                raise ValueError(f"Unsupported document type: {document_type}")

            # Process document to get raw text
            raw_documents = processor(content, file_content, url, metadata)

            # Split into chunks
            text_splitter = self._get_text_splitter(document_type, chunk_size, chunk_overlap)
            chunks = text_splitter.split_documents(raw_documents)

            # Add chunk information to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })

            logger.info(f"Processed {document_type.value} document into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing {document_type.value} document: {str(e)}")
            raise

    def _process_text(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process plain text content"""
        if content:
            text_content = content
        elif file_content:
            text_content = file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError("No text content provided")

        return [Document(page_content=text_content, metadata=metadata)]

    def _process_pdf(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process PDF files"""
        if not file_content:
            raise ValueError("No file content provided for PDF processing")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()

            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            # Update metadata for all pages
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    **metadata,
                    "page": i + 1,
                    "total_pages": len(documents)
                })

            return documents

    def _process_docx(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process DOCX files"""
        if not file_content:
            raise ValueError("No file content provided for DOCX processing")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()

            loader = Docx2txtLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(metadata)

            return documents

    def _process_html(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process HTML content"""
        if content:
            html_content = content
        elif file_content:
            html_content = file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError("No HTML content provided")

        # Parse with BeautifulSoup for better text extraction
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)

        # Extract additional metadata from HTML
        title = soup.find('title')
        if title:
            metadata["title"] = title.get_text().strip()

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata["description"] = meta_desc.get('content', '')

        return [Document(page_content=clean_text, metadata=metadata)]

    def _process_web_url(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process web URL by scraping content"""
        if not url:
            raise ValueError("No URL provided for web scraping")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            metadata["source_url"] = url
            metadata["status_code"] = response.status_code

            # Use HTML processing for the scraped content
            return self._process_html(response.text, None, None, metadata)

        except requests.RequestException as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            raise ValueError(f"Failed to scrape URL: {str(e)}")

    def _process_markdown(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process Markdown files"""
        if content:
            md_content = content
        elif file_content:
            md_content = file_content.decode('utf-8', errors='ignore')
        else:
            raise ValueError("No Markdown content provided")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.md', mode='w', encoding='utf-8') as tmp_file:
            tmp_file.write(md_content)
            tmp_file.flush()

            loader = UnstructuredMarkdownLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(metadata)

            return documents

    def _process_csv(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process CSV files"""
        if not file_content:
            raise ValueError("No file content provided for CSV processing")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()

            loader = CSVLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(metadata)

            return documents

    def _process_excel(self, content: str, file_content: bytes, url: str, metadata: Dict) -> List[Document]:
        """Process Excel files"""
        if not file_content:
            raise ValueError("No file content provided for Excel processing")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()

            loader = UnstructuredExcelLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(metadata)

            return documents

    def _get_text_splitter(self, document_type: DocumentType, chunk_size: int, chunk_overlap: int):
        """Get appropriate text splitter based on document type"""

        if document_type == DocumentType.MARKDOWN:
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif document_type in [DocumentType.PDF, DocumentType.DOCX, DocumentType.HTML, DocumentType.WEB_URL]:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n"
            )
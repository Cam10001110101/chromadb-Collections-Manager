import sys
import os
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem, 
                           QLineEdit, QLabel, QMessageBox, QTextEdit, QTableWidget, 
                           QTableWidgetItem, QSplitter, QSpinBox, QStyleFactory, 
                           QHeaderView, QComboBox, QFileDialog, QProgressBar,
                           QDoubleSpinBox, QDialog, QFormLayout, QListWidget,
                           QListWidgetItem, QCompleter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSortFilterProxyModel
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json
from gui_styles import GUIStyles
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
import chromadb
import logging
from config import OLLAMA_MODELS, EMBEDDING_MODELS, OLLAMA_URL

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setPlaceholderText("Type to filter addresses...")
        
        # Set up model and proxy model for filtering
        self.model = QStandardItemModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        # Connect text changed signal to filter
        self.lineEdit().textChanged.connect(self.filter_items)
        
        self._checked_items = set()
        self._all_items = set()
        
        # Create completer for auto-completion
        self.completer = QCompleter()
        self.completer.setModel(self.proxy_model)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompleter(self.completer)
        
        # Show dropdown on click
        self.lineEdit().mousePressEvent = self.show_popup

    def show_popup(self, event):
        super().showPopup()
        
    def filter_items(self, text):
        self.proxy_model.setFilterFixedString(text)
        if self.view().isVisible():
            self.showPopup()

    def handle_item_pressed(self, index):
        source_index = self.proxy_model.mapToSource(index)
        item = self.model.itemFromIndex(source_index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
            self._checked_items.discard(item.text())
        else:
            item.setCheckState(Qt.Checked)
            self._checked_items.add(item.text())
        
        # Update display text
        if not self._checked_items:
            self.lineEdit().setPlaceholderText("Type to filter addresses...")
            self.lineEdit().clear()
        else:
            self.setEditText(f"{len(self._checked_items)} addresses selected")

    def add_items(self, items):
        self.model.clear()
        self._checked_items.clear()
        self._all_items = set(items)
        self.lineEdit().setPlaceholderText("Type to filter addresses...")
        self.lineEdit().clear()
        
        for text in sorted(items):
            item = QStandardItem(text)
            item.setCheckState(Qt.Unchecked)
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            self.model.appendRow(item)
        
        # Connect view clicked signal after adding items
        self.view().clicked.connect(self.handle_item_pressed)

    def get_checked_items(self):
        return list(self._checked_items)

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logging.error(f"Error in worker thread: {str(e)}")
            self.error.emit(str(e))

class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        self.api_url = f"{OLLAMA_URL}/api/embeddings"
        self.model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
        
    def __call__(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
            
        all_embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model, "prompt": text}
                )
                response.raise_for_status()
                embedding = response.json().get('embedding')
                if embedding is None:
                    raise ValueError("No embedding returned from Ollama API")
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding: {str(e)}")
                return None
                
        return all_embeddings

class ModernWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        GUIStyles.apply_dark_theme(self)

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Configuration')
        self.setModal(True)
        
        layout = QFormLayout()
        
        # ChromaDB Path
        self.chroma_path = QLineEdit(os.getenv('CHROMA_DB_PATH', ''))
        layout.addRow('ChromaDB Path:', self.chroma_path)
        
        # ChromaDB Host
        self.chroma_host = QLineEdit(os.getenv('CHROMADB_HOST', 'localhost'))
        layout.addRow('ChromaDB Host:', self.chroma_host)
        
        # ChromaDB Port
        self.chroma_port = QSpinBox()
        self.chroma_port.setRange(1, 65535)
        self.chroma_port.setValue(int(os.getenv('CHROMADB_PORT', '8000')))
        layout.addRow('ChromaDB Port:', self.chroma_port)
        
        # Ollama URL
        self.ollama_url = QLineEdit(OLLAMA_URL)
        layout.addRow('Ollama URL:', self.ollama_url)
        
        # Embedding Model
        self.embedding_model = QComboBox()
        self.embedding_model.addItems(EMBEDDING_MODELS)
        default_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
        index = self.embedding_model.findText(default_model)
        if index >= 0:
            self.embedding_model.setCurrentIndex(index)
        layout.addRow('Embedding Model:', self.embedding_model)
        
        # Buttons
        button_box = QHBoxLayout()
        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        button_box.addWidget(save_btn)
        button_box.addWidget(cancel_btn)
        layout.addRow(button_box)
        
        self.setLayout(layout)
    
    def get_config(self):
        return {
            'CHROMA_DB_PATH': self.chroma_path.text(),
            'CHROMADB_HOST': self.chroma_host.text(),
            'CHROMADB_PORT': self.chroma_port.value(),
            'OLLAMA_URL': self.ollama_url.text(),
            'EMBEDDING_MODEL': self.embedding_model.currentText()
        }

class ChromaDBManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ChromaDB Management Tool')
        self.setGeometry(100, 100, 1200, 800)
        
        # Load configuration
        self.load_config()
        
        try:
            if not self.chroma_db_path:
                raise ValueError("CHROMA_DB_PATH environment variable is not set")
            
            # Initialize ChromaDB with Ollama embedding function
            self.embedding_function = OllamaEmbeddingFunction()
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize LangChain components
            self.embedding_model = OllamaEmbeddings(
                base_url=self.ollama_url,
                model=self.embedding_model_name
            )
            self.llm = ChatOllama(
                base_url=self.ollama_url,
                model=OLLAMA_MODELS[0],  # Default to first model
                temperature=0.0
            )
            
            self.init_ui()
        except Exception as e:
            self.show_connection_error("ChromaDB", str(e))

    def load_config(self):
        self.chroma_db_path = os.getenv('CHROMA_DB_PATH')
        self.chroma_host = os.getenv('CHROMADB_HOST', 'localhost')
        self.chroma_port = int(os.getenv('CHROMADB_PORT', '8000'))
        self.ollama_url = OLLAMA_URL
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')

    def save_config(self, config):
        self.chroma_db_path = config['CHROMA_DB_PATH']
        self.chroma_host = config['CHROMADB_HOST']
        self.chroma_port = config['CHROMADB_PORT']
        self.ollama_url = config['OLLAMA_URL']
        self.embedding_model_name = config['EMBEDDING_MODEL']
        
        # Update environment variables
        os.environ['CHROMA_DB_PATH'] = self.chroma_db_path
        os.environ['CHROMADB_HOST'] = self.chroma_host
        os.environ['CHROMADB_PORT'] = str(self.chroma_port)
        os.environ['OLLAMA_URL'] = self.ollama_url
        os.environ['EMBEDDING_MODEL'] = self.embedding_model_name

    def show_config_dialog(self):
        dialog = ConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            self.save_config(config)
            QMessageBox.information(self, 'Success', 'Configuration saved. Please restart the application for changes to take effect.')

    def show_connection_error(self, db_name, error_message):
        layout = QVBoxLayout()
        error_label = QLabel(f"Unable to connect to {db_name}. Error: {error_message}")
        layout.addWidget(error_label)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_ui(self):
        central_widget = ModernWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Top controls
        top_controls = QHBoxLayout()
        
        # Collection management
        self.new_collection_input = QLineEdit()
        self.new_collection_input.setPlaceholderText('New Collection Name')
        self.create_collection_btn = QPushButton('Create Collection')
        self.create_collection_btn.clicked.connect(self.create_collection)
        
        # Refresh and delete buttons
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh)
        self.delete_button = QPushButton('Delete Collection')
        self.delete_button.clicked.connect(self.delete_collection)
        
        # Add document controls
        self.import_docs_btn = QPushButton('Import Documents')
        self.import_docs_btn.clicked.connect(self.import_documents)
        
        # Configuration button
        self.config_btn = QPushButton('Settings')
        self.config_btn.clicked.connect(self.show_config_dialog)
        
        # Add buttons to top controls
        top_controls.addWidget(self.new_collection_input)
        top_controls.addWidget(self.create_collection_btn)
        top_controls.addWidget(self.refresh_button)
        top_controls.addWidget(self.delete_button)
        top_controls.addWidget(self.import_docs_btn)
        top_controls.addWidget(self.config_btn)
        
        # Add separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #666;")
        top_controls.addWidget(separator)
        
        # LLM Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("LLM Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(OLLAMA_MODELS)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        top_controls.addLayout(model_layout)
        
        # Temperature control
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 1.0)
        self.temp_spinbox.setSingleStep(0.1)
        self.temp_spinbox.setValue(0.0)  # Default temperature
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_spinbox)
        top_controls.addLayout(temp_layout)
        
        # Add separator
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #666;")
        top_controls.addWidget(separator2)
        
        # Display limit control
        self.limit_spinbox = QSpinBox()
        self.limit_spinbox.setRange(1, 1000)
        self.limit_spinbox.setValue(20)
        self.limit_spinbox.setPrefix("Display Limit: ")
        top_controls.addWidget(self.limit_spinbox)
        
        layout.addLayout(top_controls)

        # Main content area
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Collections tree and stats
        left_panel = QSplitter(Qt.Vertical)
        
        self.collections_tree = QTreeWidget()
        self.collections_tree.setHeaderLabels(['Collections'])
        self.collections_tree.itemSelectionChanged.connect(self.on_collection_selected)
        
        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        
        left_panel.addWidget(self.collections_tree)
        left_panel.addWidget(self.stats_view)
        
        # Right panel - Search and results
        right_panel = QSplitter(Qt.Vertical)
        
        # Search controls
        search_layout = QVBoxLayout()
        
        # Query input
        query_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Enter search query...')
        self.search_btn = QPushButton('Vector Search')
        self.search_btn.clicked.connect(self.vector_search)
        query_layout.addWidget(self.search_input)
        query_layout.addWidget(self.search_btn)
        search_layout.addLayout(query_layout)
        
        # From address filter
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter by From:")
        self.from_filter = CheckableComboBox()
        self.from_filter.setMinimumWidth(200)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.from_filter)
        filter_layout.addStretch()
        search_layout.addLayout(filter_layout)
        
        search_widget = QWidget()
        search_widget.setLayout(search_layout)
        right_panel.addWidget(search_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.hide()
        right_panel.addWidget(self.progress_bar)
        
        # Results views
        results_splitter = QSplitter(Qt.Vertical)
        
        # Documents table
        self.docs_table = QTableWidget()
        self.docs_table.setSortingEnabled(True)  # Enable sorting
        results_splitter.addWidget(self.docs_table)
        
        # Formatted results
        self.formatted_results_view = QTextEdit()
        self.formatted_results_view.setReadOnly(True)
        results_splitter.addWidget(self.formatted_results_view)
        
        # LLM response
        self.llm_response_view = QTextEdit()
        self.llm_response_view.setReadOnly(True)
        results_splitter.addWidget(self.llm_response_view)
        
        right_panel.addWidget(results_splitter)
        
        # Add panels to main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        main_splitter.setSizes([300, 900])  # 300px for left panel, rest for right panel
        
        layout.addWidget(main_splitter)
        
        self.refresh()

    def create_collection(self):
        name = self.new_collection_input.text().strip()
        if name:
            try:
                self.client.create_collection(
                    name=name,
                    embedding_function=self.embedding_function
                )
                self.refresh()
                self.new_collection_input.clear()
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to create collection: {str(e)}')
        else:
            QMessageBox.warning(self, 'Error', 'Please enter a collection name')

    def refresh(self):
        self.collections_tree.clear()
        try:
            collections = self.client.list_collections()
            for collection in collections:
                item = QTreeWidgetItem(self.collections_tree)
                item.setText(0, collection.name)
                count = collection.count()
                item.setToolTip(0, f'Documents: {count}')
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to list collections: {str(e)}')

    def delete_collection(self):
        item = self.collections_tree.currentItem()
        if item:
            collection_name = item.text(0)
            reply = QMessageBox.question(self, 'Confirm Delete',
                                       f'Are you sure you want to delete collection "{collection_name}"?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    self.client.delete_collection(collection_name)
                    self.refresh()
                except Exception as e:
                    QMessageBox.warning(self, 'Error', f'Failed to delete collection: {str(e)}')

    def on_collection_selected(self):
        item = self.collections_tree.currentItem()
        if item:
            self.show_collection_details(item.text(0))

    def show_collection_details(self, collection_name):
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Update stats view
            count = collection.count()
            stats = f"Collection: {collection_name}\n"
            stats += f"Document count: {count}\n"
            stats += f"Metadata: {json.dumps(collection.metadata, indent=2) if collection.metadata else 'None'}\n"
            stats += f"\nEmbedding Function: {self.embedding_model_name} (Ollama)"
            
            self.stats_view.setText(stats)
            
            # Clear previous results
            self.docs_table.setRowCount(0)
            self.formatted_results_view.clear()
            self.llm_response_view.clear()
            
            # Update From address filter
            if count > 0:
                # Get all documents to extract From addresses
                docs = collection.get(limit=count)
                from_addresses = set()
                for metadata in docs['metadatas']:
                    if metadata and 'From' in metadata:
                        from_addresses.add(metadata['From'])
                self.from_filter.add_items(from_addresses)
            else:
                self.from_filter.clear()
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'Failed to get collection details: {str(e)}')

    def vector_search(self):
        item = self.collections_tree.currentItem()
        if not item:
            QMessageBox.warning(self, 'Error', 'Please select a collection first')
            return
            
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, 'Error', 'Please enter a search query')
            return
        
        try:
            self.progress_bar.show()
            self.search_btn.setEnabled(False)
            
            # Clear previous results
            self.docs_table.setRowCount(0)
            self.formatted_results_view.clear()
            self.llm_response_view.clear()
            
            # Start search in worker thread
            self.worker = WorkerThread(self.process_query, item.text(0), query)
            self.worker.finished.connect(self.on_search_finished)
            self.worker.error.connect(self.on_search_error)
            self.worker.start()
            
        except Exception as e:
            self.progress_bar.hide()
            self.search_btn.setEnabled(True)
            QMessageBox.warning(self, 'Error', f'Search failed: {str(e)}')

    def process_query(self, collection_name, query):
        # Initialize LangChain components for the collection
        langchain_chroma = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_model
        )
        
        # Create LLM with current settings
        current_llm = ChatOllama(
            base_url=self.ollama_url,
            model=self.model_combo.currentText(),
            temperature=self.temp_spinbox.value()
        )
        
        # Create MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=langchain_chroma.as_retriever(search_kwargs={"k": self.limit_spinbox.value()}),
            llm=current_llm
        )
        
        # Get documents
        docs = retriever.invoke(query)
        
        # Generate combined output for LLM context
        combined_output = "\n---\n".join([f"Document {idx+1}:\n{doc.page_content}" for idx, doc in enumerate(docs)])
        
        # Generate LLM response
        prompt = ChatPromptTemplate.from_template("""Based on the following context, answer this question: {question}

Context: {context}

If no relevant information is found in the context, please respond with:
"I'm sorry, but I couldn't find any relevant information to answer your question. The search didn't return any results that match your query. You may want to try rephrasing your question or searching for more general terms."

Otherwise, provide a concise and informative answer based on the available information.""")
        
        chain = prompt | current_llm | StrOutputParser()
        answer = chain.invoke({"question": query, "context": combined_output if docs else "No relevant documents found."})
        
        return docs, answer

    def on_search_finished(self, result):
        docs, answer = result
        self.progress_bar.hide()
        self.search_btn.setEnabled(True)
        self.display_search_results(docs, answer)

    def on_search_error(self, error):
        self.progress_bar.hide()
        self.search_btn.setEnabled(True)
        QMessageBox.warning(self, 'Error', f'Search failed: {error}')

    def display_search_results(self, docs, answer):
        if not docs:
            self.docs_table.setRowCount(0)
            self.formatted_results_view.setText("No results found.")
            self.llm_response_view.setText(answer)
            return

        # Get selected From addresses
        selected_from = self.from_filter.get_checked_items()
        
        # Filter documents if From addresses are selected
        if selected_from:
            docs = [doc for doc in docs if doc.metadata.get('From') in selected_from]
            if not docs:
                self.docs_table.setRowCount(0)
                self.formatted_results_view.setText("No results found matching the selected From addresses.")
                self.llm_response_view.setText(answer)
                return

        # Determine all unique metadata fields across all documents
        metadata_fields = set()
        for doc in docs:
            if doc.metadata:
                metadata_fields.update(doc.metadata.keys())
        metadata_fields = sorted(list(metadata_fields))  # Sort for consistent column order

        # Set up table columns: metadata fields + content
        total_columns = len(metadata_fields) + 1  # +1 for content
        self.docs_table.setColumnCount(total_columns)
        
        # Set headers
        headers = metadata_fields + ['Content']
        self.docs_table.setHorizontalHeaderLabels(headers)
        
        # Enable sorting
        self.docs_table.setSortingEnabled(True)
        
        # Update table
        self.docs_table.setRowCount(len(docs))
        for row, doc in enumerate(docs):
            # Add metadata columns
            for col, field in enumerate(metadata_fields):
                value = doc.metadata.get(field, "N/A")
                item = QTableWidgetItem(str(value))
                self.docs_table.setItem(row, col, item)
            
            # Add content column
            content_item = QTableWidgetItem(str(doc.page_content))
            self.docs_table.setItem(row, len(metadata_fields), content_item)

        # Adjust column widths
        header = self.docs_table.horizontalHeader()
        for col in range(total_columns - 1):  # All except content column
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(total_columns - 1, QHeaderView.Stretch)  # Content column stretches

        # Update formatted view
        formatted_results_text = ""
        for idx, doc in enumerate(docs, 1):
            formatted_results_text += f"Document {idx}:\n"
            metadata = doc.metadata
            for key, value in metadata.items():
                formatted_results_text += f"<b>{key}:</b> {value}<br>"
            
            formatted_results_text += "<br><b>Email Content:</b><br>"
            formatted_results_text += f"<b>From:</b> {metadata.get('From', 'N/A')}<br>"
            formatted_results_text += f"<b>Sent:</b> {metadata.get('Sent', 'N/A')}<br>"
            formatted_results_text += f"<b>To:</b> {metadata.get('To', 'N/A')}<br>"
            formatted_results_text += f"<b>Cc:</b> {metadata.get('CC', 'N/A')}<br>"
            formatted_results_text += f"<b>Subject:</b> {metadata.get('Subject', 'N/A')}<br>"
            formatted_results_text += f"<b>Body:</b><br>{doc.page_content}<br><br>"
            formatted_results_text += "------------------------------<br><br>"

        self.formatted_results_view.setHtml(formatted_results_text)
        self.llm_response_view.setText(f"LLM Response:\n\n{answer}")

    def import_documents(self):
        item = self.collections_tree.currentItem()
        if not item:
            QMessageBox.warning(self, 'Error', 'Please select a collection first')
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document to Import",
            "",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                collection = self.client.get_collection(
                    name=item.text(0),
                    embedding_function=self.embedding_function
                )
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    if file_path.endswith('.json'):
                        data = json.load(file)
                        # Assuming JSON contains an array of objects with 'content' field
                        documents = [doc['content'] for doc in data]
                        ids = [f"doc_{i}" for i in range(len(documents))]
                    else:
                        content = file.read()
                        # Split text file into chunks (simple approach)
                        documents = [content]
                        ids = ["doc_1"]
                
                collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=[{"source": os.path.basename(file_path)} for _ in ids]
                )
                
                self.show_collection_details(item.text(0))
                QMessageBox.information(self, 'Success', 'Documents imported successfully')
                
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to import documents: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    GUIStyles.apply_dark_theme(app)
    
    window = ChromaDBManagerWindow()
    window.show()
    sys.exit(app.exec_())

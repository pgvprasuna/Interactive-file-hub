# Interactive-file-hub
# PDF Q&A Application with Vector Search

This application allows users to upload PDF files and ask questions about their content. It uses OpenAI embeddings and Qdrant vector database for semantic search capabilities.

## Features

- User authentication (register/login)
- PDF file upload and processing
- Text chunking and embedding generation
- Vector similarity search using Qdrant
- Question answering using OpenAI's GPT models
- React-based user interface

## Combined Code

The entire application has been combined into a single file (`combined_code.js`) for easy deployment and sharing. This file includes:

- Express server setup
- MongoDB models and connection
- Qdrant vector database integration
- OpenAI API integration
- Authentication middleware
- File processing utilities
- Text chunking and embedding generation
- API routes for auth, files, and Q&A
- Reference to Python PDF processing code

## Prerequisites

- Node.js (v14+)
- MongoDB (local or Atlas)
- Qdrant vector database (can be run via Docker)
- OpenAI API key

## Environment Variables

Create a `.env` file with the following variables:

```
MONGODB_URI=mongodb://localhost:27017/qa-file-processor
JWT_SECRET=your-secret-key
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333
PORT=5000
```

## Installation

1. Install dependencies:

```bash
npm install express mongoose cors jsonwebtoken bcryptjs @qdrant/js-client-rest openai dotenv
```

2. For PDF processing, install Python dependencies:

```bash
pip install openai python-dotenv PyPDF2 numpy
```

## Running the Application

1. Start MongoDB and Qdrant
2. Run the server:

```bash
node combined_code.js
```

3. For the client-side, you'll need to set up a React application separately or use the original client code from the repository.

## API Endpoints

### Authentication
- POST `/api/auth/register` - Register a new user
- POST `/api/auth/login` - Login a user

### Files
- POST `/api/files/upload` - Upload and process a file
- GET `/api/files` - Get all files for a user
- GET `/api/files/:id` - Get a single file
- DELETE `/api/files/:id` - Delete a file

### Q&A
- POST `/api/qa/ask` - Ask a question about a file
- GET `/api/qa/history/:fileId` - Get question history for a file

## Testing Qdrant Integration

The combined code includes a `QdrantTest` class that can be used to test the Qdrant integration. You can run the tests by creating a separate file that imports the class and calls the `runAllTests` method.

## Original Structure

The original application was structured as follows:

- `client/` - React frontend
- `server/` - Express backend
  - `middleware/` - Authentication middleware
  - `models/` - MongoDB models
  - `routes/` - API routes
  - `utils/` - Utility functions
- `pdf_qa.py` - Python script for PDF processing

## License

MIT

/**
 * Combined Code for PDF Q&A Application
 * This file combines both client and server code for the PDF Q&A application
 * with vector search capabilities using OpenAI embeddings and Qdrant.
 */

// =====================================================================
// CONFIGURATION AND ENVIRONMENT VARIABLES
// =====================================================================

// Load environment variables
require('dotenv').config();

// Environment variables needed:
// MONGODB_URI - MongoDB connection string
// JWT_SECRET - Secret for JWT token generation
// OPENAI_API_KEY - OpenAI API key for embeddings and completions
// QDRANT_URL - URL for Qdrant vector database (default: http://localhost:6333)
// PORT - Port for Express server (default: 5000)

// =====================================================================
// DEPENDENCIES
// =====================================================================

// Server dependencies
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { QdrantClient } = require('@qdrant/js-client-rest');
const { Configuration, OpenAIApi } = require('openai');

// Client dependencies (for reference - not used in Node.js)
// import React from 'react';
// import ReactDOM from 'react-dom/client';
// import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
// import axios from 'axios';
// import { ThemeProvider, CssBaseline, createTheme } from '@mui/material';

// =====================================================================
// OPENAI CONFIGURATION AND UTILITIES
// =====================================================================

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

// Generate embeddings for a given text
async function generateEmbeddings(text) {
  try {
    const response = await openai.createEmbedding({
      model: 'text-embedding-ada-002',
      input: text,
    });
    return response.data.data[0].embedding;
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw new Error('Failed to generate embeddings');
  }
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (normA * normB);
}

// Find most similar text chunks based on embeddings
function findMostSimilarChunks(queryEmbedding, chunks, topK = 3) {
  const similarities = chunks.map(chunk => ({
    text: chunk.text,
    similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));
  
  return similarities
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK);
}

// =====================================================================
// QDRANT CLIENT CONFIGURATION AND UTILITIES
// =====================================================================

// Initialize Qdrant client
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL || 'http://localhost:6333',
});

/**
 * Initialize a collection in Qdrant for storing embeddings
 * @param {string} collectionName - Name of the collection
 * @param {number} vectorSize - Size of the embedding vectors (1536 for OpenAI ada-002)
 * @returns {Promise<Object>} - Result of the operation
 */
async function initializeCollection(collectionName, vectorSize = 1536) {
  try {
    // Check if collection exists
    const collections = await qdrantClient.getCollections();
    const collectionExists = collections.collections.some(c => c.name === collectionName);
    
    if (collectionExists) {
      console.log(`Collection ${collectionName} already exists`);
      return { success: true, message: `Collection ${collectionName} already exists` };
    }
    
    // Create collection with specified parameters
    await qdrantClient.createCollection(collectionName, {
      vectors: {
        size: vectorSize,
        distance: 'Cosine',
      },
      optimizers_config: {
        default_segment_number: 2,
      },
    });
    
    console.log(`Collection ${collectionName} created successfully`);
    return { success: true, message: `Collection ${collectionName} created successfully` };
  } catch (error) {
    console.error('Error initializing Qdrant collection:', error);
    throw new Error(`Failed to initialize Qdrant collection: ${error.message}`);
  }
}

/**
 * Store embeddings in Qdrant
 * @param {string} collectionName - Name of the collection
 * @param {Array} points - Array of points to store
 * @returns {Promise<Object>} - Result of the operation
 */
async function storeEmbeddings(collectionName, points) {
  try {
    const result = await qdrantClient.upsert(collectionName, {
      wait: true,
      points,
    });
    
    console.log(`Stored ${points.length} embeddings in ${collectionName}`);
    return { success: true, message: `Stored ${points.length} embeddings in ${collectionName}` };
  } catch (error) {
    console.error('Error storing embeddings in Qdrant:', error);
    throw new Error(`Failed to store embeddings in Qdrant: ${error.message}`);
  }
}

/**
 * Search for similar vectors in Qdrant
 * @param {string} collectionName - Name of the collection
 * @param {Array} vector - Query vector
 * @param {number} limit - Number of results to return
 * @param {number} scoreThreshold - Minimum similarity score threshold
 * @returns {Promise<Array>} - Array of search results
 */
async function searchSimilarVectors(collectionName, vector, limit = 3, scoreThreshold = 0.7) {
  try {
    const searchResult = await qdrantClient.search(collectionName, {
      vector,
      limit,
      score_threshold: scoreThreshold,
    });
    
    return searchResult;
  } catch (error) {
    console.error('Error searching vectors in Qdrant:', error);
    throw new Error(`Failed to search vectors in Qdrant: ${error.message}`);
  }
}

/**
 * Delete points from Qdrant
 * @param {string} collectionName - Name of the collection
 * @param {Array} pointIds - Array of point IDs to delete
 * @returns {Promise<Object>} - Result of the operation
 */
async function deletePoints(collectionName, pointIds) {
  try {
    const result = await qdrantClient.delete(collectionName, {
      wait: true,
      points: pointIds,
    });
    
    console.log(`Deleted ${pointIds.length} points from ${collectionName}`);
    return { success: true, message: `Deleted ${pointIds.length} points from ${collectionName}` };
  } catch (error) {
    console.error('Error deleting points from Qdrant:', error);
    throw new Error(`Failed to delete points from Qdrant: ${error.message}`);
  }
}

/**
 * Delete a collection from Qdrant
 * @param {string} collectionName - Name of the collection to delete
 * @returns {Promise<Object>} - Result of the operation
 */
async function deleteCollection(collectionName) {
  try {
    await qdrantClient.deleteCollection(collectionName);
    console.log(`Collection ${collectionName} deleted successfully`);
    return { success: true, message: `Collection ${collectionName} deleted successfully` };
  } catch (error) {
    console.error('Error deleting Qdrant collection:', error);
    throw new Error(`Failed to delete Qdrant collection: ${error.message}`);
  }
}

// =====================================================================
// TEXT CHUNKING UTILITY
// =====================================================================

function chunkText(text, maxChunkSize = 1000) {
  // Split text into sentences using common sentence delimiters
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  
  const chunks = [];
  let currentChunk = '';
  let startIndex = 0;

  for (let sentence of sentences) {
    sentence = sentence.trim();
    
    // If adding this sentence would exceed maxChunkSize, save current chunk and start new one
    if (currentChunk.length + sentence.length > maxChunkSize && currentChunk.length > 0) {
      chunks.push({
        text: currentChunk.trim(),
        startIndex: startIndex,
        endIndex: startIndex + currentChunk.length
      });
      currentChunk = sentence;
      startIndex = startIndex + currentChunk.length;
    } else {
      currentChunk += (currentChunk ? ' ' : '') + sentence;
    }
  }

  // Add the last chunk if there's any content left
  if (currentChunk.length > 0) {
    chunks.push({
      text: currentChunk.trim(),
      startIndex: startIndex,
      endIndex: startIndex + currentChunk.length
    });
  }

  return chunks;
}

// =====================================================================
// FILE INDEXING UTILITY
// =====================================================================

/**
 * Process and index a file using OpenAI embeddings
 * @param {Object} fileData - Object containing file information
 * @param {string} fileData.filePath - Path to the file (optional if content is provided)
 * @param {string} fileData.content - File content (optional if filePath is provided)
 * @param {string} fileData.filename - Name of the file
 * @param {string} fileData.fileType - MIME type of the file
 * @param {string} fileData.userId - ID of the user who uploaded the file
 * @returns {Promise<Object>} - The saved file document with embeddings
 */
async function indexFile(fileData) {
  try {
    // Extract file data
    const { filePath, content, filename, fileType, userId } = fileData;
    
    // Get content either from provided content or by reading the file
    let fileContent = content;
    if (!fileContent && filePath) {
      fileContent = fs.readFileSync(filePath, 'utf8');
    }
    
    if (!fileContent) {
      throw new Error('No file content provided');
    }

    // Create text chunks
    const textChunks = chunkText(fileContent);
    console.log(`File chunked into ${textChunks.length} chunks`);
    
    // Generate embeddings for each chunk
    console.log('Generating embeddings...');
    const chunksWithEmbeddings = await Promise.all(
      textChunks.map(async (chunk) => ({
        ...chunk,
        embedding: await generateEmbeddings(chunk.text)
      }))
    );
    console.log('Embeddings generated successfully');
    
    // Store embeddings in Qdrant
    try {
      const collectionName = 'file_embeddings';
      
      // Initialize collection if it doesn't exist
      await initializeCollection(collectionName);
      
      // Prepare points for Qdrant
      const points = chunksWithEmbeddings.map((chunk, index) => ({
        id: `${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${index}`,
        vector: chunk.embedding,
        payload: {
          text: chunk.text,
          fileId: filename,
          startIndex: chunk.startIndex,
          endIndex: chunk.endIndex,
          metadata: { fileType, userId }
        }
      }));
      
      // Store embeddings in Qdrant
      await storeEmbeddings(collectionName, points);
      console.log(`Embeddings stored in Qdrant collection: ${collectionName}`);
    } catch (error) {
      console.error('Error storing embeddings in Qdrant:', error);
      // Continue with MongoDB storage even if Qdrant fails
    }

    // Create new file record
    const file = new File({
      filename,
      content: fileContent,
      fileType: fileType || 'text/plain',
      uploadedBy: userId,
      chunks: chunksWithEmbeddings,
      processed: true
    });

    // Save the file with embeddings
    await file.save();
    console.log(`File ${filename} indexed and saved successfully`);

    return file;
  } catch (error) {
    console.error('Error indexing file:', error);
    throw new Error(`Failed to index file: ${error.message}`);
  }
}

// =====================================================================
// MONGODB MODELS
// =====================================================================

// User Model
const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    minlength: 3
  },
  email: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    lowercase: true
  },
  password: {
    type: String,
    required: true,
    minlength: 6
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  
  try {
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error);
  }
});

// Method to compare password for login
userSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

const User = mongoose.model('User', userSchema);

// File Model
const chunkSchema = new mongoose.Schema({
  text: {
    type: String,
    required: true
  },
  embedding: {
    type: [Number],
    required: true
  },
  startIndex: {
    type: Number,
    required: true
  },
  endIndex: {
    type: Number,
    required: true
  }
});

const fileSchema = new mongoose.Schema({
  filename: {
    type: String,
    required: true,
    trim: true
  },
  content: {
    type: String,
    required: true
  },
  fileType: {
    type: String,
    required: true
  },
  uploadedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  uploadedAt: {
    type: Date,
    default: Date.now
  },
  chunks: [chunkSchema],
  processed: {
    type: Boolean,
    default: false
  }
});

// Add indexes for frequently queried fields
fileSchema.index({ uploadedBy: 1, uploadedAt: -1 });
fileSchema.index({ processed: 1 });
fileSchema.index({ filename: 1 });

const File = mongoose.model('File', fileSchema);

// Question Model
const questionChunkSchema = new mongoose.Schema({
  text: {
    type: String,
    required: true
  },
  embedding: {
    type: [Number],
    required: true
  }
});

const questionSchema = new mongoose.Schema({
  question: {
    type: String,
    required: true,
    trim: true
  },
  questionEmbedding: {
    type: [Number],
    required: true
  },
  answer: {
    type: String,
    required: true,
    trim: true
  },
  fileId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'File',
    required: true
  },
  askedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  relevanceScore: {
    type: Number,
    default: 0
  },
  chunks: [questionChunkSchema]
});

// Index for text search
questionSchema.index({ question: 'text', answer: 'text' });

const Question = mongoose.model('Question', questionSchema);

// =====================================================================
// AUTHENTICATION MIDDLEWARE
// =====================================================================

const auth = async (req, res, next) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({ message: 'No authentication token, access denied' });
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
    req.userId = decoded.userId;
    next();
  } catch (error) {
    res.status(401).json({ message: 'Token is not valid' });
  }
};

// =====================================================================
// EXPRESS ROUTES
// =====================================================================

// Auth Routes
const authRouter = express.Router();

// Register new user
authRouter.post('/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Create new user
    const user = new User({ username, email, password });
    await user.save();

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-secret-key',
      { expiresIn: '24h' }
    );

    res.status(201).json({
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Error creating user', error: error.message });
  }
});

// Login user
authRouter.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check password
    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-secret-key',
      { expiresIn: '24h' }
    );

    res.json({
      token,
      user: {
        id: user._id,
        username: user.username,
        email: user.email
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Error logging in', error: error.message });
  }
});

// File Routes
const fileRouter = express.Router();

// Upload and process a file
fileRouter.post('/upload', auth, async (req, res) => {
  try {
    const { url, filePath } = req.body;
    
    if (!url && !filePath) {
      return res.status(400).json({ message: 'URL or file path is required' });
    }
    
    let fileData;
    
    if (filePath) {
      // Process local file
      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ message: 'File not found' });
      }
      
      const filename = path.basename(filePath);
      const fileType = path.extname(filePath).substring(1);
      
      fileData = {
        filePath,
        filename,
        fileType,
        userId: req.userId
      };
    } else {
      // Process URL
      // Implementation for URL processing would go here
      return res.status(501).json({ message: 'URL processing not implemented yet' });
    }
    
    // Index the file
    const file = await indexFile(fileData);
    
    res.status(201).json({
      message: 'File processed successfully',
      file: {
        id: file._id,
        filename: file.filename,
        fileType: file.fileType,
        uploadedAt: file.uploadedAt
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Error processing file', error: error.message });
  }
});

// Get all files for a user
fileRouter.get('/', auth, async (req, res) => {
  try {
    const files = await File.find({ uploadedBy: req.userId })
      .select('_id filename fileType uploadedAt processed')
      .sort({ uploadedAt: -1 });
    
    res.json(files);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching files', error: error.message });
  }
});

// Get a single file
fileRouter.get('/:id', auth, async (req, res) => {
  try {
    const file = await File.findOne({
      _id: req.params.id,
      uploadedBy: req.userId
    });
    
    if (!file) {
      return res.status(404).json({ message: 'File not found' });
    }
    
    res.json(file);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching file', error: error.message });
  }
});

// Delete a file
fileRouter.delete('/:id', auth, async (req, res) => {
  try {
    const file = await File.findOneAndDelete({
      _id: req.params.id,
      uploadedBy: req.userId
    });
    
    if (!file) {
      return res.status(404).json({ message: 'File not found' });
    }
    
    // Delete embeddings from Qdrant
    try {
      const collectionName = 'file_embeddings';
      const pointIds = file.chunks.map((_, index) => `${file.filename.replace(/[^a-zA-Z0-9]/g, '_')}_${index}`);
      await deletePoints(collectionName, pointIds);
    } catch (error) {
      console.error('Error deleting embeddings from Qdrant:', error);
      // Continue with response even if Qdrant delete fails
    }
    
    res.json({ message: 'File deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: 'Error deleting file', error: error.message });
  }
});

// Q&A Routes
const qaRouter = express.Router();

// Ask a question about a file
qaRouter.post('/ask', auth, async (req, res) => {
  try {
    const { fileId, question } = req.body;

    if (!fileId || !question) {
      return res.status(400).json({ message: 'File ID and question are required' });
    }

    // Get file content and chunks
    const file = await File.findOne({
      _id: fileId,
      uploadedBy: req.userId
    });

    if (!file) {
      return res.status(404).json({ message: 'File not found' });
    }

    if (!file.processed || !file.chunks || file.chunks.length === 0) {
      return res.status(400).json({ message: 'File has not been processed yet' });
    }

    // Generate embeddings for the question
    const questionEmbedding = await generateEmbeddings(question);

    // Find most relevant chunks using Qdrant vector similarity
    const collectionName = 'file_embeddings';
    
    // Search for similar vectors in Qdrant
    const searchResults = await searchSimilarVectors(collectionName, questionEmbedding, 3);
    
    // Transform Qdrant results to match the expected format
    const relevantChunks = searchResults.map(result => ({
      text: result.payload.text,
      similarity: result.score
    }));

    // Combine relevant chunks into context
    const context = relevantChunks
      .map(chunk => chunk.text)
      .join('\n');

    // Use OpenAI to generate an answer based on the context
    const completion = await openai.createChatCompletion({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant that answers questions based on the provided context.'
        },
        {
          role: 'user',
          content: `Context: ${context}\n\nQuestion: ${question}\n\nAnswer the question based on the context provided. If the answer cannot be found in the context, say "I cannot find a relevant answer in the provided context."` 
        }
      ],
      temperature: 0.7,
      max_tokens: 500
    });

    const answer = completion.data.choices[0].message.content;

    // Save the question and answer
    const qa = new Question({
      question,
      questionEmbedding,
      answer,
      fileId,
      askedBy: req.userId,
      chunks: relevantChunks.map(chunk => ({
        text: chunk.text,
        embedding: chunk.embedding
      })),
      relevanceScore: relevantChunks[0]?.similarity || 0
    });

    await qa.save();

    res.json({
      question,
      answer,
      relevantChunks: relevantChunks.map(chunk => ({
        text: chunk.text,
        similarity: chunk.similarity
      }))
    });
  } catch (error) {
    res.status(500).json({ message: 'Error processing question', error: error.message });
  }
});

// Get question history for a file
qaRouter.get('/history/:fileId', auth, async (req, res) => {
  try {
    const questions = await Question.find({
      fileId: req.params.fileId,
      askedBy: req.userId
    })
      .select('question answer createdAt relevanceScore')
      .sort({ createdAt: -1 });
    
    res.json(questions);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching question history', error: error.message });
  }
});

// =====================================================================
// EXPRESS SERVER SETUP
// =====================================================================

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection with optimized settings
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/qa-file-processor';
const mongooseOptions = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
  maxPoolSize: 10,
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000,
};

mongoose.connect(MONGODB_URI, mongooseOptions)
  .then(() => console.log('Connected to MongoDB with optimized settings'))
  .catch(err => console.error('MongoDB connection error:', err));

// Routes
app.use('/api/auth', authRouter);
app.use('/api/files', fileRouter);
app.use('/api/qa', qaRouter);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: 'Something went wrong!' });
});

// =====================================================================
// QDRANT TEST CLASS
// =====================================================================

/**
 * Test class for Qdrant CRUD operations
 */
class QdrantTest {
  constructor() {
    this.testCollectionName
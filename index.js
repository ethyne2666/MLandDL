
import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';


// PDF load karne ka


async function indexDocument(){

    const PDF_PATH = './sih.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();

console.log("PDF loaded");


// Chuncking 

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

console.log("Chuncking Completed")






// vector Embedding model

const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });


console.log("Embedding model configured")






// configure Pinecone vector database

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);


console.log("Pine cone configured")





// langchain (chuncking , embedding , database)

await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });



  console.log("Data Stored in Pinecone Vector Database Successfully");



  
}


indexDocument();
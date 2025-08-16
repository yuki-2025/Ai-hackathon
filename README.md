# Ai-hackathon - Second place Winner
Ai-hackathon for Uchicago DSI

We built Genesis.AI, a retrieval-augmented chatbot that securely accesses patient records and answers general medical questions. To enable a seamless, end-to-end medical Q&A chatbot using entirely open-source tools. Our stack includes:

- **LangChain** for orchestration
- **PostgreSQL** (with vector embeddings) as both relational and vector databases
- **Streamlit** for the front-end
- **Google Cloud Vertex AI** to host and fine-tune **Llama 3-8B**

Key innovations:

1. **Dynamic SQL Generation**
    - Fine-tuned Llama 3-8B to translate natural-language medical queries into SQL on the fly.
    - Executes those SQL queries against patient tables and metadata.
2. **Vector Retrieval**
    - Indexes medical reports in PostgreSQL’s vector store.
    - Uses the same SQL-driven pipeline to retrieve relevant unstructured data.
3. **Accurate Response Generation**
    - Further fine-tuned Llama 3-8B to synthesize structured and vector-retrieved information into precise, context-aware answers.

![image](https://github.com/yuki-2025/Ai-hackathon/assets/159591455/fbc29a05-dfaa-4df8-8313-7be4d43e9777)


Demo:

![Video_2024-07-07_222911](https://github.com/yuki-2025/Ai-hackathon/assets/159591455/a4e1f4cb-c6e5-463e-9a16-3d900e8922f4)
 
![Video_2024-07-07_223634](https://github.com/yuki-2025/Ai-hackathon/assets/159591455/32d0b12a-10e3-483e-bb17-105cfeeccfed)

![WhatsApp-Video-2024-07-07-at-11 50 22-PM](https://github.com/yuki-2025/Ai-hackathon/assets/159591455/a1b70f96-b393-4688-b540-7cb556dc5340)

![WhatsApp-Video-2024-07-07-at-11 50 24-PM](https://github.com/yuki-2025/Ai-hackathon/assets/159591455/0da89f3e-e715-48ee-ae4c-5be80af64eb3)
